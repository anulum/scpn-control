# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Controller contracts.
"""
Data contracts for the SCPN Fusion-Core Control API.

Observation / action TypedDicts, feature extraction (obs → unipolar [0,1]),
action decoding (marking → slew-limited actuator commands).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TypedDict

import numpy as np
import numpy.typing as npt

SCALE_FLOOR = 1e-12  # prevent division-by-zero in feature extraction
CONTROL_ERROR_CLAMP = 1.0  # symmetric saturation for normalized error
INVARIANT_CRITICAL_FRACTION = 0.20  # 20% deviation from threshold → "critical"

# ── Observation / Action TypedDicts ──────────────────────────────────────────


class ControlObservation(TypedDict):
    """Plant observation at a single control tick."""

    R_axis_m: float
    Z_axis_m: float


class ControlAction(TypedDict, total=False):
    """Actuator command output for a single control tick.

    Keys are dynamic and depend on the Petri Net readout configuration.
    """

    # Common keys for type hinting, but any key is allowed.
    dI_PF3_A: float
    dI_PF_topbot_A: float


# ── Targets / Scales ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ControlTargets:
    """Setpoint targets for the control loop."""

    R_target_m: float = 6.2
    Z_target_m: float = 0.0


@dataclass(frozen=True)
class ControlScales:
    """Normalisation scales (error → [-1, 1] range)."""

    R_scale_m: float = 0.5
    Z_scale_m: float = 0.5


@dataclass(frozen=True)
class FeatureAxisSpec:
    """Configurable feature axis mapping from observation -> unipolar features.

    Parameters
    ----------
    obs_key : observation key to read.
    target : target/setpoint value for this axis.
    scale : normalisation scale for signed error (target - obs) / scale.
    pos_key : output feature key for positive error component.
    neg_key : output feature key for negative error component.
    """

    obs_key: str
    target: float
    scale: float
    pos_key: str
    neg_key: str


# ── Helpers ──────────────────────────────────────────────────────────────────


def _clip01(v: float) -> float:
    """Clamp *v* to [0, 1]."""
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _seed64(seed_base: int, sid: str) -> int:
    """Deterministic seed derivation: sha256(seed_base || sid) → u64."""
    h = hashlib.sha256(f"{seed_base}:{sid}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False)


def feature_error_components(
    obs_values: Sequence[float] | npt.NDArray[np.float64],
    targets: Sequence[float] | npt.NDArray[np.float64],
    scales: Sequence[float] | npt.NDArray[np.float64],
    *,
    axis_names: Sequence[str] | None = None,
    out_pos: npt.NDArray[np.float64] | None = None,
    out_neg: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return positive and negative unipolar feature components.

    This is the shared signed-error kernel for the public contract helper and
    the live controller runtime. Each component is computed from
    ``(target - observation) / scale``, clamped to ``[-1, 1]``, then split into
    positive and negative ``[0, 1]`` channels.
    """
    obs_arr = np.asarray(obs_values, dtype=np.float64)
    target_arr = np.asarray(targets, dtype=np.float64)
    scale_arr = np.asarray(scales, dtype=np.float64)
    if obs_arr.ndim != 1:
        raise ValueError("obs_values must be one-dimensional")
    if target_arr.shape != obs_arr.shape or scale_arr.shape != obs_arr.shape:
        raise ValueError("obs_values, targets, and scales must have equal shapes")
    names = list(axis_names) if axis_names is not None else [f"axis_{idx}" for idx in range(obs_arr.size)]
    if len(names) != obs_arr.size:
        raise ValueError("axis_names must match obs_values length")

    _require_finite_axis_values("Observation value for feature extraction", obs_arr, names)
    _require_finite_axis_values("Feature axis target", target_arr, names)
    _require_finite_axis_values("Feature axis scale", scale_arr, names)

    # A feature scale is a normalisation magnitude; a negative scale inverts the
    # sign of (target - observation) / scale and would drive the controller the
    # wrong way. Reject it rather than silently produce an inverted feature. Zero
    # is left to the SCALE_FLOOR guard below (numerical safety, sign preserved).
    if bool(np.any(scale_arr < 0.0)):
        bad_index = int(np.flatnonzero(scale_arr < 0.0)[0])
        raise ValueError(f"Feature axis scale must be non-negative: {names[bad_index]}")

    pos = np.empty(obs_arr.shape, dtype=np.float64) if out_pos is None else out_pos
    neg = np.empty(obs_arr.shape, dtype=np.float64) if out_neg is None else out_neg
    if pos.shape != obs_arr.shape or neg.shape != obs_arr.shape:
        raise ValueError("out_pos and out_neg must match obs_values shape")

    safe_scales = np.where(np.abs(scale_arr) > SCALE_FLOOR, scale_arr, SCALE_FLOOR)
    np.subtract(target_arr, obs_arr, out=pos)
    np.divide(pos, safe_scales, out=pos)
    np.clip(pos, -CONTROL_ERROR_CLAMP, CONTROL_ERROR_CLAMP, out=pos)
    np.negative(pos, out=neg)
    np.clip(pos, 0.0, 1.0, out=pos)
    np.clip(neg, 0.0, 1.0, out=neg)
    return pos, neg


def _require_finite_axis_values(
    message_prefix: str,
    values: npt.NDArray[np.float64],
    axis_names: Sequence[str],
) -> None:
    if bool(np.all(np.isfinite(values))):
        return
    bad_index = int(np.flatnonzero(~np.isfinite(values))[0])
    raise ValueError(f"{message_prefix} must be finite: {axis_names[bad_index]}")


# ── Feature extraction ───────────────────────────────────────────────────────


def extract_features(
    obs: Mapping[str, float],
    targets: ControlTargets,
    scales: ControlScales,
    feature_axes: Sequence[FeatureAxisSpec] | None = None,
    passthrough_keys: Sequence[str] | None = None,
) -> dict[str, float]:
    """Map observation → unipolar [0, 1] feature sources.

    By default returns keys ``x_R_pos``, ``x_R_neg``, ``x_Z_pos``, ``x_Z_neg``.
    Custom feature mappings can be supplied via ``feature_axes`` for arbitrary
    observation dictionaries.
    """
    axes = (
        list(feature_axes)
        if feature_axes is not None
        else [
            FeatureAxisSpec(
                obs_key="R_axis_m",
                target=targets.R_target_m,
                scale=scales.R_scale_m,
                pos_key="x_R_pos",
                neg_key="x_R_neg",
            ),
            FeatureAxisSpec(
                obs_key="Z_axis_m",
                target=targets.Z_target_m,
                scale=scales.Z_scale_m,
                pos_key="x_Z_pos",
                neg_key="x_Z_neg",
            ),
        ]
    )

    obs_values: list[float] = []
    target_values: list[float] = []
    scale_values: list[float] = []
    axis_names: list[str] = []
    for axis in axes:
        if axis.obs_key not in obs:
            raise KeyError(f"Missing observation key for feature extraction: {axis.obs_key}")
        obs_values.append(float(obs[axis.obs_key]))
        target_values.append(float(axis.target))
        scale_values.append(float(axis.scale))
        axis_names.append(axis.obs_key)

    pos, neg = feature_error_components(
        obs_values,
        target_values,
        scale_values,
        axis_names=axis_names,
    )
    out: dict[str, float] = {}
    for idx, axis in enumerate(axes):
        out[axis.pos_key] = float(pos[idx])
        out[axis.neg_key] = float(neg[idx])

    if passthrough_keys is not None:
        for key in passthrough_keys:
            if key not in obs:
                raise KeyError(f"Missing observation key for passthrough: {key}")
            value = float(obs[key])
            if not math.isfinite(value):
                raise ValueError(f"Passthrough observation value must be finite: {key}")
            out[key] = _clip01(value)

    return out


# ── Action decoding ─────────────────────────────────────────────────────────


@dataclass
class ActionSpec:
    """One action channel: positive/negative place differencing."""

    name: str
    pos_place: int
    neg_place: int


def decode_actions(
    marking: list[float],
    actions_spec: list[ActionSpec],
    gains: list[float],
    abs_max: list[float],
    slew_per_s: list[float],
    dt: float,
    prev: list[float],
) -> dict[str, float]:
    """Decode marking → actuator commands with gain, slew-rate, and abs clamp.

    Parameters
    ----------
    marking : current marking vector (len >= max place index).
    actions_spec : per-action pos/neg place definitions.
    gains : per-action gain multiplier.
    abs_max : per-action absolute saturation.
    slew_per_s : per-action max change rate (units/s).
    dt : control tick period (s).
    prev : previous action outputs (same length as actions_spec).

    Returns
    -------
    dict mapping action name → clamped value.  Also mutates *prev* in-place.
    """
    n_actions = len(actions_spec)
    if len(gains) != n_actions or len(abs_max) != n_actions or len(slew_per_s) != n_actions or len(prev) != n_actions:
        raise ValueError("actions_spec, gains, abs_max, slew_per_s, and prev must have equal lengths.")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and > 0.")

    prev_array = np.asarray(prev, dtype=np.float64)
    decoded = decode_action_vector(
        marking,
        [spec.pos_place for spec in actions_spec],
        [spec.neg_place for spec in actions_spec],
        gains,
        abs_max,
        slew_per_s,
        dt,
        prev_array,
    )
    result: dict[str, float] = {}
    for idx, spec in enumerate(actions_spec):
        value = float(decoded[idx])
        prev[idx] = value
        result[spec.name] = value
    return result


def decode_action_vector(
    marking: Sequence[float] | npt.NDArray[np.float64],
    pos_places: Sequence[int] | npt.NDArray[np.int64],
    neg_places: Sequence[int] | npt.NDArray[np.int64],
    gains: Sequence[float] | npt.NDArray[np.float64],
    abs_max: Sequence[float] | npt.NDArray[np.float64],
    slew_per_s: Sequence[float] | npt.NDArray[np.float64],
    dt: float,
    prev: npt.NDArray[np.float64],
    *,
    work: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Decode marking to a vector of slew- and magnitude-limited actions.

    The returned array is ``prev`` after in-place update. ``work`` may be a
    caller-owned scratch array and is never returned.
    """
    marking_arr = np.asarray(marking, dtype=np.float64)
    pos_idx = np.asarray(pos_places, dtype=np.int64)
    neg_idx = np.asarray(neg_places, dtype=np.int64)
    gains_arr = np.asarray(gains, dtype=np.float64)
    abs_max_arr = np.asarray(abs_max, dtype=np.float64)
    slew_arr = np.asarray(slew_per_s, dtype=np.float64)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and > 0.")
    if marking_arr.ndim != 1:
        raise ValueError("marking must be one-dimensional")
    n_actions = pos_idx.size
    if (
        neg_idx.shape != (n_actions,)
        or gains_arr.shape != (n_actions,)
        or abs_max_arr.shape != (n_actions,)
        or slew_arr.shape != (n_actions,)
        or prev.shape != (n_actions,)
    ):
        raise ValueError("action index, gain, limit, slew, and prev arrays must have equal shapes")
    if n_actions == 0:
        return prev
    if int(np.min(pos_idx)) < 0 or int(np.min(neg_idx)) < 0:
        raise ValueError("Action place indices must be >= 0.")
    n_places = marking_arr.size
    if int(np.max(pos_idx)) >= n_places or int(np.max(neg_idx)) >= n_places:
        raise ValueError("Action place index out of bounds for marking vector.")

    # Fail closed on non-finite inputs before touching prev: np.clip does not
    # sanitise NaN, so a NaN marking place, gain, limit, slew, or prev would
    # otherwise propagate straight to the actuator command (and poison prev for
    # the next tick). A safe command cannot be computed from a non-finite input.
    if not (
        bool(np.isfinite(marking_arr[pos_idx]).all())
        and bool(np.isfinite(marking_arr[neg_idx]).all())
        and bool(np.isfinite(gains_arr).all())
        and bool(np.isfinite(abs_max_arr).all())
        and bool(np.isfinite(slew_arr).all())
        and bool(np.isfinite(prev).all())
    ):
        raise ValueError(
            "decode inputs must be finite: a non-finite marking place, gain, "
            "abs_max, slew, or prev cannot produce a safe actuator command"
        )

    raw = np.empty((n_actions,), dtype=np.float64) if work is None else work
    if raw.shape != (n_actions,):
        raise ValueError("work must match action count")
    np.subtract(marking_arr[pos_idx], marking_arr[neg_idx], out=raw)
    raw *= gains_arr
    max_delta = slew_arr * float(dt)
    np.clip(raw, prev - max_delta, prev + max_delta, out=raw)
    np.clip(raw, -abs_max_arr, abs_max_arr, out=prev)
    return prev


# ── Physics Invariants ──────────────────────────────────────────────────────
# Hard physics constraints that the controller loop must respect.
# Violations trigger disruption mitigation protocols.
# ────────────────────────────────────────────────────────────────────────────

_VALID_COMPARATORS = ("gt", "lt", "gte", "lte")


@dataclass(frozen=True)
class PhysicsInvariant:
    """A hard physics constraint the controller loop must respect.

    Parameters
    ----------
    name : str
        Short identifier for the invariant (e.g. ``"q_min"``, ``"beta_N"``).
    description : str
        Human-readable description including the physics origin.
    threshold : float
        Threshold value for the invariant condition.
    comparator : str
        One of ``"gt"``, ``"lt"``, ``"gte"``, ``"lte"`` — the relationship
        that the *measured value* must satisfy with respect to ``threshold``
        for the invariant to hold.
    """

    name: str
    description: str
    threshold: float
    comparator: str  # "gt" | "lt" | "gte" | "lte"

    def __post_init__(self) -> None:
        if self.comparator not in _VALID_COMPARATORS:
            raise ValueError(f"Invalid comparator {self.comparator!r}; must be one of {_VALID_COMPARATORS}")
        if not math.isfinite(self.threshold):
            raise ValueError("PhysicsInvariant threshold must be finite.")


@dataclass(frozen=True)
class PhysicsInvariantViolation:
    """Record of a physics invariant violation.

    Parameters
    ----------
    invariant : PhysicsInvariant
        The invariant that was violated.
    actual_value : float
        The measured/computed value that violated the invariant.
    margin : float
        Absolute distance between ``actual_value`` and the invariant
        ``threshold`` (always >= 0).
    severity : str
        ``"warning"`` if margin <= 20 % of |threshold|,
        ``"critical"`` otherwise.
    """

    invariant: PhysicsInvariant
    actual_value: float
    margin: float
    severity: str  # "warning" | "critical"


# ── Default tokamak physics invariants ──────────────────────────────────────

DEFAULT_PHYSICS_INVARIANTS: list[PhysicsInvariant] = [
    PhysicsInvariant(
        name="q_min",
        description=(
            "Kruskal-Shafranov MHD stability limit: the edge safety factor "
            "q must exceed 1.0 to avoid the m=1/n=1 external kink mode.  "
            "Ref: Kruskal & Schwarzschild (1954); Shafranov (1970)."
        ),
        threshold=1.0,
        comparator="gt",
    ),
    PhysicsInvariant(
        name="beta_N",
        description=(
            "Troyon no-wall beta limit: normalised beta β_N = β(%) · a(m) · B_T(T) / I_P(MA) "
            "must remain below ~2.8 to avoid resistive wall modes without a conducting wall.  "
            "Ref: Troyon et al., Plasma Phys. Control. Fusion 26 (1984) 209."
        ),
        threshold=2.8,
        comparator="lt",
    ),
    PhysicsInvariant(
        name="greenwald",
        description=(
            "Greenwald density limit: the line-averaged density normalised to "
            "n_GW = I_P / (π a²) must stay below ~1.2 to avoid radiative collapse "
            "and density-limit disruptions.  "
            "Ref: Greenwald, Plasma Phys. Control. Fusion 44 (2002) R27."
        ),
        threshold=1.2,
        comparator="lt",
    ),
    PhysicsInvariant(
        name="T_i",
        description=(
            "Ion temperature cap: T_i must remain below 25 keV to stay within "
            "the operating window of current first-wall / divertor materials and "
            "avoid excessive neutron wall-loading.  "
            "Ref: ITER Physics Basis, Nucl. Fusion 39 (1999) 2137."
        ),
        threshold=25.0,
        comparator="lt",
    ),
    PhysicsInvariant(
        name="energy_conservation_error",
        description=(
            "Energy bookkeeping: the fractional mismatch between injected, "
            "radiated, and stored energy must stay below 1 % to trust the "
            "simulation state.  Tolerance: |ΔW/W| < 0.01."
        ),
        threshold=0.01,
        comparator="lt",
    ),
]


# ── Invariant checking ──────────────────────────────────────────────────────


def _is_satisfied(comparator: str, value: float, threshold: float) -> bool:
    """Return True when *value* satisfies the *comparator* w.r.t. *threshold*."""
    if comparator == "gt":
        return value > threshold
    if comparator == "lt":
        return value < threshold
    if comparator == "gte":
        return value >= threshold
    if comparator == "lte":
        return value <= threshold
    raise ValueError(f"Unknown comparator: {comparator!r}")


def check_physics_invariant(
    invariant: PhysicsInvariant,
    value: float,
) -> PhysicsInvariantViolation | None:
    """Check a single physics invariant against a measured *value*.

    Returns ``None`` if the invariant is satisfied, otherwise returns a
    :class:`PhysicsInvariantViolation` with computed margin and severity.

    Severity classification
    -----------------------
    * ``"critical"`` — margin exceeds 20 % of ``|threshold|``
      (or 20 % of 1.0 when threshold == 0).
    * ``"warning"``  — violated but within the 20 % band.

    Parameters
    ----------
    invariant : PhysicsInvariant
        The invariant to check.
    value : float
        Current measured / computed value for the quantity.
    """
    if not math.isfinite(value):
        # Non-finite values always violate; treat as critical.
        return PhysicsInvariantViolation(
            invariant=invariant,
            actual_value=value,
            margin=float("inf"),
            severity="critical",
        )

    if _is_satisfied(invariant.comparator, value, invariant.threshold):
        return None

    margin = abs(value - invariant.threshold)
    ref = abs(invariant.threshold) if invariant.threshold != 0.0 else 1.0
    severity = "critical" if margin > INVARIANT_CRITICAL_FRACTION * ref else "warning"

    return PhysicsInvariantViolation(
        invariant=invariant,
        actual_value=value,
        margin=margin,
        severity=severity,
    )


def check_all_invariants(
    values: dict[str, float],
    invariants: list[PhysicsInvariant] | None = None,
) -> list[PhysicsInvariantViolation]:
    """Check every invariant whose name appears in *values*.

    Parameters
    ----------
    values : dict
        Mapping from invariant ``name`` to the current measured value.
        Names not present in the invariant list are silently ignored.
    invariants : list, optional
        The invariant set to check.  Defaults to
        :data:`DEFAULT_PHYSICS_INVARIANTS`.

    Returns
    -------
    list[PhysicsInvariantViolation]
        All detected violations (empty list when everything is nominal).
    """
    if invariants is None:
        invariants = DEFAULT_PHYSICS_INVARIANTS

    violations: list[PhysicsInvariantViolation] = []
    for inv in invariants:
        if inv.name in values:
            v = check_physics_invariant(inv, values[inv.name])
            if v is not None:
                violations.append(v)
    return violations


def evaluate_safety_invariants(
    values: dict[str, float],
    invariants: list[PhysicsInvariant] | None = None,
) -> list[PhysicsInvariantViolation]:
    """Fail-closed safety evaluation: every invariant channel is mandatory.

    Unlike :func:`check_all_invariants`, which only checks the invariants whose
    channel is present in *values* (a subset utility), this is the safety-monitor
    entry point: it treats every invariant in the set as a safety-critical channel
    that MUST be measured. A missing channel is reported as a ``"critical"``
    violation rather than silently skipped, so a sensor dropout — including the
    degenerate ``{}`` (all channels lost) — can never be mistaken for a nominal
    plasma. Present-but-non-finite channels are already critical via
    :func:`check_physics_invariant`.

    Parameters
    ----------
    values : dict
        Mapping from invariant ``name`` to the current measured value. Every
        invariant name that is absent yields a critical missing-channel violation.
    invariants : list, optional
        The safety-critical invariant set. Defaults to
        :data:`DEFAULT_PHYSICS_INVARIANTS`.

    Returns
    -------
    list[PhysicsInvariantViolation]
        All detected violations. A missing channel is recorded with
        ``actual_value`` NaN, infinite ``margin``, and ``"critical"`` severity.
        The list is empty only when every channel is present and nominal.
    """
    if invariants is None:
        invariants = DEFAULT_PHYSICS_INVARIANTS

    violations: list[PhysicsInvariantViolation] = []
    for inv in invariants:
        if inv.name not in values:
            violations.append(
                PhysicsInvariantViolation(
                    invariant=inv,
                    actual_value=float("nan"),
                    margin=float("inf"),
                    severity="critical",
                )
            )
            continue
        violation = check_physics_invariant(inv, values[inv.name])
        if violation is not None:
            violations.append(violation)
    return violations


def should_trigger_mitigation(
    violations: list[PhysicsInvariantViolation],
) -> bool:
    """Return ``True`` if any violation has ``severity == "critical"``.

    This is the top-level disruption-mitigation gate: a single critical
    violation means the controller must engage protective actions (e.g.
    massive gas injection, current quench, or safe ramp-down).
    """
    return any(v.severity == "critical" for v in violations)


# ── Petri Net Logic Invariants ──────────────────────────────────────────────
# Formal properties of the compiled Petri Net.
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LogicInvariant:
    """A formal property of the Petri Net state.

    Parameters
    ----------
    name : str
        Short identifier for the invariant (e.g. "marking_conservation").
    description : str
        Human-readable description of the formal property.
    """

    name: str
    description: str


@dataclass(frozen=True)
class LogicInvariantViolation:
    """Record of a logic invariant violation.

    Parameters
    ----------
    invariant : LogicInvariant
        The invariant that was violated.
    message : str
        Specific details about why the invariant failed.
    """

    invariant: LogicInvariant
    message: str


def check_marking_conservation(
    marking_before: list[float],
    marking_after: list[float],
) -> LogicInvariantViolation | None:
    """Verify that total token count is conserved.

    For conservative Petri Nets, Σ M_before == Σ M_after.
    """
    sum_before = float(sum(marking_before))
    sum_after = float(sum(marking_after))
    if abs(sum_before - sum_after) >= 1e-9:
        inv = LogicInvariant(
            name="marking_conservation",
            description="Total token count must be conserved in conservative nets.",
        )
        return LogicInvariantViolation(
            invariant=inv,
            message=f"Token sum mismatch: before={sum_before:.6e}, after={sum_after:.6e}",
        )
    return None


def check_deadlock(
    marking: list[float],
    incidence_matrix: npt.NDArray[Any],
    is_terminal: bool = False,
) -> LogicInvariantViolation | None:
    """Detect if the net is in a non-terminal deadlock state.

    A net is deadlocked if no transitions are enabled and it's not a
    configured terminal state.
    """
    if is_terminal:
        return None

    # A transition j is enabled if all input places i have M[i] >= W_minus[i,j]
    W_minus = np.maximum(0, -incidence_matrix)
    n_places, n_trans = incidence_matrix.shape

    enabled_count = 0
    for j in range(n_trans):
        enabled = True
        for i in range(n_places):
            if marking[i] < W_minus[i, j]:
                enabled = False
                break
        if enabled:
            enabled_count += 1

    if enabled_count == 0:
        inv = LogicInvariant(
            name="deadlock",
            description="No transitions enabled in a non-terminal state.",
        )
        return LogicInvariantViolation(
            invariant=inv,
            message=f"Non-terminal deadlock detected: marking={marking}",
        )
    return None


def check_boundedness(
    marking: list[float],
    max_capacity: float = 1e6,
) -> LogicInvariantViolation | None:
    """Verify that no place exceeds its maximum token capacity."""
    for i, m in enumerate(marking):
        if m > max_capacity:
            inv = LogicInvariant(
                name="boundedness",
                description="Place token count must not exceed capacity.",
            )
            return LogicInvariantViolation(
                invariant=inv,
                message=f"Place {i} exceeds capacity {max_capacity}: value={m}",
            )
    return None
