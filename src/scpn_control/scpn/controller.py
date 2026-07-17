# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neuro-Symbolic Controller
"""
Neuro-Symbolic Controller — oracle + SC dual paths.

Loads a ``.scpnctl.json`` artifact and provides deterministic
``step(obs, k) → ControlAction`` with JSONL logging.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from scpn_control.core._validators import require_int

from .artifact import Artifact, validate_artifact
from .contracts import (
    ControlAction,
    ControlScales,
    ControlTargets,
    FeatureAxisSpec,
    _seed64,
    decode_action_vector,
    feature_error_components,
)
from .observation import AERControlObservation
from .runtime_safety_certificate import (
    CertificateReplayResult,
    ControllerRuntimeBinding,
    RuntimeTarget,
    assert_runtime_certificate_admissible,
)

FloatArray = NDArray[np.float64]
ControllerObservation = Mapping[str, float] | AERControlObservation

_HAS_RUST_SCPN_RUNTIME = False
_rust_dense_activations: Callable[[FloatArray, FloatArray], object] | None = None
_rust_marking_update: Callable[[FloatArray, FloatArray, FloatArray, FloatArray], object] | None = None
_rust_sample_firing: Callable[[FloatArray, int, int, bool], object] | None = None
_FAULT_INJECTION_ENV = "SCPN_ALLOW_CONTROLLER_FAULT_INJECTION"

try:
    _rust_runtime = cast(Any, import_module("scpn_control_rs"))  # pragma: no cover - optional Rust SCPN runtime path
    _rust_dense_activations = cast(  # pragma: no cover - optional Rust SCPN runtime path
        Callable[[FloatArray, FloatArray], object],
        _rust_runtime.scpn_dense_activations,
    )
    _rust_marking_update = cast(  # pragma: no cover - optional Rust SCPN runtime path
        Callable[[FloatArray, FloatArray, FloatArray, FloatArray], object],
        _rust_runtime.scpn_marking_update,
    )
    _rust_sample_firing = cast(  # pragma: no cover - optional Rust SCPN runtime path
        Callable[[FloatArray, int, int, bool], object],
        _rust_runtime.scpn_sample_firing,
    )
    _HAS_RUST_SCPN_RUNTIME = True  # pragma: no cover - optional Rust SCPN runtime path
except (ImportError, AttributeError, NameError):
    _HAS_RUST_SCPN_RUNTIME = False


def _resolve_jsonl_log_path(log_path: str, log_root: str | Path | None) -> Path:
    """Resolve a controller JSONL log path under an explicit writable root."""
    if log_root is None:
        raise ValueError("log_root is required when log_path is provided.")
    root = Path(log_root).expanduser().resolve(strict=False)
    if not root.exists() or not root.is_dir():
        raise ValueError("log_root must reference an existing directory.")

    candidate = Path(log_path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("log_path must resolve under log_root.") from exc
    if resolved.suffix != ".jsonl":
        raise ValueError("log_path must use a .jsonl suffix.")
    if resolved.exists() and not resolved.is_file():
        raise ValueError("log_path must reference a regular file.")
    return resolved


def _append_jsonl_record(path: Path, record: Mapping[str, object]) -> None:
    """Append one JSON record without following symlinks where supported."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ValueError("log_path must reference a regular file.") from exc
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")


def _canonical_json(value: object) -> str:
    """Return the canonical JSON representation used for controller digests."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _matrix_entries(data: Sequence[float], shape: Sequence[int]) -> list[list[int | float]]:
    """Return sorted non-zero dense matrix entries in sparse-coordinate form."""
    if len(shape) != 2:
        raise ValueError("artifact weight matrix shape must have two dimensions")
    rows = require_int("artifact weight matrix rows", shape[0], 1)
    cols = require_int("artifact weight matrix cols", shape[1], 1)
    if len(data) != rows * cols:
        raise ValueError("artifact weight matrix data length does not match its shape")
    entries: list[list[int | float]] = []
    for idx, value in enumerate(data):
        scalar = float(value)
        if scalar != 0.0:
            entries.append([idx // cols, idx % cols, scalar])
    return sorted(entries)


def _artifact_topology_digest(artifact: Artifact) -> str:
    """Return the runtime-certificate topology digest for a loaded artifact."""
    signature = {
        "places": [place.name for place in artifact.topology.places],
        "transitions": [
            {
                "name": transition.name,
                "threshold": float(transition.threshold),
                "delay_ticks": int(transition.delay_ticks),
            }
            for transition in artifact.topology.transitions
        ],
        "w_in": _matrix_entries(artifact.weights.w_in.data, artifact.weights.w_in.shape),
        "w_out": _matrix_entries(artifact.weights.w_out.data, artifact.weights.w_out.shape),
    }
    return hashlib.sha256(_canonical_json(signature).encode("utf-8")).hexdigest()


class NeuroSymbolicController:
    """Reference controller with oracle float and stochastic paths.

    Parameters
    ----------
    artifact : loaded ``.scpnctl.json`` artifact.
    seed_base : 64-bit base seed for deterministic stochastic execution.
    targets : control setpoint targets.
    scales : normalisation scales.
    allow_fault_injection : explicit opt-in for stochastic bit-flip fault
        injection. Nonzero ``sc_bitflip_rate`` is rejected unless this is true.
    runtime_safety_certificate : optional runtime-bound formal certificate.
        When provided, ``runtime_safety_binding``, ``runtime_safety_target``, and
        ``runtime_safety_replay`` are also required and must pass admission
        before the controller instance is usable.
    """

    def __init__(
        self,
        artifact: Artifact,
        seed_base: int,
        targets: ControlTargets,
        scales: ControlScales,
        sc_n_passes: int = 8,
        sc_bitflip_rate: float = 0.0,
        allow_fault_injection: bool = False,
        sc_binary_margin: float | None = None,
        sc_antithetic: bool = True,
        enable_oracle_diagnostics: bool = True,
        feature_axes: Sequence[FeatureAxisSpec] | None = None,
        runtime_profile: str = "adaptive",
        runtime_backend: str = "auto",
        allow_runtime_backend_fallback: bool = False,
        allow_legacy_runtime_backend_fallback: bool = False,
        rust_backend_min_problem_size: int = 1,
        sc_antithetic_chunk_size: int = 2048,
        runtime_safety_certificate: dict[str, Any] | None = None,
        runtime_safety_binding: ControllerRuntimeBinding | None = None,
        runtime_safety_target: RuntimeTarget | None = None,
        runtime_safety_replay: CertificateReplayResult | None = None,
    ) -> None:
        validate_artifact(artifact)
        self.artifact = artifact
        self.runtime_safety_certificate_payload: dict[str, Any] | None = None
        self.runtime_safety_certificate_sha256: str | None = None
        if any(
            item is not None
            for item in (
                runtime_safety_certificate,
                runtime_safety_binding,
                runtime_safety_target,
                runtime_safety_replay,
            )
        ):
            if (
                runtime_safety_certificate is None
                or runtime_safety_binding is None
                or runtime_safety_target is None
                or runtime_safety_replay is None
            ):
                raise ValueError(
                    "runtime safety certificate admission requires certificate, binding, target, and replay inputs"
                )
            artifact_topology_sha256 = _artifact_topology_digest(artifact)
            if runtime_safety_binding.petri_topology_sha256 != artifact_topology_sha256:
                raise ValueError("runtime safety certificate topology does not match the controller artifact")
            admitted = assert_runtime_certificate_admissible(
                runtime_safety_certificate,
                live_binding=runtime_safety_binding,
                live_runtime_target=runtime_safety_target,
                replay=runtime_safety_replay,
            )
            self.runtime_safety_certificate_payload = admitted
            self.runtime_safety_certificate_sha256 = cast(str, admitted["payload_sha256"])
        self.seed_base = int(seed_base)
        self.targets = targets
        self.scales = scales
        self._sc_n_passes = require_int("sc_n_passes", sc_n_passes, 1)
        self._sc_bitflip_rate = float(sc_bitflip_rate)
        if not np.isfinite(self._sc_bitflip_rate) or self._sc_bitflip_rate < 0.0 or self._sc_bitflip_rate > 1.0:
            raise ValueError("sc_bitflip_rate must be finite and in [0, 1].")
        self._allow_fault_injection = bool(allow_fault_injection)
        if self._sc_bitflip_rate > 0.0 and not self._allow_fault_injection:
            raise ValueError("sc_bitflip_rate > 0 requires allow_fault_injection=True.")
        if self._sc_bitflip_rate > 0.0 and os.environ.get(_FAULT_INJECTION_ENV) != "1":
            raise ValueError(f"sc_bitflip_rate > 0 requires {_FAULT_INJECTION_ENV}=1.")
        self._runtime_profile = runtime_profile.strip().lower()
        if self._runtime_profile not in {"adaptive", "deterministic", "traceable"}:
            raise ValueError("runtime_profile must be 'adaptive', 'deterministic', or 'traceable'")
        self._sc_antithetic = bool(sc_antithetic)
        self._enable_oracle_diagnostics = bool(enable_oracle_diagnostics)
        self._feature_axes = list(feature_axes) if feature_axes is not None else None
        self._runtime_backend_request = runtime_backend.strip().lower()
        if self._runtime_backend_request not in {"auto", "numpy", "rust"}:
            raise ValueError("runtime_backend must be 'auto', 'numpy', or 'rust'")
        if allow_runtime_backend_fallback and not allow_legacy_runtime_backend_fallback:
            raise ValueError(
                "allow_runtime_backend_fallback=True requires "
                "allow_legacy_runtime_backend_fallback=True; "
                "legacy runtime-backend fallback is disabled by default."
            )
        self._allow_runtime_backend_fallback = bool(allow_runtime_backend_fallback)
        self._allow_legacy_runtime_backend_fallback = bool(allow_legacy_runtime_backend_fallback)
        self._rust_backend_min_problem_size = require_int(
            "rust_backend_min_problem_size", rust_backend_min_problem_size, 1
        )
        self._sc_antithetic_chunk_size = require_int("sc_antithetic_chunk_size", sc_antithetic_chunk_size, 1)

        if self._feature_axes is not None:
            axes = list(self._feature_axes)
        else:
            axes = [
                FeatureAxisSpec(
                    obs_key="R_axis_m",
                    target=self.targets.R_target_m,
                    scale=self.scales.R_scale_m,
                    pos_key="x_R_pos",
                    neg_key="x_R_neg",
                ),
                FeatureAxisSpec(
                    obs_key="Z_axis_m",
                    target=self.targets.Z_target_m,
                    scale=self.scales.Z_scale_m,
                    pos_key="x_Z_pos",
                    neg_key="x_Z_neg",
                ),
            ]
        self._feature_axes_effective = axes
        self._axis_count = len(axes)
        self._axis_obs_keys = [axis.obs_key for axis in axes]
        self._axis_targets = np.asarray([axis.target for axis in axes], dtype=np.float64)
        self._axis_scales = np.asarray(
            [axis.scale if abs(axis.scale) > 1e-12 else 1e-12 for axis in axes],
            dtype=np.float64,
        )
        self._axis_pos_keys = [axis.pos_key for axis in axes]
        self._axis_neg_keys = [axis.neg_key for axis in axes]
        self._empty = np.zeros(0, dtype=np.float64)
        self._tmp_obs_vals = np.zeros(self._axis_count, dtype=np.float64)
        self._tmp_feature_pos = np.zeros(self._axis_count, dtype=np.float64)
        self._tmp_feature_neg = np.zeros(self._axis_count, dtype=np.float64)

        # Flatten weight matrices for fast indexing
        self._w_in = artifact.weights.w_in.data[:]
        self._w_out = artifact.weights.w_out.data[:]
        self._nP = artifact.nP
        self._nT = artifact.nT
        self._W_in = np.asarray(self._w_in, dtype=np.float64).reshape(self._nT, self._nP)
        self._W_out = np.asarray(self._w_out, dtype=np.float64).reshape(self._nP, self._nT)
        self._W_in_t = self._W_in.T
        self._tmp_activations = np.zeros(self._nT, dtype=np.float64)
        self._tmp_consumption = np.zeros(self._nP, dtype=np.float64)
        self._tmp_production = np.zeros(self._nP, dtype=np.float64)
        self._tmp_marking_oracle = np.zeros(self._nP, dtype=np.float64)
        self._tmp_marking_sc = np.zeros(self._nP, dtype=np.float64)
        self._tmp_marking_input = np.zeros(self._nP, dtype=np.float64)
        self._tmp_sc_counts = np.zeros(self._nT, dtype=np.int64)
        self._thresholds = np.asarray([tr.threshold for tr in artifact.topology.transitions], dtype=np.float64)
        self._delay_ticks = np.asarray(
            [max(int(getattr(tr, "delay_ticks", 0)), 0) for tr in artifact.topology.transitions],
            dtype=np.int64,
        )
        self._delay_immediate_idx = np.flatnonzero(self._delay_ticks == 0).astype(np.int64, copy=False)
        self._delay_delayed_idx = np.flatnonzero(self._delay_ticks > 0).astype(np.int64, copy=False)
        if self._delay_delayed_idx.size:
            self._delay_delayed_offsets = np.asarray(self._delay_ticks[self._delay_delayed_idx], dtype=np.int64)
            self._tmp_delay_slots = np.zeros(self._delay_delayed_idx.size, dtype=np.int64)
        else:
            self._delay_delayed_offsets = np.zeros(0, dtype=np.int64)
            self._tmp_delay_slots = np.zeros(0, dtype=np.int64)
        self._max_delay_ticks = int(np.max(self._delay_ticks)) if self._delay_ticks.size else 0
        pending_len = self._max_delay_ticks + 1
        self._oracle_pending = np.zeros((pending_len, self._nT), dtype=np.float64)
        self._sc_pending = np.zeros((pending_len, self._nT), dtype=np.float64)
        self._oracle_cursor = 0
        self._sc_cursor = 0
        self._firing_mode = artifact.meta.firing_mode
        default_margin = float(artifact.meta.firing_margin)
        self._margins = np.asarray(
            [float(tr.margin if tr.margin is not None else default_margin) for tr in artifact.topology.transitions],
            dtype=np.float64,
        )
        if sc_binary_margin is None:
            if self._runtime_profile == "adaptive":
                self._sc_binary_margin = 0.05
            else:
                self._sc_binary_margin = 0.0
        else:
            self._sc_binary_margin = float(sc_binary_margin)
            if not np.isfinite(self._sc_binary_margin) or self._sc_binary_margin < 0.0:
                raise ValueError("sc_binary_margin must be finite and >= 0.")

        problem_size = int(self._nP * self._nT)
        rust_eligible = _HAS_RUST_SCPN_RUNTIME and (problem_size >= self._rust_backend_min_problem_size)
        if self._runtime_backend_request == "numpy":
            self._runtime_backend = "numpy"
        elif self._runtime_backend_request == "rust":
            if _HAS_RUST_SCPN_RUNTIME:
                self._runtime_backend = "rust"
            elif self._allow_runtime_backend_fallback:
                self._runtime_backend = "numpy"
            else:
                raise RuntimeError(
                    "runtime_backend='rust' requested but Rust SCPN runtime is unavailable. "
                    "Install scpn_control_rs, use runtime_backend='numpy', or set "
                    "allow_runtime_backend_fallback=True and "
                    "allow_legacy_runtime_backend_fallback=True for explicit "
                    "degraded-mode operation."
                )
        else:
            if rust_eligible:
                self._runtime_backend = "rust"
            else:
                # Auto mode is an explicit backend-selection policy:
                # use Rust when available/eligible, otherwise use NumPy.
                self._runtime_backend = "numpy"
        produced_feature_keys = set(self._axis_pos_keys)
        produced_feature_keys.update(self._axis_neg_keys)
        passthrough_sources: list[str] = []
        for inj in self.artifact.initial_state.place_injections:
            src = inj.source
            if src not in produced_feature_keys and src not in passthrough_sources:
                passthrough_sources.append(src)
        self._passthrough_sources = passthrough_sources
        self._traceable_ready = len(self._passthrough_sources) == 0
        key_to_axis: dict[str, tuple[int, bool]] = {}
        for i, key in enumerate(self._axis_pos_keys):
            key_to_axis[key] = (i, True)
        for i, key in enumerate(self._axis_neg_keys):
            key_to_axis[key] = (i, False)

        # Live state
        self._marking = np.asarray(artifact.initial_state.marking, dtype=np.float64).copy()
        injections = artifact.initial_state.place_injections
        self._inj_sources = [inj.source for inj in injections]
        self._inj_count = len(self._inj_sources)
        self._inj_place_ids = np.asarray([inj.place_id for inj in injections], dtype=np.int64)
        self._inj_scales = np.asarray([inj.scale for inj in injections], dtype=np.float64)
        self._inj_offsets = np.asarray([inj.offset for inj in injections], dtype=np.float64)
        self._inj_clamp_mask = np.asarray([bool(inj.clamp_0_1) for inj in injections], dtype=np.bool_)
        self._inj_clamp_idx = np.flatnonzero(self._inj_clamp_mask)
        self._inj_has_clamp = bool(self._inj_clamp_idx.size)
        self._inj_source_axis_idx = np.full(self._inj_count, -1, dtype=np.int64)
        self._inj_source_axis_pos = np.zeros(self._inj_count, dtype=np.bool_)
        self._tmp_inj_values = np.zeros(self._inj_count, dtype=np.float64)
        passthrough_pairs: list[tuple[int, str]] = []
        for i, src in enumerate(self._inj_sources):
            axis_info = key_to_axis.get(src)
            if axis_info is not None:
                axis_idx, is_pos = axis_info
                self._inj_source_axis_idx[i] = int(axis_idx)
                self._inj_source_axis_pos[i] = bool(is_pos)
            else:
                passthrough_pairs.append((i, src))
        self._inj_passthrough_pairs = passthrough_pairs

        self._action_names = [a.name for a in artifact.readout.actions]
        self._action_pos_idx = np.asarray([a.pos_place for a in artifact.readout.actions], dtype=np.int64)
        self._action_neg_idx = np.asarray([a.neg_place for a in artifact.readout.actions], dtype=np.int64)
        self._action_gains = np.asarray(artifact.readout.gains, dtype=np.float64)
        self._action_abs_max = np.asarray(artifact.readout.abs_max, dtype=np.float64)
        self._action_slew_per_s = np.asarray(artifact.readout.slew_per_s, dtype=np.float64)
        self._action_count = len(self._action_names)
        self._dt = float(artifact.meta.dt_control_s)
        self._prev_actions = np.zeros(self._action_count, dtype=np.float64)
        self._tmp_actions = np.zeros(self._action_count, dtype=np.float64)
        self.last_oracle_firing: list[float] = []
        self.last_sc_firing: list[float] = []
        self.last_oracle_marking: list[float] = self._marking.tolist()
        self.last_sc_marking: list[float] = self._marking.tolist()

    # ── Public API ───────────────────────────────────────────────────────

    def reset(self) -> None:
        """Restore initial marking and zero previous actions."""
        np.copyto(
            self._marking,
            np.asarray(self.artifact.initial_state.marking, dtype=np.float64),
        )
        self._prev_actions.fill(0.0)
        self._oracle_pending.fill(0.0)
        self._sc_pending.fill(0.0)
        self._oracle_cursor = 0
        self._sc_cursor = 0
        self.last_oracle_firing = []
        self.last_sc_firing = []
        self.last_oracle_marking = self._marking.tolist() if self._enable_oracle_diagnostics else []
        self.last_sc_marking = self._marking.tolist()

    @property
    def runtime_backend_name(self) -> str:
        """Name of the active controller runtime backend."""
        return self._runtime_backend

    @property
    def runtime_profile_name(self) -> str:
        """Name of the active controller runtime profile."""
        return self._runtime_profile

    @property
    def runtime_safety_admitted(self) -> bool:
        """Whether a runtime safety certificate was admitted for this instance."""
        return self.runtime_safety_certificate_payload is not None

    @property
    def marking(self) -> list[float]:
        """Current Petri-net marking as a list of per-place token densities."""
        return cast(list[float], self._marking.tolist())

    @marking.setter
    def marking(self, values: Sequence[float]) -> None:
        arr = np.asarray(list(values), dtype=np.float64)
        if arr.shape != (self._nP,):
            raise ValueError(f"marking must have length {self._nP}, got {arr.size}")
        # np.clip does not sanitise NaN, so a non-finite value would otherwise
        # pass straight into the controller state and poison every later step.
        if not bool(np.isfinite(arr).all()):
            raise ValueError("marking values must be finite")
        self._marking = np.clip(arr, 0.0, 1.0)

    def step(
        self,
        obs: ControllerObservation,
        k: int,
        log_path: str | None = None,
        log_root: str | Path | None = None,
    ) -> ControlAction:
        """Execute one control tick.

        ``obs`` accepts the existing mapping contract or a typed
        ``AERControlObservation`` decoded through the same feature-injection
        path.

        Steps:
            1. ``extract_features(obs)`` → 4 unipolar features
            2. ``_inject_places(features)``
            3. ``_oracle_step()`` — float path (optional)
            4. ``_sc_step(k)`` — deterministic stochastic path
            5. ``_decode_actions()`` — gain × differencing, slew + abs clamp
            6. Optional JSONL logging
        """
        t0 = time.perf_counter()
        obs_map, aer_admission = self._normalise_observation(obs)

        # 1. Feature extraction (fast compiled mapping)
        pos_vals, neg_vals = self._compute_feature_components(obs_map)
        feats = self._build_feature_dict(obs_map, pos_vals, neg_vals) if log_path is not None else None

        # 2. Inject features into marking
        m = self._tmp_marking_input
        np.copyto(m, self._marking)
        self._inject_places(m, obs_map, pos_vals, neg_vals)

        # 3. Oracle float path (optional)
        if self._enable_oracle_diagnostics:
            f_oracle, m_oracle = self._oracle_step(m)
        else:
            f_oracle = np.asarray([], dtype=np.float64)
            m_oracle = np.asarray([], dtype=np.float64)

        # 4. Stochastic path
        f_sc, m_sc = self._sc_step(m, k)

        # Diagnostics (used by deterministic benchmark gates)
        self.last_oracle_firing = f_oracle.tolist()
        self.last_sc_firing = f_sc.tolist()
        self.last_oracle_marking = m_oracle.tolist() if self._enable_oracle_diagnostics else []
        self.last_sc_marking = m_sc.tolist()

        # Commit SC state
        np.copyto(self._marking, m_sc)

        # 5. Decode actions
        actions_dict = self._decode_actions(m_sc)

        t1 = time.perf_counter()

        # 6. Optional JSONL logging
        if log_path is not None:
            safe_log_path = _resolve_jsonl_log_path(log_path, log_root)
            rec = {
                "k": int(k),
                "obs": dict(obs_map),
                "features": feats,
                "aer_admission": aer_admission,
                "f_oracle": f_oracle.tolist(),
                "f_sc": f_sc.tolist(),
                "marking": m_sc.tolist(),
                "actions": actions_dict,
                "timing_ms": (t1 - t0) * 1000.0,
            }
            _append_jsonl_record(safe_log_path, rec)

        # Build result from all decoded actions
        return cast(ControlAction, dict(actions_dict))

    def step_traceable(
        self,
        obs_vector: Sequence[float],
        k: int,
        log_path: str | None = None,
        log_root: str | Path | None = None,
    ) -> FloatArray:
        """Execute one control tick from a fixed-order observation vector.

        The vector order is ``self._axis_obs_keys``. This avoids per-step key
        lookups/dict allocation in tight control loops.
        """
        if not self._traceable_ready:
            raise RuntimeError("step_traceable requires axis-only injections (no passthrough sources)")

        t0 = time.perf_counter()
        pos_vals, neg_vals = self._compute_feature_components_vector(obs_vector)

        m = self._tmp_marking_input
        np.copyto(m, self._marking)
        self._inject_places(m, {}, pos_vals, neg_vals)

        if self._enable_oracle_diagnostics:
            f_oracle, m_oracle = self._oracle_step(m)
        else:
            f_oracle = np.asarray([], dtype=np.float64)
            m_oracle = np.asarray([], dtype=np.float64)

        f_sc, m_sc = self._sc_step(m, k)

        self.last_oracle_firing = f_oracle.tolist()
        self.last_sc_firing = f_sc.tolist()
        self.last_oracle_marking = m_oracle.tolist() if self._enable_oracle_diagnostics else []
        self.last_sc_marking = m_sc.tolist()

        np.copyto(self._marking, m_sc)
        actions_vec = np.asarray(self._decode_actions_vector(m_sc), dtype=np.float64).copy()

        t1 = time.perf_counter()
        if log_path is not None:
            safe_log_path = _resolve_jsonl_log_path(log_path, log_root)
            obs_payload = {key: float(value) for key, value in zip(self._axis_obs_keys, obs_vector)}
            rec = {
                "k": int(k),
                "obs": obs_payload,
                "features": self._build_feature_dict(obs_payload, pos_vals, neg_vals),
                "f_oracle": f_oracle.tolist(),
                "f_sc": f_sc.tolist(),
                "marking": m_sc.tolist(),
                "actions": {name: float(actions_vec[i]) for i, name in enumerate(self._action_names)},
                "timing_ms": (t1 - t0) * 1000.0,
            }
            _append_jsonl_record(safe_log_path, rec)

        return actions_vec

    # ── Internal ─────────────────────────────────────────────────────────

    def _normalise_observation(
        self, obs: ControllerObservation
    ) -> tuple[Mapping[str, float], dict[str, int | bool] | None]:
        if isinstance(obs, AERControlObservation):
            return obs.to_feature_mapping(), obs.admission_report()
        return obs, None

    def _compute_feature_components(self, obs_map: Mapping[str, float]) -> tuple[FloatArray, FloatArray]:
        if self._axis_count == 0:
            return self._empty, self._empty

        obs_vals = self._tmp_obs_vals
        for i, key in enumerate(self._axis_obs_keys):
            if key not in obs_map:
                raise KeyError(f"Missing observation key for feature extraction: {key}")
            obs_vals[i] = float(obs_map[key])
        return self._compute_feature_components_vector(obs_vals)

    def _compute_feature_components_vector(
        self, obs_vector: Sequence[float] | FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        if self._axis_count == 0:
            return self._empty, self._empty

        obs_vals = np.asarray(obs_vector, dtype=np.float64)
        if obs_vals.shape != (self._axis_count,):
            raise ValueError(f"obs_vector must have length {self._axis_count}, got {obs_vals.size}")
        return feature_error_components(
            obs_vals,
            self._axis_targets,
            self._axis_scales,
            axis_names=self._axis_obs_keys,
            out_pos=self._tmp_feature_pos,
            out_neg=self._tmp_feature_neg,
        )

    def _build_feature_dict(
        self, obs_map: Mapping[str, float], pos_vals: FloatArray, neg_vals: FloatArray
    ) -> dict[str, float]:
        feats: dict[str, float] = {}
        for i, key in enumerate(self._axis_pos_keys):
            feats[key] = float(pos_vals[i])
        for i, key in enumerate(self._axis_neg_keys):
            feats[key] = float(neg_vals[i])
        for key in self._passthrough_sources:
            if key not in obs_map:
                raise KeyError(f"Missing observation key for passthrough: {key}")
            feats[key] = float(np.clip(obs_map[key], 0.0, 1.0))
        return feats

    def _inject_places(
        self,
        marking: FloatArray,
        obs_map: Mapping[str, float],
        pos_vals: FloatArray,
        neg_vals: FloatArray,
    ) -> None:
        """Write features into a marking vector via place_injections config."""
        if self._inj_count == 0:
            return

        values = self._tmp_inj_values
        values.fill(0.0)
        axis_mask = self._inj_source_axis_idx >= 0
        if np.any(axis_mask):
            axis_indices = self._inj_source_axis_idx[axis_mask]
            axis_pos = self._inj_source_axis_pos[axis_mask]
            values[axis_mask] = np.where(
                axis_pos,
                pos_vals[axis_indices],
                neg_vals[axis_indices],
            )
        for idx, key in self._inj_passthrough_pairs:
            if key not in obs_map:
                raise KeyError(f"Missing observation key for passthrough: {key}")
            # Match the public contract (extract_features): a non-finite passthrough
            # observation is rejected, not silently clipped. np.clip(nan) is nan, so
            # without this the live path would inject a poisoned marking the contract
            # forbids.
            value = float(obs_map[key])
            if not np.isfinite(value):
                raise ValueError(f"Passthrough observation value must be finite: {key}")
            values[idx] = float(np.clip(value, 0.0, 1.0))

        np.multiply(values, self._inj_scales, out=values)
        values += self._inj_offsets
        if self._inj_has_clamp:
            values[self._inj_clamp_idx] = np.clip(values[self._inj_clamp_idx], 0.0, 1.0)
        marking[self._inj_place_ids] = values

    def _oracle_step(self, marking: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Float-path Petri step.

        Returns (firing_vector, next_marking).
        """
        # Activation: a = W_in @ m
        a = self._dense_activations(marking)

        # Firing decision
        if self._firing_mode == "fractional":
            margins = np.maximum(self._margins, 1e-12)
            f = np.clip((a - self._thresholds) / margins, 0.0, 1.0)
        else:
            f = (a >= self._thresholds).astype(np.float64)

        f_timed, self._oracle_cursor = self._apply_transition_timing(f, self._oracle_pending, self._oracle_cursor)

        # Marking update: m' = clip(m - W_in^T @ f + W_out @ f, 0, 1)
        m2 = self._marking_update(marking, f_timed, self._tmp_marking_oracle)

        return f_timed, m2

    def _sc_step(self, marking: FloatArray, k: int) -> tuple[FloatArray, FloatArray]:
        """Deterministic stochastic path with optional bit-flip fault injection."""
        a = self._dense_activations(marking)

        if self._firing_mode == "fractional":
            margins = np.maximum(self._margins, 1e-12)
            p_fire = np.clip((a - self._thresholds) / margins, 0.0, 1.0)
        else:
            if self._sc_binary_margin > 0.0:
                # Optional smooth probability around threshold for binary mode.
                p_fire = np.clip(
                    0.5 + 0.5 * ((a - self._thresholds) / self._sc_binary_margin),
                    0.0,
                    1.0,
                )
            else:
                # Binary mode keeps exact threshold semantics for stability.
                p_fire = (a >= self._thresholds).astype(np.float64)

        if self._sc_n_passes <= 1 or (self._firing_mode == "binary" and self._sc_binary_margin <= 0.0):
            f = p_fire
            rng = None
        else:
            sample_seed = _seed64(self.seed_base, f"sc_step:{int(k)}")
            if self._runtime_backend == "rust" and _HAS_RUST_SCPN_RUNTIME and _rust_sample_firing is not None:
                sampled = _rust_sample_firing(
                    p_fire,
                    int(self._sc_n_passes),
                    int(sample_seed),
                    bool(self._sc_antithetic),
                )
                f = np.asarray(sampled, dtype=np.float64)
                if self._sc_bitflip_rate > 0.0:
                    # rust-sampling + fault-injection bitflip RNG seed; the numpy-path
                    # equivalent and the no-bitflip branch are covered. Not reliably reached on
                    # the CI rust build (verified in .venv / the rust-python-interop job).
                    rng = np.random.default_rng(
                        _seed64(self.seed_base, f"sc_flip:{int(k)}")
                    )  # pragma: no cover - optional Rust SCPN runtime path
                else:
                    rng = None
            else:
                rng = np.random.default_rng(sample_seed)
                counts = self._tmp_sc_counts
                if self._sc_antithetic and self._sc_n_passes >= 2:
                    n_pairs = (self._sc_n_passes + 1) // 2
                    counts.fill(0)
                    if self._nT <= self._sc_antithetic_chunk_size:
                        base = rng.random((n_pairs, self._nT))
                        low_hits = np.sum(base < p_fire[None, :], axis=0, dtype=np.int64)
                        if self._sc_n_passes % 2 == 0:
                            high_hits = np.sum(base > (1.0 - p_fire)[None, :], axis=0, dtype=np.int64)
                        else:
                            high_hits = np.sum(
                                base[:-1, :] > (1.0 - p_fire)[None, :],
                                axis=0,
                                dtype=np.int64,
                            )
                        counts[:] = np.asarray(low_hits + high_hits, dtype=np.int64)
                    else:
                        for start in range(0, self._nT, self._sc_antithetic_chunk_size):
                            end = min(start + self._sc_antithetic_chunk_size, self._nT)
                            p_chunk = p_fire[start:end]
                            base = rng.random((n_pairs, end - start))
                            low_hits = np.sum(
                                base < p_chunk[None, :],
                                axis=0,
                                dtype=np.int64,
                            )
                            if self._sc_n_passes % 2 == 0:
                                high_hits = np.sum(
                                    base > (1.0 - p_chunk)[None, :],
                                    axis=0,
                                    dtype=np.int64,
                                )
                            else:
                                high_hits = np.sum(
                                    base[:-1, :] > (1.0 - p_chunk)[None, :],
                                    axis=0,
                                    dtype=np.int64,
                                )
                            counts[start:end] = np.asarray(low_hits + high_hits, dtype=np.int64)
                else:
                    np.copyto(
                        counts,
                        np.asarray(
                            rng.binomial(self._sc_n_passes, p_fire, size=self._nT),
                            dtype=np.int64,
                        ),
                    )
                f = counts.astype(np.float64) / float(self._sc_n_passes)

        if self._sc_bitflip_rate > 0.0:
            if rng is None:
                rng = np.random.default_rng(_seed64(self.seed_base, f"sc_flip:{int(k)}"))
            f = self._apply_bit_flip_faults(f, rng)

        f_timed, self._sc_cursor = self._apply_transition_timing(f, self._sc_pending, self._sc_cursor)
        m2 = self._marking_update(marking, f_timed, self._tmp_marking_sc)
        if self._sc_bitflip_rate > 0.0:
            assert rng is not None
            m2 = self._apply_bit_flip_faults(m2, rng)

        return f_timed, m2

    def _apply_transition_timing(
        self,
        desired_firing: FloatArray,
        pending: FloatArray,
        cursor: int,
    ) -> tuple[FloatArray, int]:
        desired = np.asarray(np.clip(desired_firing, 0.0, 1.0), dtype=np.float64)
        if self._max_delay_ticks <= 0:
            return desired, cursor

        fired_now = np.asarray(pending[cursor], dtype=np.float64).copy()
        pending[cursor, :] = 0.0

        if self._delay_immediate_idx.size:
            idx = self._delay_immediate_idx
            fired_now[idx] = np.clip(fired_now[idx] + desired[idx], 0.0, 1.0)

        if self._delay_delayed_idx.size:  # pragma: no branch - L818 guard forces delayed_idx nonempty; #129
            np.add(cursor, self._delay_delayed_offsets, out=self._tmp_delay_slots)
            self._tmp_delay_slots %= pending.shape[0]
            idx = self._delay_delayed_idx
            pending[self._tmp_delay_slots, idx] = np.clip(pending[self._tmp_delay_slots, idx] + desired[idx], 0.0, 1.0)

        next_cursor = (cursor + 1) % pending.shape[0]
        return fired_now, next_cursor

    def _dense_activations(self, marking: FloatArray) -> FloatArray:
        if self._runtime_backend == "rust" and _HAS_RUST_SCPN_RUNTIME:
            assert _rust_dense_activations is not None
            out = _rust_dense_activations(self._W_in, marking)
            return np.asarray(out, dtype=np.float64)
        self._tmp_activations[:] = self._W_in @ marking
        return self._tmp_activations

    def _marking_update(self, marking: FloatArray, firing: FloatArray, out: FloatArray) -> FloatArray:
        if self._runtime_backend == "rust" and _HAS_RUST_SCPN_RUNTIME:
            assert _rust_marking_update is not None
            rust_out = _rust_marking_update(marking, self._W_in, self._W_out, firing)
            np.copyto(out, np.asarray(rust_out, dtype=np.float64))
            np.clip(out, 0.0, 1.0, out=out)
            return out
        self._tmp_consumption[:] = self._W_in_t @ firing
        self._tmp_production[:] = self._W_out @ firing
        out[:] = marking
        out -= self._tmp_consumption
        out += self._tmp_production
        np.clip(out, 0.0, 1.0, out=out)
        return out

    def _decode_actions(self, marking: FloatArray) -> dict[str, float]:
        actions = self._decode_actions_vector(marking)
        if actions.size == 0:
            return {}
        return {name: float(actions[i]) for i, name in enumerate(self._action_names)}

    def _decode_actions_vector(self, marking: FloatArray) -> FloatArray:
        if self._action_count == 0:
            return self._empty

        return decode_action_vector(
            marking,
            self._action_pos_idx,
            self._action_neg_idx,
            self._action_gains,
            self._action_abs_max,
            self._action_slew_per_s,
            self._dt,
            self._prev_actions,
            work=self._tmp_actions,
        )

    def _apply_bit_flip_faults(self, values: FloatArray, rng: np.random.Generator) -> FloatArray:
        """Inject bounded deterministic bit-flip faults into float vectors."""
        out = np.asarray(values, dtype=np.float64).copy()
        if self._sc_bitflip_rate <= 0.0 or out.size == 0:
            return out

        flips = rng.random(out.size) < self._sc_bitflip_rate
        if not np.any(flips):
            return out

        raw = out.view(np.uint64)
        flip_idx = np.flatnonzero(flips)
        bits = rng.integers(0, 52, size=flip_idx.size, dtype=np.uint64)
        masks = np.left_shift(np.uint64(1), bits)
        raw[flip_idx] ^= masks

        out = raw.view(np.float64)
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(out, 0.0, 1.0)
