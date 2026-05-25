# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Formal Petri Net Verification

"""Formal verification utilities for SCPN Petri-net control logic.

The verifier performs exact bounded reachability over the compiled
``StochasticPetriNet`` transition relation. It is intended for certification
artefacts where randomized simulation is insufficient: every reported
counterexample includes the transition path and marking that violates the
property.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

from scpn_control.scpn.structure import StochasticPetriNet

FormalBackend = Literal["auto", "explicit-state", "z3"]


@dataclass(frozen=True)
class ReachableMarking:
    """A reachable marking and the transition path that reaches it."""

    marking: dict[str, float]
    path: list[str]


@dataclass(frozen=True)
class FormalViolation:
    """Actionable counterexample for a failed formal property."""

    property_name: str
    message: str
    marking: dict[str, float]
    path: list[str]
    place: str | None = None
    transition: str | None = None


@dataclass(frozen=True)
class ReachabilityReport:
    """Bounded reachability result for a compiled SCPN."""

    holds: bool
    backend: str
    max_depth: int
    reachable_states: list[ReachableMarking]
    fired_transitions: set[str]

    @property
    def reachable_count(self) -> int:
        return len(self.reachable_states)


@dataclass(frozen=True)
class FormalPropertyReport:
    """Result for a safety, liveness, or temporal-logic proof obligation."""

    holds: bool
    backend: str
    max_depth: int
    violations: list[FormalViolation] = field(default_factory=list)
    checked_specs: list[str] = field(default_factory=list)
    dead_transitions: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class FormalVerificationReport:
    """Combined formal contract report for reachability, safety, and liveness."""

    holds: bool
    reachability: ReachabilityReport
    safety: FormalPropertyReport
    liveness: FormalPropertyReport
    temporal: FormalPropertyReport


@dataclass(frozen=True)
class AlwaysBounded:
    """Temporal safety spec: all reachable markings keep named places in bounds."""

    name: str
    bounds: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class EventuallyFires:
    """Bounded liveness spec: a transition fires on at least one reachable path."""

    name: str
    transition: str


@dataclass(frozen=True)
class NeverCoMarked:
    """Temporal safety spec: two places are never simultaneously marked above threshold."""

    name: str
    place_a: str
    place_b: str
    threshold: float = 0.0


def _as_fraction(value: float) -> Fraction:
    return Fraction(str(float(value)))


def _as_float_marking(place_names: list[str], values: tuple[Fraction, ...]) -> dict[str, float]:
    return {place: float(values[i]) for i, place in enumerate(place_names)}


def _resolve_backend(backend: FormalBackend) -> str:
    if backend == "explicit-state":
        return backend
    if backend == "z3":
        try:
            import z3  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError("backend='z3' requires the optional z3-solver package") from exc
        return "z3"
    if backend == "auto":
        try:
            import z3  # noqa: F401
        except ModuleNotFoundError:
            return "explicit-state"
        return "z3"
    raise ValueError(f"unsupported formal verification backend: {backend}")


class FormalPetriNetVerifier:
    """Bounded formal verifier for compiled ``StochasticPetriNet`` objects.

    The current production backend is exact explicit-state reachability over
    rational markings. ``backend='z3'`` is accepted only when the optional solver
    package is installed; the same finite transition relation is then eligible
    for SMT-backed proof extensions without changing public contracts.
    """

    def __init__(self, net: StochasticPetriNet, *, backend: FormalBackend = "auto") -> None:
        if not net.is_compiled:
            raise RuntimeError("Net must be compiled before formal verification.")
        if net.W_in is None or net.W_out is None:
            raise RuntimeError("Net must be compiled before formal verification.")
        self.net = net
        self.backend = _resolve_backend(backend)
        self.place_names = net.place_names
        self.transition_names = net.transition_names
        self._place_index = {name: idx for idx, name in enumerate(self.place_names)}
        self._transition_index = {name: idx for idx, name in enumerate(self.transition_names)}
        self._initial = tuple(_as_fraction(value) for value in net.get_initial_marking())
        self._inputs, self._outputs, self._inhibitors = self._extract_transition_relation()

    def analyze_reachability(self, *, max_depth: int) -> ReachabilityReport:
        """Enumerate all markings reachable within ``max_depth`` firings."""
        states = self._reachable_state_map(max_depth=max_depth)
        reachable = [ReachableMarking(_as_float_marking(self.place_names, marking), path) for marking, path in states]
        fired = {transition for _marking, path in states for transition in path}
        return ReachabilityReport(
            holds=True,
            backend=self.backend,
            max_depth=max_depth,
            reachable_states=reachable,
            fired_transitions=fired,
        )

    def prove_marking_bounds(
        self,
        bounds: dict[str, tuple[float, float]],
        *,
        max_depth: int,
    ) -> FormalPropertyReport:
        """Prove that all reachable markings satisfy per-place bounds."""
        self._validate_bounds(bounds)
        violations: list[FormalViolation] = []
        fractional_bounds = {
            place: (_as_fraction(lower), _as_fraction(upper)) for place, (lower, upper) in bounds.items()
        }
        for marking, path in self._reachable_state_map(max_depth=max_depth):
            for place, (lower, upper) in fractional_bounds.items():
                value = marking[self._place_index[place]]
                if value < lower or value > upper:
                    violations.append(
                        FormalViolation(
                            property_name="marking_bounds",
                            message=f"place {place!r} reached {float(value):.12g} outside [{float(lower)}, {float(upper)}]",
                            marking=_as_float_marking(self.place_names, marking),
                            path=path,
                            place=place,
                        )
                    )
                    return FormalPropertyReport(
                        holds=False,
                        backend=self.backend,
                        max_depth=max_depth,
                        violations=violations,
                    )
        return FormalPropertyReport(holds=True, backend=self.backend, max_depth=max_depth)

    def prove_transition_liveness(self, *, max_depth: int) -> FormalPropertyReport:
        """Prove bounded transition liveness by checking every transition fires on some path."""
        reachability = self.analyze_reachability(max_depth=max_depth)
        dead = set(self.transition_names) - reachability.fired_transitions
        if not dead:
            return FormalPropertyReport(holds=True, backend=self.backend, max_depth=max_depth)
        violations = [
            FormalViolation(
                property_name="transition_liveness",
                message=f"transition {transition!r} never fires within depth {max_depth}",
                marking=dict(reachability.reachable_states[0].marking),
                path=[],
                transition=transition,
            )
            for transition in sorted(dead)
        ]
        return FormalPropertyReport(
            holds=False,
            backend=self.backend,
            max_depth=max_depth,
            violations=violations,
            dead_transitions=dead,
        )

    def verify_temporal_specs(
        self,
        specs: list[AlwaysBounded | EventuallyFires | NeverCoMarked],
        *,
        max_depth: int,
    ) -> FormalPropertyReport:
        """Verify bounded temporal safety/liveness specifications."""
        states = self._reachable_state_map(max_depth=max_depth)
        checked: list[str] = []
        violations: list[FormalViolation] = []

        for spec in specs:
            checked.append(spec.name)
            if isinstance(spec, AlwaysBounded):
                report = self.prove_marking_bounds(spec.bounds, max_depth=max_depth)
                if not report.holds:
                    first = report.violations[0]
                    violations.append(
                        FormalViolation(
                            property_name=spec.name,
                            message=first.message,
                            marking=first.marking,
                            path=first.path,
                            place=first.place,
                        )
                    )
                    break
            elif isinstance(spec, EventuallyFires):
                self._require_transition(spec.transition)
                if not any(spec.transition in path for _marking, path in states):
                    violations.append(
                        FormalViolation(
                            property_name=spec.name,
                            message=f"transition {spec.transition!r} never fires within depth {max_depth}",
                            marking=_as_float_marking(self.place_names, states[0][0]),
                            path=[],
                            transition=spec.transition,
                        )
                    )
                    break
            elif isinstance(spec, NeverCoMarked):
                self._require_place(spec.place_a)
                self._require_place(spec.place_b)
                threshold = _as_fraction(spec.threshold)
                ia = self._place_index[spec.place_a]
                ib = self._place_index[spec.place_b]
                for marking, path in states:
                    if marking[ia] > threshold and marking[ib] > threshold:
                        violations.append(
                            FormalViolation(
                                property_name=spec.name,
                                message=(
                                    f"places {spec.place_a!r} and {spec.place_b!r} are both above "
                                    f"{float(threshold):.12g}"
                                ),
                                marking=_as_float_marking(self.place_names, marking),
                                path=path,
                            )
                        )
                        break
                if violations:
                    break
            else:
                raise TypeError(f"unsupported temporal specification: {type(spec).__name__}")

        return FormalPropertyReport(
            holds=not violations,
            backend=self.backend,
            max_depth=max_depth,
            violations=violations,
            checked_specs=checked,
        )

    def _reachable_state_map(self, *, max_depth: int) -> list[tuple[tuple[Fraction, ...], list[str]]]:
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        visited: dict[tuple[Fraction, ...], list[str]] = {self._initial: []}
        queue: deque[tuple[tuple[Fraction, ...], list[str]]] = deque([(self._initial, [])])
        while queue:
            marking, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for transition in self.transition_names:
                next_marking = self._fire_if_enabled(marking, transition)
                if next_marking is None or next_marking in visited:
                    continue
                next_path = [*path, transition]
                visited[next_marking] = next_path
                queue.append((next_marking, next_path))
        return list(visited.items())

    def _fire_if_enabled(self, marking: tuple[Fraction, ...], transition: str) -> tuple[Fraction, ...] | None:
        t_idx = self._transition_index[transition]
        for p_idx, weight in self._inputs[t_idx].items():
            if marking[p_idx] < weight:
                return None
        for p_idx, threshold in self._inhibitors[t_idx].items():
            if marking[p_idx] >= threshold:
                return None
        updated = list(marking)
        for p_idx, weight in self._inputs[t_idx].items():
            updated[p_idx] -= weight
        for p_idx, weight in self._outputs[t_idx].items():
            updated[p_idx] += weight
        if any(value < 0 for value in updated):
            return None
        return tuple(updated)

    def _extract_transition_relation(
        self,
    ) -> tuple[list[dict[int, Fraction]], list[dict[int, Fraction]], list[dict[int, Fraction]]]:
        inputs: list[dict[int, Fraction]] = [dict() for _ in self.transition_names]
        outputs: list[dict[int, Fraction]] = [dict() for _ in self.transition_names]
        inhibitors: list[dict[int, Fraction]] = [dict() for _ in self.transition_names]

        for source, target, weight, is_inhibitor in self.net._arcs:  # noqa: SLF001 - verifier is a trusted SCPN peer.
            amount = _as_fraction(abs(weight))
            if source in self._place_index:
                t_idx = self._transition_index[target]
                p_idx = self._place_index[source]
                if is_inhibitor or weight < 0.0:
                    inhibitors[t_idx][p_idx] = amount
                else:
                    inputs[t_idx][p_idx] = inputs[t_idx].get(p_idx, Fraction(0)) + amount
            else:
                t_idx = self._transition_index[source]
                p_idx = self._place_index[target]
                outputs[t_idx][p_idx] = outputs[t_idx].get(p_idx, Fraction(0)) + amount
        return inputs, outputs, inhibitors

    def _validate_bounds(self, bounds: dict[str, tuple[float, float]]) -> None:
        if not bounds:
            raise ValueError("marking bounds must not be empty")
        for place, (lower, upper) in bounds.items():
            self._require_place(place)
            if lower > upper:
                raise ValueError(f"lower bound exceeds upper bound for place {place!r}")

    def _require_place(self, place: str) -> None:
        if place not in self._place_index:
            raise ValueError(f"unknown place {place!r}")

    def _require_transition(self, transition: str) -> None:
        if transition not in self._transition_index:
            raise ValueError(f"unknown transition {transition!r}")


def verify_formal_contracts(
    net: StochasticPetriNet,
    *,
    max_depth: int,
    marking_bounds: dict[str, tuple[float, float]],
    temporal_specs: list[AlwaysBounded | EventuallyFires | NeverCoMarked] | None = None,
    backend: FormalBackend = "auto",
) -> FormalVerificationReport:
    """Run reachability, marking-bound, liveness, and temporal proof obligations."""
    verifier = FormalPetriNetVerifier(net, backend=backend)
    reachability = verifier.analyze_reachability(max_depth=max_depth)
    safety = verifier.prove_marking_bounds(marking_bounds, max_depth=max_depth)
    liveness = verifier.prove_transition_liveness(max_depth=max_depth)
    temporal = verifier.verify_temporal_specs(temporal_specs or [], max_depth=max_depth)
    return FormalVerificationReport(
        holds=safety.holds and liveness.holds and temporal.holds,
        reachability=reachability,
        safety=safety,
        liveness=liveness,
        temporal=temporal,
    )
