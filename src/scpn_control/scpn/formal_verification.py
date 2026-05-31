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

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from scpn_control.scpn.structure import StochasticPetriNet

FormalBackend = Literal["auto", "explicit-state", "z3"]
SAFETY_CERTIFICATE_SCHEMA_VERSION = "scpn-control.safety-certificate.v1"
SAFETY_CERTIFICATE_SCOPE = "bounded formal safety certificate for compiled SCPN Petri-net control logic"
SAFETY_CERTIFICATE_CLAIM_BOUNDARY = "not a facility safety approval, hardware timing certificate, or unbounded proof"


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
class CTLFormula:
    """Bounded CTL formula over a compiled SCPN transition system.

    Supported operators are the certification-relevant bounded subset that can
    produce exact counterexamples over the finite firing-depth transition
    relation: ``AG`` safety, ``EF`` reachability, and ``AG_EF`` recurrence.
    """

    name: str
    operator: Literal["AG", "EF", "AG_EF"]
    target: Literal["marking_bounds", "transition_fires", "not_comarked", "marked"]
    params: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("CTL formula name must be a non-empty string")
        if self.operator not in {"AG", "EF", "AG_EF"}:
            raise ValueError(f"unsupported CTL operator: {self.operator}")
        if self.target not in {"marking_bounds", "transition_fires", "not_comarked", "marked"}:
            raise ValueError(f"unsupported CTL target: {self.target}")
        if not isinstance(self.params, dict) or not self.params:
            raise ValueError("CTL formula params must be a non-empty mapping")

    @property
    def certificate_name(self) -> str:
        """Return the certificate-stable formula name."""

        return f"CTL:{self.name}:{self.operator}"

    @classmethod
    def ag_bounded(cls, name: str, bounds: dict[str, tuple[float, float]]) -> CTLFormula:
        """Create ``AG`` marking-bound safety: all paths always satisfy bounds."""

        return cls(name=name, operator="AG", target="marking_bounds", params={"bounds": bounds})

    @classmethod
    def ef_fires(cls, name: str, transition: str) -> CTLFormula:
        """Create ``EF`` transition reachability: some path eventually fires."""

        return cls(name=name, operator="EF", target="transition_fires", params={"transition": transition})

    @classmethod
    def ag_not_comarked(cls, name: str, place_a: str, place_b: str, *, threshold: float = 0.0) -> CTLFormula:
        """Create ``AG`` mutual-exclusion safety for two marked places."""

        return cls(
            name=name,
            operator="AG",
            target="not_comarked",
            params={"place_a": place_a, "place_b": place_b, "threshold": threshold},
        )

    @classmethod
    def ag_ef_marked(cls, name: str, place: str, *, threshold: float = 0.0) -> CTLFormula:
        """Create ``AG EF`` recurrence: every reachable state can recover marking."""

        return cls(name=name, operator="AG_EF", target="marked", params={"place": place, "threshold": threshold})


@dataclass(frozen=True)
class LTLFormula:
    """Bounded LTL formula over all finite firing paths of a compiled SCPN."""

    name: str
    operator: Literal["G", "F", "G_implies_F"]
    target: Literal["marking_bounds", "transition_fires", "fire_leads_to_marking"]
    params: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("LTL formula name must be a non-empty string")
        if self.operator not in {"G", "F", "G_implies_F"}:
            raise ValueError(f"unsupported LTL operator: {self.operator}")
        if self.target not in {"marking_bounds", "transition_fires", "fire_leads_to_marking"}:
            raise ValueError(f"unsupported LTL target: {self.target}")
        if not isinstance(self.params, dict) or not self.params:
            raise ValueError("LTL formula params must be a non-empty mapping")

    @property
    def certificate_name(self) -> str:
        """Return the certificate-stable formula name."""

        return f"LTL:{self.name}:{self.operator}"

    @classmethod
    def globally_bounded(cls, name: str, bounds: dict[str, tuple[float, float]]) -> LTLFormula:
        """Create ``G`` marking-bound safety over every bounded path."""

        return cls(name=name, operator="G", target="marking_bounds", params={"bounds": bounds})

    @classmethod
    def eventually_fires(cls, name: str, transition: str) -> LTLFormula:
        """Create bounded ``F`` transition reachability."""

        return cls(name=name, operator="F", target="transition_fires", params={"transition": transition})

    @classmethod
    def globally_fire_leads_to_marking(
        cls,
        name: str,
        trigger_transition: str,
        target_place: str,
        *,
        threshold: float = 0.0,
        within: int = 1,
    ) -> LTLFormula:
        """Create ``G(trigger -> F<=within marked)`` response checking."""

        return cls(
            name=name,
            operator="G_implies_F",
            target="fire_leads_to_marking",
            params={
                "trigger_transition": trigger_transition,
                "target_place": target_place,
                "threshold": threshold,
                "within": within,
            },
        )


@dataclass(frozen=True)
class PlaceInvariant:
    """Algebraic P-invariant over named places.

    The verifier proves that ``sum(weights[p] * marking[p])`` is unchanged by
    every transition and then checks the bounded reachable graph against the
    same invariant value. This is a structural proof obligation, not a sampled
    simulation assertion.
    """

    name: str
    weights: dict[str, float]


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


@dataclass(frozen=True)
class AlwaysEventuallyMarked:
    """Bounded recurrence spec: every reachable path can reach a marked place."""

    name: str
    place: str
    threshold: float = 0.0


@dataclass(frozen=True)
class FireLeadsToMarking:
    """Bounded response spec: a transition firing is followed by a marked place."""

    name: str
    trigger_transition: str
    target_place: str
    threshold: float = 0.0
    within: int = 1


def _as_fraction(value: float) -> Fraction:
    value_float = float(value)
    if not value_float == value_float or value_float in {float("inf"), float("-inf")}:
        raise ValueError("formal verification numeric values must be finite")
    return Fraction(str(value_float))


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
        self._validate_max_depth(max_depth)
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
        self._validate_max_depth(max_depth)
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

    def prove_place_invariants(
        self,
        invariants: list[PlaceInvariant],
        *,
        max_depth: int,
    ) -> FormalPropertyReport:
        """Prove algebraic place invariants and bounded reachable consistency."""
        self._validate_max_depth(max_depth)
        if not invariants:
            raise ValueError("place invariants must not be empty")
        states = self._reachable_state_map(max_depth=max_depth)
        checked: list[str] = []
        violations: list[FormalViolation] = []
        for invariant in invariants:
            checked.append(invariant.name)
            weights = self._validate_invariant_weights(invariant)
            initial_value = self._invariant_value(self._initial, weights)
            for transition in self.transition_names:
                delta = self._transition_invariant_delta(transition, weights)
                if delta != 0:
                    violations.append(
                        FormalViolation(
                            property_name=invariant.name,
                            message=(
                                f"transition {transition!r} changes invariant {invariant.name!r} by {float(delta):.12g}"
                            ),
                            marking=_as_float_marking(self.place_names, self._initial),
                            path=[],
                            transition=transition,
                        )
                    )
                    return FormalPropertyReport(
                        holds=False,
                        backend=self.backend,
                        max_depth=max_depth,
                        violations=violations,
                        checked_specs=checked,
                    )
            for marking, path in states:
                value = self._invariant_value(marking, weights)
                if value != initial_value:
                    violations.append(
                        FormalViolation(
                            property_name=invariant.name,
                            message=(
                                f"reachable marking changes invariant {invariant.name!r}: "
                                f"{float(value):.12g} != {float(initial_value):.12g}"
                            ),
                            marking=_as_float_marking(self.place_names, marking),
                            path=path,
                        )
                    )
                    return FormalPropertyReport(
                        holds=False,
                        backend=self.backend,
                        max_depth=max_depth,
                        violations=violations,
                        checked_specs=checked,
                    )
        return FormalPropertyReport(holds=True, backend=self.backend, max_depth=max_depth, checked_specs=checked)

    def prove_transition_liveness(self, *, max_depth: int) -> FormalPropertyReport:
        """Prove bounded transition liveness by checking every transition fires on some path."""
        self._validate_max_depth(max_depth)
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
        specs: list[AlwaysBounded | EventuallyFires | NeverCoMarked | AlwaysEventuallyMarked | FireLeadsToMarking],
        *,
        max_depth: int,
    ) -> FormalPropertyReport:
        """Verify bounded temporal safety/liveness specifications."""
        self._validate_max_depth(max_depth)
        states = self._reachable_state_map(max_depth=max_depth)
        paths = self._reachable_path_tree(max_depth=max_depth)
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
            elif isinstance(spec, AlwaysEventuallyMarked):
                self._require_place(spec.place)
                threshold = _as_fraction(spec.threshold)
                p_idx = self._place_index[spec.place]
                for marking, path in paths:
                    if marking[p_idx] > threshold:
                        continue
                    remaining_depth = max_depth - len(path)
                    descendants = self._reachable_path_tree_from(marking, path, max_depth=remaining_depth)
                    if not any(desc_marking[p_idx] > threshold for desc_marking, _desc_path in descendants):
                        violations.append(
                            FormalViolation(
                                property_name=spec.name,
                                message=(
                                    f"place {spec.place!r} is not inevitably recoverable above "
                                    f"{float(threshold):.12g} within remaining depth {remaining_depth}"
                                ),
                                marking=_as_float_marking(self.place_names, marking),
                                path=path,
                                place=spec.place,
                            )
                        )
                        break
                if violations:
                    break
            elif isinstance(spec, FireLeadsToMarking):
                self._require_transition(spec.trigger_transition)
                self._require_place(spec.target_place)
                if isinstance(spec.within, bool) or int(spec.within) != spec.within or spec.within < 0:
                    raise ValueError("FireLeadsToMarking.within must be an integer >= 0")
                threshold = _as_fraction(spec.threshold)
                p_idx = self._place_index[spec.target_place]
                for marking, path in paths:
                    if not path or path[-1] != spec.trigger_transition:
                        continue
                    descendants = self._reachable_path_tree_from(marking, path, max_depth=int(spec.within))
                    if not any(desc_marking[p_idx] > threshold for desc_marking, _desc_path in descendants):
                        violations.append(
                            FormalViolation(
                                property_name=spec.name,
                                message=(
                                    f"transition {spec.trigger_transition!r} is not followed by "
                                    f"place {spec.target_place!r} above {float(threshold):.12g} "
                                    f"within {int(spec.within)} firings"
                                ),
                                marking=_as_float_marking(self.place_names, marking),
                                path=path,
                                transition=spec.trigger_transition,
                                place=spec.target_place,
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

    def verify_ctl_specs(self, specs: list[CTLFormula], *, max_depth: int) -> FormalPropertyReport:
        """Verify bounded CTL formulas over the compiled Petri-net relation."""

        temporal_specs: list[
            AlwaysBounded | EventuallyFires | NeverCoMarked | AlwaysEventuallyMarked | FireLeadsToMarking
        ] = [self._compile_ctl_formula(spec) for spec in specs]
        return self.verify_temporal_specs(temporal_specs, max_depth=max_depth)

    def verify_ltl_specs(self, specs: list[LTLFormula], *, max_depth: int) -> FormalPropertyReport:
        """Verify bounded LTL formulas over all finite firing paths."""

        temporal_specs: list[
            AlwaysBounded | EventuallyFires | NeverCoMarked | AlwaysEventuallyMarked | FireLeadsToMarking
        ] = [self._compile_ltl_formula(spec) for spec in specs]
        return self.verify_temporal_specs(temporal_specs, max_depth=max_depth)

    def _compile_ctl_formula(
        self,
        spec: CTLFormula,
    ) -> AlwaysBounded | EventuallyFires | NeverCoMarked | AlwaysEventuallyMarked:
        label = spec.certificate_name
        if spec.operator == "AG" and spec.target == "marking_bounds":
            bounds = spec.params.get("bounds")
            if not isinstance(bounds, dict):
                raise ValueError(f"CTL formula {spec.name!r} requires marking bounds")
            return AlwaysBounded(label, bounds)
        if spec.operator == "EF" and spec.target == "transition_fires":
            transition = spec.params.get("transition")
            if not isinstance(transition, str) or not transition:
                raise ValueError(f"CTL formula {spec.name!r} requires a transition")
            return EventuallyFires(label, transition)
        if spec.operator == "AG" and spec.target == "not_comarked":
            place_a = spec.params.get("place_a")
            place_b = spec.params.get("place_b")
            if not isinstance(place_a, str) or not isinstance(place_b, str) or not place_a or not place_b:
                raise ValueError(f"CTL formula {spec.name!r} requires two places")
            return NeverCoMarked(label, place_a, place_b, threshold=float(spec.params.get("threshold", 0.0)))
        if spec.operator == "AG_EF" and spec.target == "marked":
            place = spec.params.get("place")
            if not isinstance(place, str) or not place:
                raise ValueError(f"CTL formula {spec.name!r} requires a place")
            return AlwaysEventuallyMarked(label, place, threshold=float(spec.params.get("threshold", 0.0)))
        raise ValueError(f"unsupported CTL formula combination: {spec.operator}/{spec.target}")

    def _compile_ltl_formula(
        self,
        spec: LTLFormula,
    ) -> AlwaysBounded | EventuallyFires | FireLeadsToMarking:
        label = spec.certificate_name
        if spec.operator == "G" and spec.target == "marking_bounds":
            bounds = spec.params.get("bounds")
            if not isinstance(bounds, dict):
                raise ValueError(f"LTL formula {spec.name!r} requires marking bounds")
            return AlwaysBounded(label, bounds)
        if spec.operator == "F" and spec.target == "transition_fires":
            transition = spec.params.get("transition")
            if not isinstance(transition, str) or not transition:
                raise ValueError(f"LTL formula {spec.name!r} requires a transition")
            return EventuallyFires(label, transition)
        if spec.operator == "G_implies_F" and spec.target == "fire_leads_to_marking":
            trigger = spec.params.get("trigger_transition")
            target = spec.params.get("target_place")
            if not isinstance(trigger, str) or not isinstance(target, str) or not trigger or not target:
                raise ValueError(f"LTL formula {spec.name!r} requires trigger transition and target place")
            return FireLeadsToMarking(
                label,
                trigger,
                target,
                threshold=float(spec.params.get("threshold", 0.0)),
                within=int(spec.params.get("within", 1)),
            )
        raise ValueError(f"unsupported LTL formula combination: {spec.operator}/{spec.target}")

    def _reachable_state_map(self, *, max_depth: int) -> list[tuple[tuple[Fraction, ...], list[str]]]:
        self._validate_max_depth(max_depth)
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

    def _reachable_path_tree(self, *, max_depth: int) -> list[tuple[tuple[Fraction, ...], list[str]]]:
        return self._reachable_path_tree_from(self._initial, [], max_depth=max_depth)

    def _reachable_path_tree_from(
        self,
        marking: tuple[Fraction, ...],
        path: list[str],
        *,
        max_depth: int,
    ) -> list[tuple[tuple[Fraction, ...], list[str]]]:
        self._validate_max_depth(max_depth)
        states: list[tuple[tuple[Fraction, ...], list[str]]] = [(marking, path)]
        queue: deque[tuple[tuple[Fraction, ...], list[str], int]] = deque([(marking, path, 0)])
        while queue:
            current, current_path, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for transition in self.transition_names:
                next_marking = self._fire_if_enabled(current, transition)
                if next_marking is None:
                    continue
                next_path = [*current_path, transition]
                states.append((next_marking, next_path))
                queue.append((next_marking, next_path, depth + 1))
        return states

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
            _as_fraction(lower)
            _as_fraction(upper)

    def _validate_invariant_weights(self, invariant: PlaceInvariant) -> dict[int, Fraction]:
        if not invariant.name:
            raise ValueError("place invariant name must not be empty")
        if not invariant.weights:
            raise ValueError(f"place invariant {invariant.name!r} must include at least one place weight")
        weights: dict[int, Fraction] = {}
        for place, weight in invariant.weights.items():
            self._require_place(place)
            weights[self._place_index[place]] = _as_fraction(weight)
        if all(weight == 0 for weight in weights.values()):
            raise ValueError(f"place invariant {invariant.name!r} must not be identically zero")
        return weights

    def _invariant_value(self, marking: tuple[Fraction, ...], weights: dict[int, Fraction]) -> Fraction:
        return sum((weight * marking[p_idx] for p_idx, weight in weights.items()), Fraction(0))

    def _transition_invariant_delta(self, transition: str, weights: dict[int, Fraction]) -> Fraction:
        t_idx = self._transition_index[transition]
        delta = Fraction(0)
        for p_idx, weight in self._inputs[t_idx].items():
            delta -= weights.get(p_idx, Fraction(0)) * weight
        for p_idx, weight in self._outputs[t_idx].items():
            delta += weights.get(p_idx, Fraction(0)) * weight
        return delta

    @staticmethod
    def _validate_max_depth(max_depth: int) -> None:
        if isinstance(max_depth, bool) or int(max_depth) != max_depth or max_depth < 0:
            raise ValueError("max_depth must be an integer >= 0")

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
    temporal_specs: list[AlwaysBounded | EventuallyFires | NeverCoMarked | AlwaysEventuallyMarked | FireLeadsToMarking]
    | None = None,
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


def build_safety_certificate_payload(
    report: FormalVerificationReport,
    *,
    ctl_report: FormalPropertyReport | None = None,
    ltl_report: FormalPropertyReport | None = None,
    artifact_sha256: str | None = None,
    issuer: str = "scpn-control",
) -> dict[str, Any]:
    """Build a schema-versioned tamper-evident formal safety certificate."""

    if artifact_sha256 is not None and not _is_sha256(artifact_sha256):
        raise ValueError("artifact_sha256 must be a SHA-256 hex digest")
    if not isinstance(issuer, str) or not issuer:
        raise ValueError("safety certificate issuer must be a non-empty string")
    sections = {
        "reachability": _jsonable(asdict(report.reachability)),
        "safety": _jsonable(asdict(report.safety)),
        "liveness": _jsonable(asdict(report.liveness)),
        "temporal": _jsonable(asdict(report.temporal)),
        "ctl": _jsonable(asdict(ctl_report)) if ctl_report is not None else None,
        "ltl": _jsonable(asdict(ltl_report)) if ltl_report is not None else None,
    }
    holds = report.holds
    if ctl_report is not None:
        holds = holds and ctl_report.holds
    if ltl_report is not None:
        holds = holds and ltl_report.holds
    payload: dict[str, Any] = {
        "schema_version": SAFETY_CERTIFICATE_SCHEMA_VERSION,
        "status": "pass" if holds else "fail",
        "holds": holds,
        "backend": report.reachability.backend,
        "max_depth": report.reachability.max_depth,
        "issuer": issuer,
        "artifact_sha256": artifact_sha256,
        "checked_specs": _certificate_checked_specs(report, ctl_report, ltl_report),
        "scope": SAFETY_CERTIFICATE_SCOPE,
        "claim_boundary": SAFETY_CERTIFICATE_CLAIM_BOUNDARY,
        "sections": sections,
    }
    payload["payload_sha256"] = _payload_digest(payload)
    return validate_safety_certificate_payload(payload)


def write_safety_certificate(
    report: FormalVerificationReport,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
    ctl_report: FormalPropertyReport | None = None,
    ltl_report: FormalPropertyReport | None = None,
    artifact_sha256: str | None = None,
    issuer: str = "scpn-control",
) -> dict[str, Any]:
    """Persist a formal safety certificate as JSON and Markdown artifacts."""

    payload = build_safety_certificate_payload(
        report,
        ctl_report=ctl_report,
        ltl_report=ltl_report,
        artifact_sha256=artifact_sha256,
        issuer=issuer,
    )
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_target.write_text(_render_safety_certificate_markdown(payload), encoding="utf-8")
    return payload


def validate_safety_certificate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a schema-versioned formal safety certificate payload."""

    if payload.get("schema_version") != SAFETY_CERTIFICATE_SCHEMA_VERSION:
        raise ValueError("safety certificate schema_version is unsupported")
    if payload.get("status") not in {"pass", "fail"}:
        raise ValueError("safety certificate status must be pass or fail")
    if not isinstance(payload.get("holds"), bool):
        raise ValueError("safety certificate holds must be a boolean")
    if payload["status"] == "pass" and not payload["holds"]:
        raise ValueError("passing safety certificate must hold")
    if payload["status"] == "fail" and payload["holds"]:
        raise ValueError("failed safety certificate must not hold")
    if payload.get("backend") not in {"explicit-state", "z3"}:
        raise ValueError("safety certificate backend is unsupported")
    if isinstance(payload.get("max_depth"), bool) or not isinstance(payload.get("max_depth"), int):
        raise ValueError("safety certificate max_depth must be an integer")
    if payload["max_depth"] < 0:
        raise ValueError("safety certificate max_depth must be non-negative")
    if not isinstance(payload.get("issuer"), str) or not payload["issuer"]:
        raise ValueError("safety certificate issuer must be a non-empty string")
    artifact_sha256 = payload.get("artifact_sha256")
    if artifact_sha256 is not None and (not isinstance(artifact_sha256, str) or not _is_sha256(artifact_sha256)):
        raise ValueError("safety certificate artifact_sha256 must be a SHA-256 hex digest or null")
    if not isinstance(payload.get("checked_specs"), list) or not payload["checked_specs"]:
        raise ValueError("safety certificate checked_specs must be a non-empty list")
    if any(not isinstance(spec, str) or not spec for spec in payload["checked_specs"]):
        raise ValueError("safety certificate checked_specs must contain non-empty strings")
    if len(payload["checked_specs"]) != len(set(payload["checked_specs"])):
        raise ValueError("safety certificate checked_specs must be unique")
    if payload.get("scope") != SAFETY_CERTIFICATE_SCOPE:
        raise ValueError("safety certificate scope is unsupported")
    if payload.get("claim_boundary") != SAFETY_CERTIFICATE_CLAIM_BOUNDARY:
        raise ValueError("safety certificate claim_boundary is unsupported")
    if not isinstance(payload.get("sections"), dict):
        raise ValueError("safety certificate sections must be an object")
    sections = payload["sections"]
    required_sections = ("reachability", "safety", "liveness", "temporal")
    section_holds = True
    for section_name in (*required_sections, "ctl", "ltl"):
        section = sections.get(section_name)
        if section is None:
            if section_name in required_sections:
                raise ValueError(f"safety certificate {section_name} section must be an object")
            continue
        if not isinstance(section, dict):
            raise ValueError(f"safety certificate {section_name} section must be an object")
        if not isinstance(section.get("holds"), bool):
            raise ValueError(f"safety certificate {section_name} section holds must be a boolean")
        if section.get("backend") != payload["backend"]:
            raise ValueError(f"safety certificate {section_name} section backend must match certificate backend")
        if section.get("max_depth") != payload["max_depth"]:
            raise ValueError(f"safety certificate {section_name} section depth must match certificate depth")
        checked = section.get("checked_specs", [])
        if not isinstance(checked, list):
            raise ValueError(f"safety certificate {section_name} section checked_specs must be a list")
        if any(not isinstance(spec, str) or not spec for spec in checked):
            raise ValueError(f"safety certificate {section_name} section checked_specs must contain non-empty strings")
        section_holds = section_holds and section["holds"]
    if payload["holds"] != section_holds:
        raise ValueError("safety certificate section holds must match certificate holds")
    if payload["checked_specs"] != _certificate_checked_specs_from_sections(sections):
        raise ValueError("safety certificate checked_specs must match certificate sections")
    declared_digest = payload.get("payload_sha256")
    if not isinstance(declared_digest, str) or not _is_sha256(declared_digest):
        raise ValueError("safety certificate payload_sha256 must be a SHA-256 hex digest")
    if _payload_digest(payload) != declared_digest.lower():
        raise ValueError("safety certificate payload_sha256 does not match payload")
    return payload


def _certificate_checked_specs(
    report: FormalVerificationReport,
    ctl_report: FormalPropertyReport | None,
    ltl_report: FormalPropertyReport | None,
) -> list[str]:
    specs = ["marking_bounds", "transition_liveness"]
    for section in (report.temporal, ctl_report, ltl_report):
        if section is None:
            continue
        for spec in section.checked_specs:
            if spec not in specs:
                specs.append(spec)
    return specs


def _certificate_checked_specs_from_sections(sections: dict[str, Any]) -> list[str]:
    specs = ["marking_bounds", "transition_liveness"]
    for section_name in ("temporal", "ctl", "ltl"):
        section = sections.get(section_name)
        if section is None:
            continue
        if not isinstance(section, dict):
            continue
        for spec in section.get("checked_specs", []):
            if spec not in specs:
                specs.append(spec)
    return specs


def _payload_digest(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("payload_sha256", None)
    blob = json.dumps(canonical, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _render_safety_certificate_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SCPN Formal Safety Certificate",
        "",
        f"- Schema: `{payload['schema_version']}`",
        f"- Status: `{payload['status']}`",
        f"- Backend: `{payload['backend']}`",
        f"- Max depth: `{payload['max_depth']}`",
        f"- Issuer: `{payload['issuer']}`",
        f"- Payload SHA-256: `{payload['payload_sha256']}`",
        f"- Scope: {payload['scope']}.",
        f"- Claim boundary: {payload['claim_boundary']}.",
        "",
        "## Checked specifications",
        "",
    ]
    lines.extend(f"- `{spec}`" for spec in payload["checked_specs"])
    lines.extend(["", "## Section status", ""])
    for section_name, section in payload["sections"].items():
        if section is None:
            lines.append(f"- `{section_name}`: not supplied")
        else:
            lines.append(f"- `{section_name}`: holds=`{section['holds']}`")
    violations: list[dict[str, Any]] = []
    for section in payload["sections"].values():
        if isinstance(section, dict):
            violations.extend(section.get("violations", []))
    if violations:
        lines.extend(["", "## Counterexamples", ""])
        for violation in violations:
            lines.append(f"- `{violation['property_name']}` path={violation['path']} message={violation['message']}")
    lines.append("")
    return "\n".join(lines)
