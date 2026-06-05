# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Z3 Petri Net Model Checking

"""Bounded SMT model checking for compiled SCPN Petri-net control logic."""

from __future__ import annotations

import math
import json
import hashlib
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from scpn_control.scpn.formal_verification import (
    AlwaysBounded,
    AlwaysEventuallyMarked,
    CTLFormula,
    EventuallyFires,
    FireLeadsToMarking,
    FormalViolation,
    LTLFormula,
    NeverCoMarked,
    _as_float_marking,
    _as_fraction,
)
from scpn_control.scpn.structure import StochasticPetriNet

Z3_FORMAL_REPORT_SCHEMA_VERSION = "scpn-control.z3-formal-report.v1"
Z3_FORMAL_REPORT_SCOPE = "bounded SMT evidence for compiled Petri-net control logic"
Z3_FORMAL_REPORT_CLAIM_BOUNDARY = "not hardware timing evidence, PCS certification, or unbounded liveness proof"
SYMBIYOSYS_SYMBOLIC_CONTRACT_VERSION = "scpn-control.symbiyosys-contract.v1"
SYMBIOSYS_SYMBOLIC_CONTRACT_VERSION = SYMBIYOSYS_SYMBOLIC_CONTRACT_VERSION
RTI_CONTRACT_BUDGET_NS = 50.0
_Z3_BLOCKED_SOLVER_LABEL = "z3-solver unavailable"
_Z3_BLOCKED_CHECKED_SPECS = ["z3_solver_available"]

_Z3_REPORT_PASS_FAIL_KEYS = frozenset(
    {
        "backend",
        "checked_specs",
        "claim_boundary",
        "holds",
        "max_depth",
        "payload_sha256",
        "safety",
        "schema_version",
        "scope",
        "solver",
        "status",
        "temporal",
    }
)
_Z3_REPORT_BLOCKED_KEYS = _Z3_REPORT_PASS_FAIL_KEYS | {"reason"}
_Z3_REPORT_SECTION_KEYS = frozenset(
    {
        "backend",
        "checked_specs",
        "holds",
        "max_depth",
        "solver_status",
        "violations",
    }
)
_Z3_REPORT_VIOLATION_KEYS = frozenset(
    {
        "marking",
        "message",
        "path",
        "place",
        "property_name",
        "transition",
    }
)


@dataclass(frozen=True)
class Z3ModelCheckingReport:
    """Result of a bounded Z3 model-checking obligation."""

    holds: bool
    backend: str
    max_depth: int
    solver_status: str
    violations: list[FormalViolation] = field(default_factory=list)
    checked_specs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Z3FormalVerificationReport:
    """Combined bounded SMT formal-verification report."""

    holds: bool
    backend: str
    max_depth: int
    safety: Z3ModelCheckingReport
    temporal: Z3ModelCheckingReport


def _require_z3() -> Any:
    try:
        import z3
    except ModuleNotFoundError as exc:
        raise RuntimeError("Z3 model checking requires the optional z3-solver package") from exc
    return z3


def _z3_fraction(z3: Any, value: Fraction) -> Any:
    return z3.RealVal(f"{value.numerator}/{value.denominator}")


class Z3BoundedModelChecker:
    """Bounded SMT model checker for compiled ``StochasticPetriNet`` graphs."""

    def __init__(self, net: StochasticPetriNet) -> None:
        if not net.is_compiled:
            raise RuntimeError("Net must be compiled before Z3 model checking.")
        if net.W_in is None or net.W_out is None:
            raise RuntimeError("Net must be compiled before Z3 model checking.")
        self.net = net
        self.place_names = net.place_names
        self.transition_names = net.transition_names
        self._place_index = {name: idx for idx, name in enumerate(self.place_names)}
        self._transition_index = {name: idx for idx, name in enumerate(self.transition_names)}
        self._initial = tuple(_as_fraction(value) for value in net.get_initial_marking())
        self._inputs, self._outputs, self._inhibitors = self._extract_transition_relation()

    def prove_marking_bounds(
        self,
        bounds: dict[str, tuple[float, float]],
        *,
        max_depth: int,
        weight_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> Z3ModelCheckingReport:
        """Prove marking bounds by asking Z3 for a reachable counterexample."""
        self._validate_max_depth(max_depth)
        self._validate_bounds(bounds)
        z3, solver, markings, firings, _idle = self._transition_system(max_depth, weight_bounds=weight_bounds or {})
        violation_terms = []
        fractional_bounds = {
            place: (_as_fraction(lower), _as_fraction(upper)) for place, (lower, upper) in bounds.items()
        }
        for step in range(max_depth + 1):
            for place, (lower, upper) in fractional_bounds.items():
                p_idx = self._place_index[place]
                violation_terms.append(markings[step][p_idx] < _z3_fraction(z3, lower))
                violation_terms.append(markings[step][p_idx] > _z3_fraction(z3, upper))
        solver.add(z3.Or(*violation_terms))
        status = solver.check()
        if status == z3.unsat:
            return Z3ModelCheckingReport(True, "z3", max_depth, "unsat")
        if status != z3.sat:
            return Z3ModelCheckingReport(False, "z3", max_depth, str(status))
        model = solver.model()
        path = self._path_from_model(model, firings, max_depth)
        for step in range(max_depth + 1):
            marking = self._marking_from_model(z3, model, markings[step])
            for place, (lower, upper) in fractional_bounds.items():
                value = Fraction(str(marking[place]))
                if value < lower or value > upper:
                    return Z3ModelCheckingReport(
                        False,
                        "z3",
                        max_depth,
                        "sat",
                        [
                            FormalViolation(
                                property_name="marking_bounds",
                                message=(
                                    f"place {place!r} reached {float(value):.12g} outside "
                                    f"[{float(lower)}, {float(upper)}]"
                                ),
                                marking={key: float(val) for key, val in marking.items()},
                                path=path[:step],
                                place=place,
                            )
                        ],
                    )
        raise RuntimeError("Z3 returned a model without an identifiable marking-bound violation.")

    def verify_temporal_specs(
        self,
        specs: list[AlwaysBounded | EventuallyFires | NeverCoMarked | AlwaysEventuallyMarked | FireLeadsToMarking],
        *,
        max_depth: int,
        weight_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> Z3ModelCheckingReport:
        """Verify supported bounded temporal specifications with Z3."""
        self._validate_max_depth(max_depth)
        checked_weight_bounds = weight_bounds or {}
        checked: list[str] = []
        for spec in specs:
            checked.append(spec.name)
            if isinstance(spec, AlwaysBounded):
                report = self.prove_marking_bounds(
                    spec.bounds, max_depth=max_depth, weight_bounds=checked_weight_bounds
                )
                if not report.holds:
                    return Z3ModelCheckingReport(
                        False, "z3", max_depth, report.solver_status, report.violations, checked
                    )
            elif isinstance(spec, EventuallyFires):
                report = self._prove_eventually_fires(spec, max_depth=max_depth, weight_bounds=checked_weight_bounds)
                if not report.holds:
                    return Z3ModelCheckingReport(
                        False, "z3", max_depth, report.solver_status, report.violations, checked
                    )
            elif isinstance(spec, NeverCoMarked):
                report = self._prove_never_comarked(spec, max_depth=max_depth, weight_bounds=checked_weight_bounds)
                if not report.holds:
                    return Z3ModelCheckingReport(
                        False, "z3", max_depth, report.solver_status, report.violations, checked
                    )
            elif isinstance(spec, FireLeadsToMarking):
                report = self._prove_fire_leads_to_marking(
                    spec, max_depth=max_depth, weight_bounds=checked_weight_bounds
                )
                if not report.holds:
                    return Z3ModelCheckingReport(
                        False, "z3", max_depth, report.solver_status, report.violations, checked
                    )
            elif isinstance(spec, AlwaysEventuallyMarked):
                report = self._prove_always_eventually_marked(
                    spec, max_depth=max_depth, weight_bounds=checked_weight_bounds
                )
                if not report.holds:
                    return Z3ModelCheckingReport(
                        False, "z3", max_depth, report.solver_status, report.violations, checked
                    )
            else:
                raise TypeError(f"unsupported temporal specification: {type(spec).__name__}")
        return Z3ModelCheckingReport(True, "z3", max_depth, "unsat", checked_specs=checked)

    def verify_ctl_specs(self, specs: list[CTLFormula], *, max_depth: int) -> Z3ModelCheckingReport:
        """Verify bounded CTL formulas with the Z3 transition relation."""

        temporal_specs: list[
            AlwaysBounded | EventuallyFires | NeverCoMarked | AlwaysEventuallyMarked | FireLeadsToMarking
        ] = [self._compile_ctl_formula(spec) for spec in specs]
        return self.verify_temporal_specs(temporal_specs, max_depth=max_depth)

    def verify_ltl_specs(self, specs: list[LTLFormula], *, max_depth: int) -> Z3ModelCheckingReport:
        """Verify bounded LTL formulas with the Z3 transition relation."""

        temporal_specs: list[
            AlwaysBounded | EventuallyFires | NeverCoMarked | AlwaysEventuallyMarked | FireLeadsToMarking
        ] = [self._compile_ltl_formula(spec) for spec in specs]
        return self.verify_temporal_specs(temporal_specs, max_depth=max_depth)

    def verify_trigger_response_latency(
        self,
        *,
        trigger_transition: str,
        response_transition: str,
        max_latency_ns: float,
        tick_period_ns: float = 10.0,
        max_depth: int,
        weight_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> Z3ModelCheckingReport:
        """Verify a bounded trigger->response latency contract in transition steps.

        The bound is interpreted as:
        at most ``ceil(max_latency_ns / tick_period_ns)`` non-idle transition
        steps from a trigger fire to the first subsequent response fire. Idle
        stalls are intentionally discharged by the separate no-stall SymbiYosys
        contract rather than conflated with trigger-response latency.
        """
        self._require_transition(trigger_transition)
        self._require_transition(response_transition)
        if not math.isfinite(max_latency_ns) or max_latency_ns <= 0.0:
            raise ValueError("max_latency_ns must be finite and > 0")
        if max_latency_ns > RTI_CONTRACT_BUDGET_NS:
            raise ValueError("max_latency_ns must not exceed 50.0 ns contract budget")
        if not math.isfinite(tick_period_ns) or tick_period_ns <= 0.0:
            raise ValueError("tick_period_ns must be finite and > 0")
        max_latency_steps = int(math.ceil(max_latency_ns / tick_period_ns))
        if max_latency_steps < 1:
            raise ValueError("computed latency window must be >= 1")
        self._validate_max_depth(max_depth)
        if max_depth < max_latency_steps:
            raise ValueError("max_depth must be >= ceil(max_latency_ns / tick_period_ns)")

        z3, solver, markings, firings, idle = self._transition_system(max_depth, weight_bounds=weight_bounds or {})
        t_idx = self._transition_index[trigger_transition]
        r_idx = self._transition_index[response_transition]
        checks = []
        for step in range(max_depth):
            deadline = step + max_latency_steps
            if deadline >= max_depth:
                continue
            response_window = [firings[window][r_idx] for window in range(step + 1, deadline + 1)]
            if response_window:
                non_idle_window = [z3.Not(idle[window]) for window in range(step + 1, deadline + 1)]
                checks.append(z3.And(firings[step][t_idx], *non_idle_window, z3.Not(z3.Or(*response_window))))
            else:
                checks.append(firings[step][t_idx])
        if not checks:
            return Z3ModelCheckingReport(
                False,
                "z3",
                max_depth,
                "sat",
                [
                    FormalViolation(
                        property_name="trigger_response_latency",
                        message="latency horizon exceeds configured depth; cannot discharge within max_depth",
                        marking=_as_float_marking(self.place_names, self._initial),
                        path=[],
                    )
                ],
                ["trigger_response_latency"],
            )
        solver.add(z3.Or(*checks))
        status = solver.check()
        if status == z3.unsat:
            return Z3ModelCheckingReport(True, "z3", max_depth, "unsat", checked_specs=["trigger_response_latency"])
        if status != z3.sat:
            return Z3ModelCheckingReport(
                False, "z3", max_depth, str(status), checked_specs=["trigger_response_latency"]
            )
        model = solver.model()
        path = self._path_from_model(model, firings, max_depth)
        marking = self._marking_from_model(z3, model, markings[min(len(path), max_depth)])
        violation = FormalViolation(
            property_name="trigger_response_latency",
            message=(
                f"transition {trigger_transition!r} can fire without {response_transition!r} "
                f"within {max_latency_steps} steps (tick_period_ns={float(tick_period_ns):.12g})"
            ),
            marking={key: float(val) for key, val in marking.items()},
            path=path,
            transition=trigger_transition,
        )
        return Z3ModelCheckingReport(False, "z3", max_depth, "sat", [violation], ["trigger_response_latency"])

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

    def _prove_eventually_fires(
        self, spec: EventuallyFires, *, max_depth: int, weight_bounds: dict[str, tuple[float, float]] | None = None
    ) -> Z3ModelCheckingReport:
        self._require_transition(spec.transition)
        z3, solver, markings, firings, _idle = self._transition_system(max_depth, weight_bounds=weight_bounds or {})
        t_idx = self._transition_index[spec.transition]
        solver.add(z3.Or(*(firings[step][t_idx] for step in range(max_depth))))
        status = solver.check()
        if status == z3.sat:
            return Z3ModelCheckingReport(True, "z3", max_depth, "sat", checked_specs=[spec.name])
        violation = FormalViolation(
            property_name=spec.name,
            message=f"transition {spec.transition!r} cannot fire within depth {max_depth}",
            marking=_as_float_marking(self.place_names, self._initial),
            path=[],
            transition=spec.transition,
        )
        return Z3ModelCheckingReport(False, "z3", max_depth, str(status), [violation], [spec.name])

    def _prove_never_comarked(
        self, spec: NeverCoMarked, *, max_depth: int, weight_bounds: dict[str, tuple[float, float]] | None = None
    ) -> Z3ModelCheckingReport:
        self._require_place(spec.place_a)
        self._require_place(spec.place_b)
        z3, solver, markings, firings, _idle = self._transition_system(max_depth, weight_bounds=weight_bounds or {})
        threshold = _as_fraction(spec.threshold)
        ia = self._place_index[spec.place_a]
        ib = self._place_index[spec.place_b]
        solver.add(
            z3.Or(
                *(
                    z3.And(
                        markings[step][ia] > _z3_fraction(z3, threshold),
                        markings[step][ib] > _z3_fraction(z3, threshold),
                    )
                    for step in range(max_depth + 1)
                )
            )
        )
        status = solver.check()
        if status == z3.unsat:
            return Z3ModelCheckingReport(True, "z3", max_depth, "unsat", checked_specs=[spec.name])
        if status != z3.sat:
            return Z3ModelCheckingReport(False, "z3", max_depth, str(status), checked_specs=[spec.name])
        model = solver.model()
        path = self._path_from_model(model, firings, max_depth)
        for step in range(max_depth + 1):
            marking = self._marking_from_model(z3, model, markings[step])
            if marking[spec.place_a] > threshold and marking[spec.place_b] > threshold:
                violation = FormalViolation(
                    property_name=spec.name,
                    message=(f"places {spec.place_a!r} and {spec.place_b!r} are both above {float(threshold):.12g}"),
                    marking={key: float(val) for key, val in marking.items()},
                    path=path[:step],
                )
                return Z3ModelCheckingReport(False, "z3", max_depth, "sat", [violation], [spec.name])
        raise RuntimeError("Z3 returned a model without an identifiable co-marking violation.")

    def _prove_fire_leads_to_marking(
        self,
        spec: FireLeadsToMarking,
        *,
        max_depth: int,
        weight_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> Z3ModelCheckingReport:
        self._require_transition(spec.trigger_transition)
        self._require_place(spec.target_place)
        if isinstance(spec.within, bool) or int(spec.within) != spec.within or spec.within < 0:
            raise ValueError("FireLeadsToMarking.within must be an integer >= 0")
        z3, solver, markings, firings, _idle = self._transition_system(max_depth, weight_bounds=weight_bounds or {})
        t_idx = self._transition_index[spec.trigger_transition]
        p_idx = self._place_index[spec.target_place]
        threshold = _as_fraction(spec.threshold)
        violation_terms = []
        for step in range(max_depth):
            deadline = step + int(spec.within) + 1
            if deadline > max_depth:
                violation_terms.append(firings[step][t_idx])
                continue
            response_window = [
                markings[future][p_idx] > _z3_fraction(z3, threshold) for future in range(step + 1, deadline + 1)
            ]
            violation_terms.append(z3.And(firings[step][t_idx], z3.Not(z3.Or(*response_window))))
        solver.add(z3.Or(*violation_terms))
        status = solver.check()
        if status == z3.unsat:
            return Z3ModelCheckingReport(True, "z3", max_depth, "unsat", checked_specs=[spec.name])
        if status != z3.sat:
            return Z3ModelCheckingReport(False, "z3", max_depth, str(status), checked_specs=[spec.name])
        model = solver.model()
        path = self._path_from_model(model, firings, max_depth)
        marking = self._marking_from_model(z3, model, markings[min(len(path), max_depth)])
        violation = FormalViolation(
            property_name=spec.name,
            message=(
                f"transition {spec.trigger_transition!r} is not followed by place "
                f"{spec.target_place!r} above {float(threshold):.12g} within {int(spec.within)} firings"
            ),
            marking={key: float(val) for key, val in marking.items()},
            path=path,
            transition=spec.trigger_transition,
            place=spec.target_place,
        )
        return Z3ModelCheckingReport(False, "z3", max_depth, "sat", [violation], [spec.name])

    def _prove_always_eventually_marked(
        self,
        spec: AlwaysEventuallyMarked,
        *,
        max_depth: int,
        weight_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> Z3ModelCheckingReport:
        self._require_place(spec.place)
        z3, solver, markings, firings, _idle = self._transition_system(max_depth, weight_bounds=weight_bounds or {})
        p_idx = self._place_index[spec.place]
        threshold = _as_fraction(spec.threshold)
        solver.add(z3.And(*(markings[step][p_idx] <= _z3_fraction(z3, threshold) for step in range(max_depth + 1))))
        status = solver.check()
        if status == z3.unsat:
            return Z3ModelCheckingReport(True, "z3", max_depth, "unsat", checked_specs=[spec.name])
        if status != z3.sat:
            return Z3ModelCheckingReport(False, "z3", max_depth, str(status), checked_specs=[spec.name])
        model = solver.model()
        path = self._path_from_model(model, firings, max_depth)
        marking = self._marking_from_model(z3, model, markings[min(len(path), max_depth)])
        violation = FormalViolation(
            property_name=spec.name,
            message=f"place {spec.place!r} is not marked above {float(threshold):.12g} on a bounded path",
            marking={key: float(val) for key, val in marking.items()},
            path=path,
            place=spec.place,
        )
        return Z3ModelCheckingReport(False, "z3", max_depth, "sat", [violation], [spec.name])

    def _transition_system(
        self, max_depth: int, *, weight_bounds: dict[str, tuple[float, float]] | None = None
    ) -> tuple[Any, Any, list[list[Any]], list[list[Any]], list[Any]]:
        z3 = _require_z3()
        solver = z3.Solver()
        validated_bounds = self._validate_weight_bounds(weight_bounds or {})
        markings = [
            [z3.Real(f"m_{step}_{place_idx}") for place_idx in range(len(self.place_names))]
            for step in range(max_depth + 1)
        ]
        firings = [
            [z3.Bool(f"fire_{step}_{transition_idx}") for transition_idx in range(len(self.transition_names))]
            for step in range(max_depth)
        ]
        idle = [z3.Bool(f"idle_{step}") for step in range(max_depth)]
        for p_idx, value in enumerate(self._initial):
            solver.add(markings[0][p_idx] == _z3_fraction(z3, value))
        for step in range(max_depth + 1):
            for p_idx in range(len(self.place_names)):
                solver.add(markings[step][p_idx] >= _z3_fraction(z3, Fraction(0)))
        for step in range(max_depth):
            weights = {
                t_idx: z3.Real(f"weight_{step}_{transition}") for transition, t_idx in self._transition_index.items()
            }
            for transition, t_idx in self._transition_index.items():
                lower, upper = validated_bounds.get(transition, (Fraction(1), Fraction(1)))
                solver.add(weights[t_idx] >= _z3_fraction(z3, lower))
                solver.add(weights[t_idx] <= _z3_fraction(z3, upper))
                solver.add(weights[t_idx] >= 0)
            solver.add(
                z3.PbEq(
                    [(idle[step], 1), *((firings[step][t_idx], 1) for t_idx in range(len(self.transition_names)))], 1
                )
            )
            solver.add(
                z3.Implies(
                    idle[step],
                    z3.And(
                        *(markings[step + 1][p_idx] == markings[step][p_idx] for p_idx in range(len(self.place_names)))
                    ),
                )
            )
            for transition, t_idx in self._transition_index.items():
                transition_weight = weights[t_idx]
                constraints = []
                for p_idx, amount in self._inputs[t_idx].items():
                    constraints.append(markings[step][p_idx] >= transition_weight * _z3_fraction(z3, amount))
                for p_idx, threshold in self._inhibitors[t_idx].items():
                    constraints.append(markings[step][p_idx] < transition_weight * _z3_fraction(z3, threshold))
                for p_idx in range(len(self.place_names)):
                    base_delta = self._outputs[t_idx].get(p_idx, Fraction(0)) - self._inputs[t_idx].get(
                        p_idx, Fraction(0)
                    )
                    constraints.append(
                        markings[step + 1][p_idx]
                        == markings[step][p_idx] + (transition_weight * _z3_fraction(z3, base_delta))
                    )
                if not constraints:
                    raise RuntimeError(f"transition {transition!r} produced an empty SMT constraint set")
                solver.add(z3.Implies(firings[step][t_idx], z3.And(*constraints)))
        return z3, solver, markings, firings, idle

    def build_symbiyosys_contract(
        self,
        *,
        max_depth: int,
        weight_bounds: dict[str, tuple[float, float]] | None = None,
        trigger_transition: str | None = None,
        response_transition: str | None = None,
        max_latency_ns: float = 50.0,
        tick_period_ns: float = 10.0,
        no_stall_window_ns: float = 50.0,
    ) -> dict[str, str]:
        """Build a bounded symbolic contract for SymbiYosys/SMTBMC review.

        The emitted contract uses `assume` bounds for hot-swappable transition
        weights and `assert` obligations for bounded trigger->response latency and
        bounded non-stall progression.
        """
        self._validate_max_depth(max_depth)
        validated_bounds = self._validate_weight_bounds(weight_bounds or {})
        if trigger_transition is not None:
            self._require_transition(trigger_transition)
        if response_transition is not None:
            self._require_transition(response_transition)
        if (trigger_transition is None) ^ (response_transition is None):
            raise ValueError("trigger_transition and response_transition must be provided together")
        if not math.isfinite(max_latency_ns) or max_latency_ns <= 0.0:
            raise ValueError("max_latency_ns must be finite and > 0")
        if max_latency_ns > RTI_CONTRACT_BUDGET_NS:
            raise ValueError("max_latency_ns must not exceed 50.0 ns contract budget")
        if not math.isfinite(no_stall_window_ns) or no_stall_window_ns <= 0.0:
            raise ValueError("no_stall_window_ns must be finite and > 0")
        if no_stall_window_ns > RTI_CONTRACT_BUDGET_NS:
            raise ValueError("no_stall_window_ns must not exceed 50.0 ns contract budget")
        if not math.isfinite(tick_period_ns) or tick_period_ns <= 0.0:
            raise ValueError("tick_period_ns must be finite and > 0")
        max_latency_steps = max(1, int(math.ceil(max_latency_ns / tick_period_ns)))
        no_stall_steps = max(1, int(math.ceil(no_stall_window_ns / tick_period_ns)))
        if max_depth < max_latency_steps:
            raise ValueError("max_depth must be >= ceil(max_latency_ns / tick_period_ns)")
        if max_depth < no_stall_steps:
            raise ValueError("max_depth must be >= ceil(no_stall_window_ns / tick_period_ns)")

        def _as_smt_real(value: Fraction) -> str:
            return f"(/ {int(value.numerator)} {int(value.denominator)})"

        def _format_asserted_weight_assumption(step: int, transition: str, lower: Fraction, upper: Fraction) -> None:
            weight_symbol = f"weight_{step}_{transition}"
            smt2_lines.append(f"(assume (>= {weight_symbol} {_as_smt_real(lower)}))")
            smt2_lines.append(f"(assume (<= {weight_symbol} {_as_smt_real(upper)}))")

        smt2_lines: list[str] = ["(set-logic QF_NRA)"]
        for step in range(max_depth + 1):
            for p_idx in range(len(self.place_names)):
                smt2_lines.append(f"(declare-fun m_{step}_{p_idx} () Real)")
            if step < max_depth:
                smt2_lines.append(f"(declare-fun idle_{step} () Bool)")
                for t_idx, transition in enumerate(self.transition_names):
                    smt2_lines.append(f"(declare-fun fire_{step}_{transition} () Bool)")
                    smt2_lines.append(f"(declare-fun weight_{step}_{transition} () Real)")

        for p_idx, value in enumerate(self._initial):
            smt2_lines.append(f"(assert (= m_0_{p_idx} {_as_smt_real(value)}))")

        for step in range(max_depth + 1):
            for p_idx in range(len(self.place_names)):
                smt2_lines.append(f"(assert (>= m_{step}_{p_idx} 0.0))")

        for step in range(max_depth):
            firing_terms = [f"fire_{step}_{transition}" for transition in self.transition_names]
            if firing_terms:
                smt2_lines.append(f"(assert (or idle_{step} {' '.join(firing_terms)}))")
                for left_idx, left in enumerate(firing_terms):
                    for right in firing_terms[left_idx + 1 :]:
                        smt2_lines.append(f"(assert (not (and {left} {right})))")
            else:
                smt2_lines.append(f"(assert idle_{step})")

            transition_names = list(self._transition_index.keys())
            for transition in transition_names:
                t_idx = self._transition_index[transition]
                transition_weight = f"weight_{step}_{transition}"
                lower, upper = validated_bounds[transition]
                _format_asserted_weight_assumption(step, transition, lower, upper)

                for p_idx, amount in self._inputs[t_idx].items():
                    amount_expr = _as_smt_real(amount)
                    smt2_lines.append(
                        f"(assert (=> fire_{step}_{transition} (>= m_{step}_{p_idx} (* {transition_weight} {amount_expr})))"
                    )
                for p_idx, threshold in self._inhibitors[t_idx].items():
                    threshold_expr = _as_smt_real(threshold)
                    smt2_lines.append(
                        f"(assert (=> fire_{step}_{transition} (< m_{step}_{p_idx} (* {transition_weight} {threshold_expr})))"
                    )
                for p_idx in range(len(self.place_names)):
                    input_delta = self._inputs[t_idx].get(p_idx, Fraction(0))
                    output_delta = self._outputs[t_idx].get(p_idx, Fraction(0))
                    delta_expr = _as_smt_real(output_delta - input_delta)
                    smt2_lines.append(
                        f"(assert (=> fire_{step}_{transition} (= m_{step + 1}_{p_idx} "
                        f"(+ m_{step}_{p_idx} (* {transition_weight} {delta_expr}))))"
                    )
            for p_idx in range(len(self.place_names)):
                smt2_lines.append(f"(assert (=> idle_{step} (= m_{step + 1}_{p_idx} m_{step}_{p_idx})))")

        if not self.transition_names:
            smt2_lines.append("(assert false) ; no transitions means vacuous safety contract")

        if trigger_transition is not None and response_transition is not None:
            t_idx = self._transition_index[trigger_transition]
            r_idx = self._transition_index[response_transition]
            for step in range(max_depth - max_latency_steps + 1):
                response_window_start = step + 1
                response_window_end = min(max_depth, step + max_latency_steps + 1)
                response_window = " ".join(
                    f"fire_{idx}_{self.transition_names[r_idx]}"
                    for idx in range(response_window_start, response_window_end)
                )
                if response_window:
                    smt2_lines.append(
                        f"(assert (=> fire_{step}_{self.transition_names[t_idx]} (or {response_window})))"
                    )
                else:
                    smt2_lines.append(f"(assert (not fire_{step}_{self.transition_names[t_idx]}))")

        for step in range(max_depth - no_stall_steps + 1):
            stall_window = " ".join(f"idle_{step + offset}" for offset in range(no_stall_steps))
            smt2_lines.append(f"(assert (not (and {stall_window})))")

        smt2_lines.append("(check-sat)")
        smt2_body = "\n".join(line for line in smt2_lines if line.strip())
        sby_config = "\n".join(
            [
                "[options]",
                "mode bmc",
                f"depth {max_depth}",
                "append 1",
                "",
                "[engines]",
                "smtbmc",
                "",
                "[script]",
                "read_smt2 contract.smt2",
                "prep -top top",
                "flatten",
                "bmc",
                "",
                "[files]",
                "contract.smt2",
            ]
        )
        metadata = json.dumps(
            {
                "schema_version": SYMBIYOSYS_SYMBOLIC_CONTRACT_VERSION,
                "rti_contract_budget_ns": RTI_CONTRACT_BUDGET_NS,
                "max_depth": max_depth,
                "tick_period_ns": tick_period_ns,
                "max_latency_ns": max_latency_ns,
                "no_stall_window_ns": no_stall_window_ns,
                "trigger_transition": trigger_transition,
                "response_transition": response_transition,
                "weight_bounds": {k: [float(lo), float(hi)] for k, (lo, hi) in validated_bounds.items()},
            },
            sort_keys=True,
        )
        return {
            "sby": sby_config,
            "smt2": smt2_body,
            "metadata": metadata,
        }

    def _path_from_model(self, model: Any, firings: list[list[Any]], max_depth: int) -> list[str]:
        path: list[str] = []
        for step in range(max_depth):
            for transition, t_idx in self._transition_index.items():
                if bool(model.evaluate(firings[step][t_idx], model_completion=True)):
                    path.append(transition)
                    break
        return path

    def _marking_from_model(self, z3: Any, model: Any, marking_vars: list[Any]) -> dict[str, Fraction]:
        marking: dict[str, Fraction] = {}
        for place, variable in zip(self.place_names, marking_vars, strict=True):
            value = model.evaluate(variable, model_completion=True)
            if z3.is_rational_value(value):
                marking[place] = Fraction(value.numerator_as_long(), value.denominator_as_long())
            else:
                marking[place] = Fraction(str(value))
        return marking

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

    def _validate_weight_bounds(
        self, weight_bounds: dict[str, tuple[float, float]]
    ) -> dict[str, tuple[Fraction, Fraction]]:
        normalized: dict[str, tuple[Fraction, Fraction]] = {}
        for transition in self.transition_names:
            normalized[transition] = (Fraction(1), Fraction(1))
        for transition, (lower, upper) in weight_bounds.items():
            self._require_transition(transition)
            if lower > upper:
                raise ValueError(f"weight bound lower exceeds upper bound for transition {transition!r}")
            normalized[transition] = (_as_fraction(lower), _as_fraction(upper))
            if normalized[transition][0] < 0:
                raise ValueError(f"weight bound for transition {transition!r} must be non-negative")
        return normalized

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


def verify_z3_formal_contracts(
    net: StochasticPetriNet,
    *,
    max_depth: int,
    marking_bounds: dict[str, tuple[float, float]],
    temporal_specs: list[AlwaysBounded | EventuallyFires | NeverCoMarked | AlwaysEventuallyMarked | FireLeadsToMarking]
    | None = None,
    weight_bounds: dict[str, tuple[float, float]] | None = None,
) -> Z3FormalVerificationReport:
    """Run bounded Z3 safety and temporal proof obligations."""
    checker = Z3BoundedModelChecker(net)
    safety = checker.prove_marking_bounds(marking_bounds, max_depth=max_depth, weight_bounds=weight_bounds)
    temporal = checker.verify_temporal_specs(temporal_specs or [], max_depth=max_depth, weight_bounds=weight_bounds)
    return Z3FormalVerificationReport(
        holds=safety.holds and temporal.holds,
        backend="z3",
        max_depth=max_depth,
        safety=safety,
        temporal=temporal,
    )


def write_z3_formal_report(
    report: Z3FormalVerificationReport,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Persist a bounded SMT report as JSON and Markdown evidence artifacts."""
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_z3_formal_report_payload(report)
    json_target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_target.write_text(_render_markdown(payload), encoding="utf-8")


def build_z3_formal_report_payload(report: Z3FormalVerificationReport) -> dict[str, Any]:
    """Return the schema-versioned JSON payload for a bounded Z3 report."""
    safety = asdict(report.safety)
    temporal = asdict(report.temporal)
    checked_specs = _checked_specs(safety, temporal)
    payload: dict[str, Any] = {
        "schema_version": Z3_FORMAL_REPORT_SCHEMA_VERSION,
        "status": "pass" if report.holds else "fail",
        "backend": report.backend,
        "solver": _z3_solver_label(),
        "holds": report.holds,
        "max_depth": report.max_depth,
        "checked_specs": checked_specs,
        "scope": Z3_FORMAL_REPORT_SCOPE,
        "claim_boundary": Z3_FORMAL_REPORT_CLAIM_BOUNDARY,
        "safety": safety,
        "temporal": temporal,
    }
    payload["payload_sha256"] = _payload_digest(payload)
    return validate_z3_formal_report_payload(payload)


def build_blocked_z3_formal_report_payload(reason: str) -> dict[str, Any]:
    """Return a schema-versioned blocked report for unavailable SMT evidence."""
    payload: dict[str, Any] = {
        "schema_version": Z3_FORMAL_REPORT_SCHEMA_VERSION,
        "status": "blocked",
        "backend": "z3",
        "solver": _Z3_BLOCKED_SOLVER_LABEL,
        "holds": False,
        "max_depth": 0,
        "checked_specs": list(_Z3_BLOCKED_CHECKED_SPECS),
        "scope": Z3_FORMAL_REPORT_SCOPE,
        "claim_boundary": Z3_FORMAL_REPORT_CLAIM_BOUNDARY,
        "reason": reason,
        "safety": None,
        "temporal": None,
    }
    payload["payload_sha256"] = _payload_digest(payload)
    return validate_z3_formal_report_payload(payload)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_unknown_keys(payload: dict[str, Any], *, allowed: frozenset[str], context: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"{context} unknown fields: {', '.join(unknown)}")


def _reject_missing_keys(payload: dict[str, Any], *, required: frozenset[str], context: str) -> None:
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{context} missing fields: {', '.join(missing)}")


def _is_finite_json_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))


def _validate_z3_violation_record(violation: Any, *, context: str) -> None:
    if not isinstance(violation, dict):
        raise ValueError(f"{context} violation must be an object")
    _reject_unknown_keys(violation, allowed=_Z3_REPORT_VIOLATION_KEYS, context=f"{context} violation")
    _reject_missing_keys(violation, required=_Z3_REPORT_VIOLATION_KEYS, context=f"{context} violation")
    if not isinstance(violation["property_name"], str) or not violation["property_name"]:
        raise ValueError(f"{context} violation property_name must be a non-empty string")
    if not isinstance(violation["message"], str) or not violation["message"]:
        raise ValueError(f"{context} violation message must be a non-empty string")
    marking = violation["marking"]
    if not isinstance(marking, dict):
        raise ValueError(f"{context} violation marking must be an object")
    for place, value in marking.items():
        if not isinstance(place, str) or not place:
            raise ValueError(f"{context} violation marking keys must be non-empty strings")
        if not _is_finite_json_number(value):
            raise ValueError(f"{context} violation marking values must be finite numbers")
    path = violation["path"]
    if not isinstance(path, list) or any(not isinstance(step, str) or not step for step in path):
        raise ValueError(f"{context} violation path must contain transition-name strings")
    for optional_name in ("place", "transition"):
        value = violation[optional_name]
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{context} violation {optional_name} must be null or a non-empty string")


def _validate_z3_section_solver_consistency(section: dict[str, Any], *, context: str) -> None:
    solver_status = section["solver_status"]
    holds = section["holds"]
    violations = section["violations"]
    if solver_status == "unsat":
        if not holds:
            raise ValueError(f"{context} unsat section must hold")
        if violations:
            raise ValueError(f"{context} unsat section must not carry violations")
        return
    if solver_status == "sat":
        if holds:
            raise ValueError(f"{context} sat section must not hold")
        if not violations:
            raise ValueError(f"{context} sat section must carry counterexample violations")
        return
    if holds:
        raise ValueError(f"{context} unknown section must not hold")


def _validate_z3_section_checked_specs(checked_specs: list[Any], *, context: str) -> None:
    if any(not isinstance(spec, str) or not spec for spec in checked_specs):
        raise ValueError(f"{context} checked_specs must contain non-empty strings")
    if len(checked_specs) != len(set(checked_specs)):
        raise ValueError(f"{context} checked_specs must be unique")


def load_z3_formal_report(path: str | Path) -> dict[str, Any]:
    """Load and validate a duplicate-key-safe Z3 formal evidence report."""
    report_path = Path(path)
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise ValueError("Z3 formal report must be a JSON object")
    return validate_z3_formal_report_payload(payload)


def validate_z3_formal_report_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a schema-versioned Z3 formal evidence report."""
    if payload.get("schema_version") != Z3_FORMAL_REPORT_SCHEMA_VERSION:
        raise ValueError("Z3 formal report schema_version is unsupported")
    if payload.get("backend") != "z3":
        raise ValueError("Z3 formal report backend must be 'z3'")
    status = payload.get("status")
    if status not in {"pass", "fail", "blocked"}:
        raise ValueError("Z3 formal report status must be pass, fail, or blocked")
    _reject_unknown_keys(
        payload,
        allowed=_Z3_REPORT_BLOCKED_KEYS if status == "blocked" else _Z3_REPORT_PASS_FAIL_KEYS,
        context="Z3 formal report",
    )
    if not isinstance(payload.get("solver"), str) or not payload["solver"]:
        raise ValueError("Z3 formal report solver must be a non-empty string")
    if not isinstance(payload.get("holds"), bool):
        raise ValueError("Z3 formal report holds must be a boolean")
    if isinstance(payload.get("max_depth"), bool) or not isinstance(payload.get("max_depth"), int):
        raise ValueError("Z3 formal report max_depth must be an integer")
    if payload["max_depth"] < 0:
        raise ValueError("Z3 formal report max_depth must be non-negative")
    if not isinstance(payload.get("checked_specs"), list) or not payload["checked_specs"]:
        raise ValueError("Z3 formal report checked_specs must be a non-empty list")
    if any(not isinstance(spec, str) or not spec for spec in payload["checked_specs"]):
        raise ValueError("Z3 formal report checked_specs must contain non-empty strings")
    if len(payload["checked_specs"]) != len(set(payload["checked_specs"])):
        raise ValueError("Z3 formal report checked_specs must be unique")
    if payload.get("scope") != Z3_FORMAL_REPORT_SCOPE:
        raise ValueError("Z3 formal report scope is unsupported")
    if payload.get("claim_boundary") != Z3_FORMAL_REPORT_CLAIM_BOUNDARY:
        raise ValueError("Z3 formal report claim_boundary is unsupported")
    declared_digest = payload.get("payload_sha256")
    if not isinstance(declared_digest, str) or len(declared_digest) != 64:
        raise ValueError("Z3 formal report payload_sha256 must be a SHA-256 hex digest")
    if _payload_digest(payload) != declared_digest.lower():
        raise ValueError("Z3 formal report payload_sha256 does not match payload")
    if status == "blocked":
        if payload["holds"]:
            raise ValueError("blocked Z3 formal report must not hold")
        if payload["solver"] != _Z3_BLOCKED_SOLVER_LABEL:
            raise ValueError("blocked Z3 formal report solver must be z3-solver unavailable")
        if payload["max_depth"] != 0:
            raise ValueError("blocked Z3 formal report max_depth must be 0")
        if payload["checked_specs"] != _Z3_BLOCKED_CHECKED_SPECS:
            raise ValueError("blocked Z3 formal report checked_specs must only contain z3_solver_available")
        if not isinstance(payload.get("reason"), str) or not payload["reason"]:
            raise ValueError("blocked Z3 formal report must include a reason")
        if payload.get("safety") is not None or payload.get("temporal") is not None:
            raise ValueError("blocked Z3 formal report must not carry proof sections")
        return payload
    if status == "pass" and not payload["holds"]:
        raise ValueError("passing Z3 formal report must hold")
    if status == "fail" and payload["holds"]:
        raise ValueError("failed Z3 formal report must not hold")
    for section_name in ("safety", "temporal"):
        section = payload.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"Z3 formal report {section_name} section must be an object")
        _reject_unknown_keys(section, allowed=_Z3_REPORT_SECTION_KEYS, context=f"Z3 formal report {section_name}")
        if section.get("backend") != "z3":
            raise ValueError(f"Z3 formal report {section_name} backend must be 'z3'")
        if section.get("max_depth") != payload["max_depth"]:
            raise ValueError(f"Z3 formal report {section_name} depth must match report depth")
        if section.get("solver_status") not in {"sat", "unsat", "unknown"}:
            raise ValueError(f"Z3 formal report {section_name} solver_status is unsupported")
        if not isinstance(section.get("holds"), bool):
            raise ValueError(f"Z3 formal report {section_name} holds must be a boolean")
        if not isinstance(section.get("violations"), list):
            raise ValueError(f"Z3 formal report {section_name} violations must be a list")
        for violation in section["violations"]:
            _validate_z3_violation_record(violation, context=f"Z3 formal report {section_name}")
        _validate_z3_section_solver_consistency(section, context=f"Z3 formal report {section_name}")
        if not isinstance(section.get("checked_specs"), list):
            raise ValueError(f"Z3 formal report {section_name} checked_specs must be a list")
        _validate_z3_section_checked_specs(section["checked_specs"], context=f"Z3 formal report {section_name}")
    if payload["holds"] != (payload["safety"]["holds"] and payload["temporal"]["holds"]):
        raise ValueError("Z3 formal report holds must match safety and temporal sections")
    if payload["checked_specs"] != _checked_specs(payload["safety"], payload["temporal"]):
        raise ValueError("Z3 formal report checked_specs must match proof sections")
    return payload


def _checked_specs(safety: dict[str, Any], temporal: dict[str, Any]) -> list[str]:
    specs = ["marking_bounds"]
    for section in (safety, temporal):
        for spec in section.get("checked_specs", []):
            if spec not in specs:
                specs.append(spec)
    return specs


def _payload_digest(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("payload_sha256", None)
    blob = json.dumps(canonical, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _z3_solver_label() -> str:
    try:
        import z3
    except ModuleNotFoundError:
        return "z3-solver unavailable"
    version = getattr(z3, "get_version_string", lambda: "unknown")()
    return f"z3-solver {version}"


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SCPN Z3 Formal Verification Report",
        "",
        f"- Schema: `{payload['schema_version']}`",
        f"- Status: `{payload['status']}`",
        f"- Backend: `{payload['backend']}`",
        f"- Solver: `{payload['solver']}`",
        f"- Max depth: `{payload['max_depth']}`",
        f"- Payload SHA-256: `{payload['payload_sha256']}`",
        f"- Scope: {payload['scope']}.",
        f"- Claim boundary: {payload['claim_boundary']}.",
        "",
    ]
    if payload["status"] == "blocked":
        lines.extend([f"- Reason: {payload['reason']}", ""])
        return "\n".join(lines)
    lines.extend(
        [
            "## Safety",
            "",
            f"- Holds: `{payload['safety']['holds']}`",
            f"- Solver status: `{payload['safety']['solver_status']}`",
            "",
            "## Temporal",
            "",
            f"- Holds: `{payload['temporal']['holds']}`",
            f"- Solver status: `{payload['temporal']['solver_status']}`",
            f"- Checked specs: `{', '.join(payload['checked_specs'])}`",
            "",
        ]
    )
    violations = [*payload["safety"]["violations"], *payload["temporal"]["violations"]]
    if violations:
        lines.extend(["## Counterexamples", ""])
        for violation in violations:
            lines.append(f"- `{violation['property_name']}` path={violation['path']} message={violation['message']}")
        lines.append("")
    return "\n".join(lines)
