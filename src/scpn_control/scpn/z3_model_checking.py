# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Z3 Petri Net Model Checking

"""Bounded SMT model checking for compiled SCPN Petri-net control logic."""

from __future__ import annotations

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
    ) -> Z3ModelCheckingReport:
        """Prove marking bounds by asking Z3 for a reachable counterexample."""
        self._validate_max_depth(max_depth)
        self._validate_bounds(bounds)
        z3, solver, markings, firings, _idle = self._transition_system(max_depth)
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
    ) -> Z3ModelCheckingReport:
        """Verify supported bounded temporal specifications with Z3."""
        self._validate_max_depth(max_depth)
        checked: list[str] = []
        for spec in specs:
            checked.append(spec.name)
            if isinstance(spec, AlwaysBounded):
                report = self.prove_marking_bounds(spec.bounds, max_depth=max_depth)
                if not report.holds:
                    return Z3ModelCheckingReport(
                        False, "z3", max_depth, report.solver_status, report.violations, checked
                    )
            elif isinstance(spec, EventuallyFires):
                report = self._prove_eventually_fires(spec, max_depth=max_depth)
                if not report.holds:
                    return Z3ModelCheckingReport(
                        False, "z3", max_depth, report.solver_status, report.violations, checked
                    )
            elif isinstance(spec, NeverCoMarked):
                report = self._prove_never_comarked(spec, max_depth=max_depth)
                if not report.holds:
                    return Z3ModelCheckingReport(
                        False, "z3", max_depth, report.solver_status, report.violations, checked
                    )
            elif isinstance(spec, FireLeadsToMarking):
                report = self._prove_fire_leads_to_marking(spec, max_depth=max_depth)
                if not report.holds:
                    return Z3ModelCheckingReport(
                        False, "z3", max_depth, report.solver_status, report.violations, checked
                    )
            elif isinstance(spec, AlwaysEventuallyMarked):
                report = self._prove_always_eventually_marked(spec, max_depth=max_depth)
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

    def _prove_eventually_fires(self, spec: EventuallyFires, *, max_depth: int) -> Z3ModelCheckingReport:
        self._require_transition(spec.transition)
        z3, solver, markings, firings, _idle = self._transition_system(max_depth)
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

    def _prove_never_comarked(self, spec: NeverCoMarked, *, max_depth: int) -> Z3ModelCheckingReport:
        self._require_place(spec.place_a)
        self._require_place(spec.place_b)
        z3, solver, markings, firings, _idle = self._transition_system(max_depth)
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

    def _prove_fire_leads_to_marking(self, spec: FireLeadsToMarking, *, max_depth: int) -> Z3ModelCheckingReport:
        self._require_transition(spec.trigger_transition)
        self._require_place(spec.target_place)
        if isinstance(spec.within, bool) or int(spec.within) != spec.within or spec.within < 0:
            raise ValueError("FireLeadsToMarking.within must be an integer >= 0")
        z3, solver, markings, firings, _idle = self._transition_system(max_depth)
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

    def _prove_always_eventually_marked(self, spec: AlwaysEventuallyMarked, *, max_depth: int) -> Z3ModelCheckingReport:
        self._require_place(spec.place)
        z3, solver, markings, firings, _idle = self._transition_system(max_depth)
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

    def _transition_system(self, max_depth: int) -> tuple[Any, Any, list[list[Any]], list[list[Any]], list[Any]]:
        z3 = _require_z3()
        solver = z3.Solver()
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
                constraints = []
                for p_idx, amount in self._inputs[t_idx].items():
                    constraints.append(markings[step][p_idx] >= _z3_fraction(z3, amount))
                for p_idx, threshold in self._inhibitors[t_idx].items():
                    constraints.append(markings[step][p_idx] < _z3_fraction(z3, threshold))
                for p_idx in range(len(self.place_names)):
                    delta = self._outputs[t_idx].get(p_idx, Fraction(0)) - self._inputs[t_idx].get(p_idx, Fraction(0))
                    constraints.append(markings[step + 1][p_idx] == markings[step][p_idx] + _z3_fraction(z3, delta))
                if not constraints:
                    raise RuntimeError(f"transition {transition!r} produced an empty SMT constraint set")
                solver.add(z3.Implies(firings[step][t_idx], z3.And(*constraints)))
        return z3, solver, markings, firings, idle

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
) -> Z3FormalVerificationReport:
    """Run bounded Z3 safety and temporal proof obligations."""
    checker = Z3BoundedModelChecker(net)
    safety = checker.prove_marking_bounds(marking_bounds, max_depth=max_depth)
    temporal = checker.verify_temporal_specs(temporal_specs or [], max_depth=max_depth)
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
        "solver": "z3-solver unavailable",
        "holds": False,
        "max_depth": 0,
        "checked_specs": ["z3_solver_available"],
        "scope": Z3_FORMAL_REPORT_SCOPE,
        "claim_boundary": Z3_FORMAL_REPORT_CLAIM_BOUNDARY,
        "reason": reason,
        "safety": None,
        "temporal": None,
    }
    payload["payload_sha256"] = _payload_digest(payload)
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
        if not isinstance(section.get("checked_specs"), list):
            raise ValueError(f"Z3 formal report {section_name} checked_specs must be a list")
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
