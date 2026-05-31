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
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from scpn_control.scpn.formal_verification import (
    AlwaysBounded,
    AlwaysEventuallyMarked,
    EventuallyFires,
    FireLeadsToMarking,
    FormalViolation,
    NeverCoMarked,
    _as_float_marking,
    _as_fraction,
)
from scpn_control.scpn.structure import StochasticPetriNet


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
    json_target.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_target.write_text(_render_markdown(report), encoding="utf-8")


def _render_markdown(report: Z3FormalVerificationReport) -> str:
    status = "pass" if report.holds else "fail"
    lines = [
        "# SCPN Z3 Formal Verification Report",
        "",
        f"- Status: `{status}`",
        f"- Backend: `{report.backend}`",
        f"- Max depth: `{report.max_depth}`",
        "- Scope: bounded SMT evidence for compiled Petri-net control logic.",
        "- Claim boundary: not hardware timing evidence, PCS certification, or unbounded liveness proof.",
        "",
        "## Safety",
        "",
        f"- Holds: `{report.safety.holds}`",
        f"- Solver status: `{report.safety.solver_status}`",
        "",
        "## Temporal",
        "",
        f"- Holds: `{report.temporal.holds}`",
        f"- Solver status: `{report.temporal.solver_status}`",
        f"- Checked specs: `{', '.join(report.temporal.checked_specs)}`",
        "",
    ]
    violations = [*report.safety.violations, *report.temporal.violations]
    if violations:
        lines.extend(["## Counterexamples", ""])
        for violation in violations:
            lines.append(f"- `{violation.property_name}` path={violation.path} message={violation.message}")
        lines.append("")
    return "\n".join(lines)
