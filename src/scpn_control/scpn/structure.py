# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Stochastic Petri net structure.
"""
Packet A — Stochastic Petri Net structure definition.

Pure Python + numpy + scipy.  No sc_neurocore dependency.

Builds two sparse matrices that encode the net topology:
    W_in  : (n_transitions, n_places)  — input arc weights
    W_out : (n_places, n_transitions)  — output arc weights
"""

from __future__ import annotations

from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from scpn_control._typing import AnyFloatArray, FloatArray


class _NodeKind(Enum):
    PLACE = auto()
    TRANSITION = auto()


class StochasticPetriNet:
    """Stochastic Petri Net with sparse matrix representation.

    Usage::

        net = StochasticPetriNet()
        net.add_place("P_red",   initial_tokens=1.0)
        net.add_place("P_green", initial_tokens=0.0)
        net.add_transition("T_r2g", threshold=0.5)
        net.add_arc("P_red", "T_r2g", weight=1.0)
        net.add_arc("T_r2g", "P_green", weight=1.0)
        net.compile()
    """

    def __init__(self) -> None:
        # Ordered registries -------------------------------------------------
        self._places: list[str] = []
        self._place_tokens: list[float] = []
        self._place_idx: dict[str, int] = {}

        self._transitions: list[str] = []
        self._transition_thresholds: list[float] = []
        self._transition_delays: list[int] = []
        self._transition_idx: dict[str, int] = {}

        # Node kind lookup (for arc validation) --------------------------------
        self._kind: dict[str, _NodeKind] = {}

        # Arc storage (resolved at compile time) --------------------------------
        # Each arc: (source_name, target_name, weight, inhibitor_flag)
        self._arcs: list[tuple[str, str, float, bool]] = []

        # Compiled products ----------------------------------------------------
        self.W_in: sparse.csr_matrix | None = None  # (nT, nP)
        self.W_out: sparse.csr_matrix | None = None  # (nP, nT)
        self._compiled: bool = False
        self._last_validation_report: dict[str, object] | None = None

    # ── Builder API ──────────────────────────────────────────────────────────

    def add_place(self, name: str, initial_tokens: float = 0.0) -> None:
        """Add a place (state variable) with token density in [0, 1]."""
        if name in self._kind:
            raise ValueError(f"Node '{name}' already exists.")
        if not 0.0 <= initial_tokens <= 1.0:
            raise ValueError(f"initial_tokens must be in [0, 1], got {initial_tokens}")
        idx = len(self._places)
        self._places.append(name)
        self._place_tokens.append(initial_tokens)
        self._place_idx[name] = idx
        self._kind[name] = _NodeKind.PLACE
        self._compiled = False

    def add_transition(self, name: str, threshold: float = 0.5, delay_ticks: int = 0) -> None:
        """Add a transition (logic gate) with a firing threshold."""
        if name in self._kind:
            raise ValueError(f"Node '{name}' already exists.")
        if threshold < 0.0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        if delay_ticks < 0:
            raise ValueError(f"delay_ticks must be >= 0, got {delay_ticks}")
        idx = len(self._transitions)
        self._transitions.append(name)
        self._transition_thresholds.append(threshold)
        self._transition_delays.append(int(delay_ticks))
        self._transition_idx[name] = idx
        self._kind[name] = _NodeKind.TRANSITION
        self._compiled = False

    def add_arc(
        self,
        source: str,
        target: str,
        weight: float = 1.0,
        inhibitor: bool = False,
    ) -> None:
        """Add a directed arc between a Place and a Transition (either direction).

        Valid arcs:
            Place      -> Transition  (input arc,  stored in W_in)
            Transition -> Place       (output arc, stored in W_out)

        Parameters
        ----------
        inhibitor : bool, default False
            If True, arc is encoded as a negative weight inhibitor input arc.
            Only valid for ``Place -> Transition`` edges. Controller artifact
            export and runtime admission reject this negative matrix encoding
            until the artifact topology carries inhibitor arcs explicitly.

        Raises ``ValueError`` for same-kind connections or unknown nodes.
        """
        if source not in self._kind:
            raise ValueError(f"Unknown node '{source}'.")
        if target not in self._kind:
            raise ValueError(f"Unknown node '{target}'.")

        src_kind = self._kind[source]
        tgt_kind = self._kind[target]

        if src_kind == tgt_kind:
            raise ValueError(
                f"Arc must connect Place<->Transition, got {src_kind.name}->{tgt_kind.name} ('{source}'->'{target}')."
            )
        if inhibitor:
            if not (src_kind is _NodeKind.PLACE and tgt_kind is _NodeKind.TRANSITION):
                raise ValueError("inhibitor arcs are only supported for Place->Transition.")
            if weight <= 0.0:
                raise ValueError(f"inhibitor arc weight must be > 0 (magnitude), got {weight}")
            stored_weight = -abs(float(weight))
        else:
            if weight <= 0.0:
                raise ValueError(f"weight must be > 0, got {weight}")
            stored_weight = float(weight)

        self._arcs.append((source, target, stored_weight, bool(inhibitor)))
        self._compiled = False

    # ── Compile ──────────────────────────────────────────────────────────────

    def compile(
        self,
        validate_topology: bool = False,
        strict_validation: bool = False,
        allow_inhibitor: bool = False,
    ) -> None:
        """Build sparse W_in and W_out matrices from the arc list.

        Parameters
        ----------
        validate_topology : bool, default False
            Compute and store topology diagnostics in
            :attr:`last_validation_report`.
        strict_validation : bool, default False
            If True, raise ``ValueError`` when topology diagnostics contain
            dead nodes or unseeded place cycles.
        allow_inhibitor : bool, default False
            Allow negative ``Place -> Transition`` weights for inhibitor arcs.
            If False, compile fails when inhibitor arcs are present. This mode
            is suitable for structure/formal-analysis paths; controller
            artifacts fail closed on negative input weights.
        """
        nP = len(self._places)
        nT = len(self._transitions)
        if nP == 0 or nT == 0:
            raise ValueError("Net must have at least one place and one transition.")

        self._last_validation_report = None
        if validate_topology or strict_validation:
            report = self.validate_topology()
            self._last_validation_report = report
            has_issues = bool(
                report["dead_places"]
                or report["dead_transitions"]
                or report["unseeded_place_cycles"]
                or report["input_weight_overflow_transitions"]
            )
            if strict_validation and has_issues:
                issues: list[str] = []
                if report["dead_places"]:
                    issues.append(f"dead_places={report['dead_places']}")
                if report["dead_transitions"]:
                    issues.append(f"dead_transitions={report['dead_transitions']}")
                if report["unseeded_place_cycles"]:
                    issues.append(f"unseeded_place_cycles={report['unseeded_place_cycles']}")
                if report["input_weight_overflow_transitions"]:
                    issues.append(f"input_weight_overflow_transitions={report['input_weight_overflow_transitions']}")
                raise ValueError("Topology validation failed: " + "; ".join(issues))

        # COO accumulators
        in_rows: list[int] = []
        in_cols: list[int] = []
        in_vals: list[float] = []

        out_rows: list[int] = []
        out_cols: list[int] = []
        out_vals: list[float] = []

        for src, tgt, w, is_inhibitor in self._arcs:
            if self._kind[src] is _NodeKind.PLACE:
                if w < 0.0 and not allow_inhibitor:
                    raise ValueError("Negative input arc weights detected; re-run compile with allow_inhibitor=True.")
                # Place -> Transition  => W_in[transition_idx, place_idx]
                t_idx = self._transition_idx[tgt]
                p_idx = self._place_idx[src]
                in_rows.append(t_idx)
                in_cols.append(p_idx)
                in_vals.append(w)
            else:
                # Defensive: add_arc rejects inhibitor arcs on Transition->Place
                # edges and any non-positive weight, so a T->P edge reaching here
                # can be neither an inhibitor nor negative-weight arc.
                if is_inhibitor or w < 0.0:
                    raise ValueError("Inhibitor arcs are only valid on Place->Transition edges.")
                # Transition -> Place  => W_out[place_idx, transition_idx]
                t_idx = self._transition_idx[src]
                p_idx = self._place_idx[tgt]
                out_rows.append(p_idx)
                out_cols.append(t_idx)
                out_vals.append(w)

        self.W_in = self._csr_from_triplets(in_rows, in_cols, in_vals, shape=(nT, nP))
        self.W_out = self._csr_from_triplets(out_rows, out_cols, out_vals, shape=(nP, nT))
        self._compiled = True

    @staticmethod
    def _csr_from_triplets(
        row_indices: list[int],
        col_indices: list[int],
        values: list[float],
        *,
        shape: tuple[int, int],
    ) -> sparse.csr_matrix:
        """Build a float64 CSR matrix from COO-style triplets.

        The direct CSR construction avoids SciPy's COO index-dtype inference path,
        which can be brittle when test instrumentation reloads NumPy. Duplicate
        triplets are coalesced to match COO-to-CSR summation semantics.
        """
        n_rows, n_cols = shape
        if not values:
            return sparse.csr_matrix(shape, dtype=np.float64)

        triplets = sorted(zip(row_indices, col_indices, values, strict=True), key=lambda item: (item[0], item[1]))
        coalesced_rows: list[int] = []
        coalesced_cols: list[int] = []
        coalesced_values: list[float] = []

        current_row, current_col, current_value = triplets[0]
        for row, col, value in triplets[1:]:
            if row == current_row and col == current_col:
                current_value += value
                continue
            coalesced_rows.append(current_row)
            coalesced_cols.append(current_col)
            coalesced_values.append(current_value)
            current_row, current_col, current_value = row, col, value

        coalesced_rows.append(current_row)
        coalesced_cols.append(current_col)
        coalesced_values.append(current_value)

        index_dtype = np.int32 if max(n_rows, n_cols, len(coalesced_cols)) <= np.iinfo(np.int32).max else np.int64
        indptr = np.zeros(n_rows + 1, dtype=index_dtype)
        for row in coalesced_rows:
            indptr[row + 1] += 1
        np.cumsum(indptr, out=indptr)

        indices = np.asarray(coalesced_cols, dtype=index_dtype)
        data = np.asarray(coalesced_values, dtype=np.float64)
        return sparse.csr_matrix((data, indices, indptr), shape=(n_rows, n_cols), dtype=np.float64)

    def validate_topology(self) -> dict[str, object]:
        """Return topology diagnostics without mutating compiled matrices.

        Diagnostics:
        - ``dead_places``: places with zero total degree.
        - ``dead_transitions``: transitions with zero total degree.
        - ``unseeded_place_cycles``: place-level SCC cycles with no initial
          token mass.
        - ``input_weight_overflow_transitions``: transitions whose positive
          input arc-weight sum exceeds 1.0.
        """
        place_in_deg = {p: 0 for p in self._places}
        place_out_deg = {p: 0 for p in self._places}
        trans_in_deg = {t: 0 for t in self._transitions}
        trans_out_deg = {t: 0 for t in self._transitions}
        transition_positive_input_weight_sum = {t: 0.0 for t in self._transitions}

        transition_inputs: dict[str, list[str]] = {t: [] for t in self._transitions}
        transition_outputs: dict[str, list[str]] = {t: [] for t in self._transitions}

        for src, tgt, w, _is_inhibitor in self._arcs:
            if self._kind[src] is _NodeKind.PLACE:
                place_out_deg[src] += 1
                trans_in_deg[tgt] += 1
                transition_inputs[tgt].append(src)
                if w > 0.0:
                    transition_positive_input_weight_sum[tgt] += float(w)
            else:
                trans_out_deg[src] += 1
                place_in_deg[tgt] += 1
                transition_outputs[src].append(tgt)

        dead_places = sorted(p for p in self._places if (place_in_deg[p] + place_out_deg[p]) == 0)
        dead_transitions = sorted(t for t in self._transitions if (trans_in_deg[t] + trans_out_deg[t]) == 0)

        place_adj: dict[str, set[str]] = {p: set() for p in self._places}
        for t in self._transitions:
            inputs = transition_inputs[t]
            outputs = transition_outputs[t]
            for src_place in inputs:
                for dst_place in outputs:
                    place_adj[src_place].add(dst_place)

        unseeded_place_cycles: list[list[str]] = []
        for comp in self._strongly_connected_components(place_adj):
            has_cycle = len(comp) > 1
            if not has_cycle:
                node = comp[0]
                has_cycle = node in place_adj[node]
            if not has_cycle:
                continue

            if all(self._place_tokens[self._place_idx[p]] <= 0.0 for p in comp):
                unseeded_place_cycles.append(sorted(comp))

        unseeded_place_cycles.sort(key=lambda c: tuple(c))
        input_weight_overflow_transitions = sorted(
            t for t, total_w in transition_positive_input_weight_sum.items() if total_w > (1.0 + 1e-12)
        )
        return {
            "dead_places": dead_places,
            "dead_transitions": dead_transitions,
            "unseeded_place_cycles": unseeded_place_cycles,
            "input_weight_overflow_transitions": input_weight_overflow_transitions,
        }

    @staticmethod
    def _strongly_connected_components(graph: dict[str, set[str]]) -> list[list[str]]:
        """Tarjan SCC for small place graphs."""
        index = 0
        stack: list[str] = []
        on_stack: set[str] = set()
        indices: dict[str, int] = {}
        lowlink: dict[str, int] = {}
        components: list[list[str]] = []

        def strongconnect(v: str) -> None:
            nonlocal index
            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            on_stack.add(v)

            for w in graph[v]:
                if w not in indices:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], indices[w])

            if lowlink[v] == indices[v]:
                comp: list[str] = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    comp.append(w)
                    if w == v:
                        break
                components.append(comp)

        for node in graph:
            if node not in indices:
                strongconnect(node)
        return components

    # ── Accessors ────────────────────────────────────────────────────────────

    @property
    def n_places(self) -> int:
        """Number of places in the net."""
        return len(self._places)

    @property
    def n_transitions(self) -> int:
        """Number of transitions in the net."""
        return len(self._transitions)

    @property
    def place_names(self) -> list[str]:
        """Place names in insertion order."""
        return list(self._places)

    @property
    def transition_names(self) -> list[str]:
        """Transition names in insertion order."""
        return list(self._transitions)

    @property
    def is_compiled(self) -> bool:
        """Whether the net has been compiled."""
        return self._compiled

    @property
    def last_validation_report(self) -> dict[str, object] | None:
        """The most recent validation report, or ``None`` if not yet validated."""
        return self._last_validation_report

    def get_initial_marking(self) -> FloatArray:
        """Return (n_places,) float64 vector of initial token densities."""
        return np.array(self._place_tokens, dtype=np.float64)

    def get_thresholds(self) -> FloatArray:
        """Return (n_transitions,) float64 vector of firing thresholds."""
        return np.array(self._transition_thresholds, dtype=np.float64)

    def get_delay_ticks(self) -> NDArray[np.int64]:
        """Return (n_transitions,) int64 vector of transition delay ticks."""
        return np.array(self._transition_delays, dtype=np.int64)

    def summary(self) -> str:
        """Human-readable summary of the net."""
        lines = [
            f"StochasticPetriNet  "
            f"P={self.n_places}  T={self.n_transitions}  "
            f"Arcs={len(self._arcs)}  compiled={self._compiled}",
            "",
            "Places:",
        ]
        for i, (name, tok) in enumerate(zip(self._places, self._place_tokens)):
            lines.append(f"  [{i}] {name:20s}  tokens={tok:.3f}")

        lines.append("")
        lines.append("Transitions:")
        for i, (name, th) in enumerate(zip(self._transitions, self._transition_thresholds)):
            delay = self._transition_delays[i]
            lines.append(f"  [{i}] {name:20s}  threshold={th:.3f}  delay_ticks={delay}")

        lines.append("")
        lines.append("Arcs:")
        for src, tgt, w, is_inhibitor in self._arcs:
            extra = " [inhibitor]" if is_inhibitor else ""
            lines.append(f"  {src} --({w:.3f})--> {tgt}{extra}")

        if self._compiled:
            assert self.W_in is not None
            assert self.W_out is not None
            lines.append("")
            lines.append(f"W_in  (nT={self.W_in.shape[0]}, nP={self.W_in.shape[1]})  nnz={self.W_in.nnz}")
            lines.append(f"W_out (nP={self.W_out.shape[0]}, nT={self.W_out.shape[1]})  nnz={self.W_out.nnz}")
        return "\n".join(lines)

    # ── Formal verification ───────────────────────────────────────────────

    def verify_boundedness(self, n_steps: int = 500, n_trials: int = 100) -> dict[str, object]:
        """Verify that markings stay in [0, 1] under random firing.

        Runs *n_trials* random trajectories of *n_steps* each,
        checking that all markings remain in [0, 1] at every step.

        Returns
        -------
        dict
            {"bounded": bool, "max_marking": float, "min_marking": float,
             "n_trials": int, "n_steps": int}
        """
        if not self._compiled:
            raise RuntimeError("Net must be compiled before verification.")

        rng = np.random.default_rng(42)
        max_seen = -np.inf
        min_seen = np.inf
        bounded = True

        _W_in = self.W_in
        _W_out = self.W_out
        # Defensive: the _compiled guard above guarantees the matrices are set.
        if _W_in is None or _W_out is None:
            raise RuntimeError("Net must be compiled before verification.")
        W_in_dense: AnyFloatArray = _W_in.toarray() if hasattr(_W_in, "toarray") else np.asarray(_W_in)
        W_out_dense: AnyFloatArray = _W_out.toarray() if hasattr(_W_out, "toarray") else np.asarray(_W_out)

        for _ in range(n_trials):
            marking = self.get_initial_marking().copy()
            for _ in range(n_steps):
                # Check enabling
                thresholds = self.get_thresholds()
                enabled = np.all(W_in_dense <= marking[np.newaxis, :] + 1e-9, axis=1)
                enabled &= rng.random(len(thresholds)) < thresholds

                # Fire enabled transitions
                for t in np.where(enabled)[0]:
                    marking -= W_in_dense[t, :]
                    marking += W_out_dense[:, t]

                # Clamp (as the controller does)
                marking = np.clip(marking, 0.0, 1.0)

                max_seen = max(max_seen, float(np.max(marking)))
                min_seen = min(min_seen, float(np.min(marking)))

        return {
            "bounded": bounded,
            "max_marking": float(max_seen),
            "min_marking": float(min_seen),
            "n_trials": n_trials,
            "n_steps": n_steps,
        }

    def verify_liveness(self, n_steps: int = 200, n_trials: int = 1000) -> dict[str, object]:
        """Verify that all transitions fire at least once in random campaigns.

        The random campaign also validates every raw post-firing marking before
        controller-style clipping. A campaign that reaches a non-finite marking
        or a marking outside ``[0, 1]`` is reported as not live because the
        transition-fire percentages were measured on an invalid walk.

        Returns
        -------
        dict
            {"live": bool, "transition_fire_pct": dict[str, float],
             "min_fire_pct": float, "marking_bounds_valid": bool,
             "marking_violation": dict[str, object] | None}
        """
        if not self._compiled:
            raise RuntimeError("Net must be compiled before verification.")

        n_steps = self._require_positive_verification_count("n_steps", n_steps)
        n_trials = self._require_positive_verification_count("n_trials", n_trials)
        rng = np.random.default_rng(42)
        n_T = self.n_transitions
        fire_counts = np.zeros(n_T, dtype=int)
        min_marking = np.inf
        max_marking = -np.inf
        marking_bounds_valid = True
        marking_violation: dict[str, object] | None = None

        _W_in = self.W_in
        _W_out = self.W_out
        # Defensive: the _compiled guard above guarantees the matrices are set.
        if _W_in is None or _W_out is None:
            raise RuntimeError("Net must be compiled before verification.")
        W_in_dense: AnyFloatArray = _W_in.toarray() if hasattr(_W_in, "toarray") else np.asarray(_W_in)
        W_out_dense: AnyFloatArray = _W_out.toarray() if hasattr(_W_out, "toarray") else np.asarray(_W_out)
        thresholds = self.get_thresholds()

        for trial_index in range(n_trials):
            marking = self.get_initial_marking().copy()
            fired_this_trial = np.zeros(n_T, dtype=bool)

            for step_index in range(n_steps):
                enabled = np.all(W_in_dense <= marking[np.newaxis, :] + 1e-9, axis=1)
                enabled &= rng.random(n_T) < thresholds

                for t in np.where(enabled)[0]:
                    marking -= W_in_dense[t, :]
                    marking += W_out_dense[:, t]
                    fired_this_trial[t] = True
                    min_marking = min(min_marking, float(np.min(marking)))
                    max_marking = max(max_marking, float(np.max(marking)))
                    violation = self._marking_bounds_violation(
                        marking,
                        trial_index=trial_index,
                        step_index=step_index,
                        transition_name=self._transitions[int(t)],
                    )
                    if violation is not None:
                        marking_bounds_valid = False
                        if marking_violation is None:
                            marking_violation = violation

                marking = np.clip(marking, 0.0, 1.0)
                min_marking = min(min_marking, float(np.min(marking)))
                max_marking = max(max_marking, float(np.max(marking)))

            fire_counts += fired_this_trial.astype(int)

        fire_pct = {name: round(float(fire_counts[i]) / n_trials, 4) for i, name in enumerate(self._transitions)}
        min_pct = float(np.min(fire_counts)) / n_trials

        return {
            "live": bool(marking_bounds_valid and min_pct >= 0.99),
            "transition_fire_pct": fire_pct,
            "min_fire_pct": round(min_pct, 4),
            "marking_bounds_valid": marking_bounds_valid,
            "marking_violation": marking_violation,
            "max_marking": round(max_marking, 4),
            "min_marking": round(min_marking, 4),
            "n_trials": n_trials,
            "n_steps": n_steps,
        }

    @staticmethod
    def _require_positive_verification_count(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _marking_bounds_violation(
        marking: FloatArray,
        *,
        trial_index: int,
        step_index: int,
        transition_name: str,
    ) -> dict[str, object] | None:
        if not np.all(np.isfinite(marking)):
            return {
                "reason": "non_finite_marking",
                "trial": trial_index,
                "step": step_index,
                "transition": transition_name,
                "min_marking": float(np.nanmin(marking)),
                "max_marking": float(np.nanmax(marking)),
            }
        min_marking = float(np.min(marking))
        max_marking = float(np.max(marking))
        if min_marking < -1.0e-9 or max_marking > 1.0 + 1.0e-9:
            return {
                "reason": "marking_out_of_bounds",
                "trial": trial_index,
                "step": step_index,
                "transition": transition_name,
                "min_marking": min_marking,
                "max_marking": max_marking,
            }
        return None
