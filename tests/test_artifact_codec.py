# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Artifact codec and proof-admission tests.
"""Edge case tests for the u64 compact codec and artifact validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scpn_control.scpn.artifact import (
    ActionReadout,
    Artifact,
    ArtifactMeta,
    ArtifactValidationError,
    CompilerInfo,
    FixedPoint,
    FormalVerificationEvidence,
    InitialState,
    PackedWeights,
    PackedWeightsGroup,
    PlaceInjection,
    PlaceSpec,
    Readout,
    SeedPolicy,
    Topology,
    TransitionSpec,
    WeightMatrix,
    Weights,
    _decode_u64_compact,
    compute_artifact_payload_sha256,
    decode_u64_compact,
    encode_u64_compact,
    get_artifact_json_schema,
    load_artifact,
    save_artifact,
    validate_safety_critical_artifact,
)
from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.structure import StochasticPetriNet
from scpn_control.scpn.z3_model_checking import (
    Z3FormalVerificationReport,
    Z3ModelCheckingReport,
    build_z3_formal_report_payload,
)
from scpn_control.scpn.lean_verification import (
    LeanFormalVerificationReport,
    build_lean_formal_report_payload,
    compute_assumption_sha256,
)


def _build_artifact_file(tmp_path: Path) -> Path:
    net = StochasticPetriNet()
    for name in ("P0", "P1", "P2", "P3"):
        net.add_place(name)
    net.add_transition("T0", threshold=0.5)
    net.add_transition("T1", threshold=0.5)
    net.add_arc("P0", "T0", 0.8)
    net.add_arc("P1", "T0", 0.6)
    net.add_arc("T0", "P2", 0.7)
    net.add_arc("P2", "T1", 0.5)
    net.add_arc("T1", "P3", 0.9)

    compiled = FusionCompiler(bitstream_length=64, seed=0).compile(net)
    artifact = compiled.export_artifact(
        name="artifact-validation-contract",
        readout_config={
            "actions": [{"name": "act0", "pos_place": 2, "neg_place": 3}],
            "gains": [1.0],
            "abs_max": [10.0],
            "slew_per_s": [100.0],
        },
        injection_config=[
            {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 1, "source": "x_R_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        ],
    )
    path = tmp_path / "valid.scpnctl.json"
    save_artifact(artifact, str(path))
    return path


def _write_mutated_artifact(source: Path, output: Path, mutations: dict[str, object]) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    for dotted_path, value in mutations.items():
        target = payload
        keys = dotted_path.split(".")
        for key in keys[:-1]:
            target = target[int(key)] if key.isdigit() else target[key]
        last = keys[-1]
        if last.isdigit():
            target[int(last)] = value
        else:
            target[last] = value
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def _write_mutated_artifact_raw(source: Path, output: Path, mutator) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    mutator(payload)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def _passing_formal_evidence(artifact_path: Path, report_path: Path | None = None) -> dict[str, object]:
    artifact = load_artifact(str(artifact_path))
    report_sha256 = "a" * 64
    solver = "z3-solver 4.16.0"
    max_depth = 8
    checked_specs = ["marking_bounds", "move_eventually_fires"]
    if report_path is not None:
        import hashlib

        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
        solver = payload["solver"]
        max_depth = payload["max_depth"]
        checked_specs = payload["checked_specs"]
    return {
        "required": True,
        "status": "pass",
        "backend": "z3",
        "solver": solver,
        "max_depth": max_depth,
        "checked_specs": checked_specs,
        "artifact_sha256": compute_artifact_payload_sha256(artifact),
        "report_sha256": report_sha256,
        "claim_boundary": f"bounded SMT proof through depth {max_depth} over compiled transition relation",
        "report_uri": "validation/reports/scpn_z3_formal.json",
        "generated_utc": "2026-05-31T00:00:00Z",
    }


def _passing_lean_evidence(artifact_path: Path, report_path: Path | None = None) -> dict[str, object]:
    artifact = load_artifact(str(artifact_path))
    report_sha256 = "c" * 64
    if report_path is not None:
        import hashlib

        report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
    proof_assumptions = [
        "bounded actuator command interval from exported artifact readout limits",
        "bounded SNN marking interval [0, 1] from compiled artifact topology",
    ]
    return {
        "required": True,
        "status": "pass",
        "backend": "lean4",
        "solver": "Lean 4.13.0",
        "max_depth": 0,
        "checked_specs": ["pid.actuator_saturation", "snn.marking_bounds"],
        "artifact_sha256": compute_artifact_payload_sha256(artifact),
        "report_sha256": report_sha256,
        "claim_boundary": "bounded Lean proof over compiled controller envelope and exported artifact hash",
        "report_uri": "validation/reports/scpn_lean4_formal.json",
        "generated_utc": "2026-06-02T00:00:00Z",
        "lean_version": "4.13.0",
        "lakefile_sha256": "d" * 64,
        "proof_source_sha256": "e" * 64,
        "theorem_names": [
            "ScpnControl.PID.actuatorSaturationPreserved",
            "ScpnControl.SNN.markingBoundsPreserved",
        ],
        "theorem_modules": ["ScpnControl.PID", "ScpnControl.SNN"],
        "proved_contracts": ["pid.actuator_saturation", "snn.marking_bounds"],
        "module_paths": [
            "src/scpn_control/control/pid_controller.py",
            "src/scpn_control/scpn/controller.py",
        ],
        "safety_case_ids": ["SC-PID-ACTUATOR-SATURATION", "SC-SNN-MARKING-BOUNDS"],
        "proof_assumptions": proof_assumptions,
        "assumption_sha256": compute_assumption_sha256(proof_assumptions),
    }


def _write_valid_z3_report(path: Path) -> None:
    payload = build_z3_formal_report_payload(
        Z3FormalVerificationReport(
            holds=True,
            backend="z3",
            max_depth=8,
            safety=Z3ModelCheckingReport(True, "z3", 8, "unsat"),
            temporal=Z3ModelCheckingReport(
                True,
                "z3",
                8,
                "unsat",
                checked_specs=["move_eventually_fires"],
            ),
        )
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reseal_z3_report_payload(payload: dict[str, object]) -> dict[str, object]:
    canonical = dict(payload)
    canonical.pop("payload_sha256", None)
    blob = json.dumps(canonical, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(blob).hexdigest()
    return payload


def _write_valid_lean_report(path: Path, evidence: dict[str, object]) -> None:
    payload = build_lean_formal_report_payload(
        LeanFormalVerificationReport(
            status=str(evidence["status"]),
            solver=str(evidence["solver"]),
            lean_version=str(evidence["lean_version"]),
            checked_specs=list(evidence["checked_specs"]),
            artifact_sha256=str(evidence["artifact_sha256"]),
            proof_source_sha256=str(evidence["proof_source_sha256"]),
            lakefile_sha256=str(evidence["lakefile_sha256"]),
            theorem_names=list(evidence["theorem_names"]),
            theorem_modules=list(evidence["theorem_modules"]),
            proved_contracts=list(evidence["proved_contracts"]),
            module_paths=list(evidence["module_paths"]),
            safety_case_ids=list(evidence["safety_case_ids"]),
            claim_boundary=str(evidence["claim_boundary"]),
            proof_assumptions=list(evidence["proof_assumptions"]),
        )
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manual_artifact() -> Artifact:
    return Artifact(
        meta=ArtifactMeta(
            artifact_version="1.0.0",
            name="manual-safety-artifact",
            dt_control_s=0.001,
            stream_length=64,
            fixed_point=FixedPoint(data_width=16, fraction_bits=10, signed=False),
            firing_mode="binary",
            seed_policy=SeedPolicy(id="default", hash_fn="splitmix64", rng_family="xoshiro256++"),
            created_utc="2026-05-31T00:00:00Z",
            compiler=CompilerInfo(name="FusionCompiler", version="1.0.0", git_sha="abc1234"),
        ),
        topology=Topology(
            places=[PlaceSpec(id=0, name="P0"), PlaceSpec(id=1, name="P1")],
            transitions=[TransitionSpec(id=0, name="T0", threshold=0.5, delay_ticks=1)],
        ),
        weights=Weights(
            w_in=WeightMatrix(shape=[1, 2], data=[1.0, 0.0]),
            w_out=WeightMatrix(shape=[2, 1], data=[0.0, 1.0]),
            packed=PackedWeightsGroup(
                words_per_stream=1,
                w_in_packed=PackedWeights(shape=[1, 2, 1], data_u64=[2**64 - 1, 0]),
                w_out_packed=PackedWeights(shape=[2, 1, 1], data_u64=[0, 2**64 - 1]),
            ),
        ),
        readout=Readout(
            actions=[ActionReadout(id=0, name="act0", pos_place=1, neg_place=0)],
            gains=[1.0],
            abs_max=[10.0],
            slew_per_s=[100.0],
        ),
        initial_state=InitialState(
            marking=[1.0, 0.0],
            place_injections=[PlaceInjection(place_id=0, source="x_R_pos", scale=1.0, offset=0.0, clamp_0_1=True)],
        ),
    )


def _passing_manual_evidence(artifact: Artifact) -> FormalVerificationEvidence:
    return FormalVerificationEvidence(
        required=True,
        status="pass",
        backend="explicit-state",
        solver="repository-explicit-state",
        max_depth=4,
        checked_specs=["marking_bounds"],
        artifact_sha256=compute_artifact_payload_sha256(artifact),
        report_sha256="b" * 64,
        claim_boundary="bounded explicit-state proof through depth 4",
        report_uri="validation/reports/manual.json",
        generated_utc="2026-05-31T00:00:00Z",
    )


class TestCompactCodecRoundTrip:
    def test_empty_list(self):
        encoded = encode_u64_compact([])
        decoded = decode_u64_compact(encoded)
        assert decoded == []

    def test_single_value(self):
        encoded = encode_u64_compact([42])
        decoded = decode_u64_compact(encoded)
        assert decoded == [42]

    def test_large_values(self):
        data = [0, 1, 2**63 - 1, 2**64 - 1]
        encoded = encode_u64_compact(data)
        decoded = decode_u64_compact(encoded)
        assert decoded == data

    def test_many_values(self):
        data = list(range(500))
        encoded = encode_u64_compact(data)
        decoded = decode_u64_compact(encoded)
        assert decoded == data

    def test_encoding_format(self):
        encoded = encode_u64_compact([1, 2, 3])
        assert encoded["encoding"] == "u64-le-zlib-base64"
        assert encoded["count"] == 3
        assert "data_u64_b64_zlib" in encoded


class TestDecodeErrors:
    def test_wrong_encoding_tag(self):
        with pytest.raises(ArtifactValidationError, match="Unsupported packed encoding"):
            _decode_u64_compact({"encoding": "raw", "data_u64_b64_zlib": "AA=="})

    def test_missing_payload(self):
        with pytest.raises(ArtifactValidationError, match="Missing compact packed payload"):
            _decode_u64_compact({"encoding": "u64-le-zlib-base64", "data_u64_b64_zlib": 123})

    def test_invalid_base64(self):
        with pytest.raises(ArtifactValidationError, match="Invalid base64"):
            _decode_u64_compact(
                {
                    "encoding": "u64-le-zlib-base64",
                    "data_u64_b64_zlib": "!!!not_base64!!!",
                }
            )

    def test_invalid_zlib_payload(self):
        import base64

        bad_bytes = base64.b64encode(b"not_valid_zlib").decode("ascii")
        with pytest.raises(ArtifactValidationError, match="Invalid compact"):
            _decode_u64_compact(
                {
                    "encoding": "u64-le-zlib-base64",
                    "data_u64_b64_zlib": bad_bytes,
                }
            )

    def test_byte_length_not_multiple_of_8(self):
        import base64
        import zlib

        raw = b"\x00" * 7  # 7 bytes, not divisible by 8
        compressed = zlib.compress(raw)
        payload = base64.b64encode(compressed).decode("ascii")
        with pytest.raises(ArtifactValidationError, match="not divisible by 8"):
            _decode_u64_compact(
                {
                    "encoding": "u64-le-zlib-base64",
                    "data_u64_b64_zlib": payload,
                    "count": None,
                }
            )

    def test_negative_count(self):
        encoded = encode_u64_compact([1, 2])
        encoded["count"] = -1
        with pytest.raises(ArtifactValidationError, match="exceeds limit"):
            _decode_u64_compact(encoded)

    def test_count_exceeds_available(self):
        encoded = encode_u64_compact([1])
        encoded["count"] = 99
        with pytest.raises(ArtifactValidationError, match="Invalid compact packed count"):
            _decode_u64_compact(encoded)

    def test_invalid_count_type(self):
        import base64
        import zlib

        raw = b"\x00" * 8
        compressed = zlib.compress(raw)
        payload = base64.b64encode(compressed).decode("ascii")
        with pytest.raises(ArtifactValidationError, match="Invalid compact packed count type"):
            _decode_u64_compact(
                {
                    "encoding": "u64-le-zlib-base64",
                    "data_u64_b64_zlib": payload,
                    "count": "bad",
                }
            )

    def test_none_count_uses_available(self):
        encoded = encode_u64_compact([10, 20, 30])
        del encoded["count"]  # force None path
        decoded = _decode_u64_compact(encoded)
        assert decoded == [10, 20, 30]


class TestManualArtifactContracts:
    def test_payload_hash_excludes_proof_envelope_but_tracks_payload(self) -> None:
        artifact = _manual_artifact()
        before = compute_artifact_payload_sha256(artifact)
        artifact.formal_verification = _passing_manual_evidence(artifact)

        assert compute_artifact_payload_sha256(artifact) == before

        artifact.meta.stream_length = 128
        assert compute_artifact_payload_sha256(artifact) != before

    def test_safety_validation_accepts_explicit_state_without_report_root(self) -> None:
        artifact = _manual_artifact()
        artifact.formal_verification = _passing_manual_evidence(artifact)

        validate_safety_critical_artifact(artifact)

    @pytest.mark.parametrize(
        "report_uri",
        [
            "",
            "/absolute/report.json",
            "file:report.json",
            "https://example.invalid/report.json",
            "~/.cache/report.json",
            "validation\\report.json",
        ],
    )
    def test_formal_report_uri_rejects_unsafe_paths(self, report_uri: str) -> None:
        artifact = _manual_artifact()
        evidence = _passing_manual_evidence(artifact)
        evidence.report_uri = report_uri
        artifact.formal_verification = evidence

        with pytest.raises(ArtifactValidationError, match="report_uri"):
            validate_safety_critical_artifact(artifact)

    def test_save_and_load_roundtrip_preserves_compact_packed_output(self, tmp_path: Path) -> None:
        artifact = _manual_artifact()
        out = tmp_path / "manual.scpnctl.json"

        save_artifact(artifact, out, compact_packed=True)
        loaded = load_artifact(out)

        assert loaded.weights.packed is not None
        assert loaded.weights.packed.w_in_packed.data_u64 == [2**64 - 1, 0]
        assert loaded.weights.packed.w_out_packed is not None
        assert loaded.weights.packed.w_out_packed.data_u64 == [0, 2**64 - 1]


class TestArtifactValidationContract:
    @pytest.fixture()
    def artifact_path(self, tmp_path: Path) -> Path:
        return _build_artifact_file(tmp_path)

    @pytest.mark.parametrize(
        ("mutation", "match"),
        [
            ({"meta.firing_mode": "invalid"}, "firing_mode"),
            ({"meta.fixed_point.data_width": 0}, "data_width"),
            ({"meta.fixed_point.fraction_bits": -1}, "fraction_bits"),
            ({"meta.stream_length": 0}, "stream_length"),
            ({"meta.dt_control_s": 0.0}, "dt_control_s"),
            ({"weights.w_in.data.0": 1.5}, "w_in"),
            ({"weights.w_out.data.0": -0.1}, "w_out"),
            ({"topology.transitions.0.threshold": 1.5}, "threshold"),
            ({"topology.transitions.0.delay_ticks": -1}, "delay_ticks"),
            ({"weights.w_in.data": [0.5]}, "w_in data length"),
            ({"weights.w_out.data": [0.5]}, "w_out data length"),
            ({"initial_state.marking": [0.5]}, "marking"),
            ({"readout.gains.per_action": [1.0, 2.0, 3.0]}, "gains"),
            ({"readout.limits.per_action_abs_max": []}, "abs_max"),
            ({"readout.limits.slew_per_s": [1.0, 2.0]}, "slew_per_s"),
        ],
    )
    def test_artifact_schema_rejects_invalid_physics_and_control_fields(
        self,
        artifact_path: Path,
        tmp_path: Path,
        mutation: dict[str, object],
        match: str,
    ) -> None:
        bad_path = _write_mutated_artifact(artifact_path, tmp_path / "invalid.scpnctl.json", mutation)

        with pytest.raises(ArtifactValidationError, match=match):
            load_artifact(str(bad_path))

    def test_artifact_rejects_malformed_formal_verification_section(self, artifact_path: Path, tmp_path: Path) -> None:
        bad_path = _write_mutated_artifact_raw(
            artifact_path,
            tmp_path / "malformed-formal-section.scpnctl.json",
            lambda payload: payload.__setitem__("formal_verification", []),
        )

        with pytest.raises(ArtifactValidationError, match="formal_verification"):
            load_artifact(str(bad_path))

    def test_artifact_json_schema_declares_safety_critical_formal_evidence_contract(self) -> None:
        schema = get_artifact_json_schema()
        formal = schema["properties"]["formal_verification"]

        assert "formal_verification" in schema["properties"]
        assert formal["required"] == [
            "required",
            "status",
            "backend",
            "solver",
            "max_depth",
            "checked_specs",
            "artifact_sha256",
            "report_sha256",
            "claim_boundary",
        ]
        assert formal["properties"]["backend"]["enum"] == ["explicit-state", "lean4", "z3"]
        assert formal["properties"]["status"]["enum"] == ["pass", "fail", "blocked"]
        assert formal["properties"]["artifact_sha256"]["pattern"] == "^[0-9a-fA-F]{64}$"
        assert formal["additionalProperties"] is False
        assert formal["properties"]["lean_version"]["type"] == "string"
        assert formal["properties"]["proof_source_sha256"]["pattern"] == "^[0-9a-fA-F]{64}$"

    def test_artifact_rejects_out_of_range_initial_marking(self, artifact_path: Path, tmp_path: Path) -> None:
        bad_path = _write_mutated_artifact_raw(
            artifact_path,
            tmp_path / "invalid-marking.scpnctl.json",
            lambda payload: payload["initial_state"]["marking"].__setitem__(0, 1.5),
        )

        with pytest.raises(ArtifactValidationError, match="marking"):
            load_artifact(str(bad_path))

    def test_artifact_rejects_bool_place_injection_id(self, artifact_path: Path, tmp_path: Path) -> None:
        bad_path = _write_mutated_artifact_raw(
            artifact_path,
            tmp_path / "invalid-injection-place.scpnctl.json",
            lambda payload: payload["initial_state"]["place_injections"][0].__setitem__("place_id", True),
        )

        with pytest.raises(ArtifactValidationError, match="place_id"):
            load_artifact(str(bad_path))

    def test_artifact_rejects_empty_injection_source(self, artifact_path: Path, tmp_path: Path) -> None:
        bad_path = _write_mutated_artifact_raw(
            artifact_path,
            tmp_path / "invalid-injection-source.scpnctl.json",
            lambda payload: payload["initial_state"]["place_injections"][0].__setitem__("source", ""),
        )

        with pytest.raises(ArtifactValidationError, match="source"):
            load_artifact(str(bad_path))

    def test_artifact_rejects_bool_readout_place(self, artifact_path: Path, tmp_path: Path) -> None:
        bad_path = _write_mutated_artifact_raw(
            artifact_path,
            tmp_path / "invalid-readout-place.scpnctl.json",
            lambda payload: payload["readout"]["actions"][0].__setitem__("pos_place", True),
        )

        with pytest.raises(ArtifactValidationError, match="pos_place"):
            load_artifact(str(bad_path))

    def test_artifact_rejects_readout_place_outside_topology(self, artifact_path: Path, tmp_path: Path) -> None:
        bad_path = _write_mutated_artifact_raw(
            artifact_path,
            tmp_path / "invalid-readout-index.scpnctl.json",
            lambda payload: payload["readout"]["actions"][0].__setitem__("neg_place", 999),
        )

        with pytest.raises(ArtifactValidationError, match="neg_place"):
            load_artifact(str(bad_path))

    def test_safety_critical_artifact_requires_formal_proof_manifest(self, artifact_path: Path) -> None:
        with pytest.raises(ArtifactValidationError, match="formal_verification"):
            load_artifact(str(artifact_path), require_formal_verification=True)

    def test_formal_proof_manifest_rejects_unknown_fields(self, artifact_path: Path, tmp_path: Path) -> None:
        def mutate(payload: dict[str, object]) -> None:
            evidence = _passing_formal_evidence(artifact_path)
            evidence["certification_status"] = "certified"
            payload["formal_verification"] = evidence

        bad_path = _write_mutated_artifact_raw(artifact_path, tmp_path / "unknown-formal-field.scpnctl.json", mutate)

        with pytest.raises(ArtifactValidationError, match="unsupported fields"):
            load_artifact(str(bad_path), require_formal_verification=True)

    def test_safety_critical_artifact_accepts_passing_z3_formal_proof(
        self,
        artifact_path: Path,
        tmp_path: Path,
    ) -> None:
        proven_path = _write_mutated_artifact_raw(
            artifact_path,
            tmp_path / "proven.scpnctl.json",
            lambda payload: payload.__setitem__("formal_verification", _passing_formal_evidence(artifact_path)),
        )

        artifact = load_artifact(str(proven_path), require_formal_verification=True)

        assert artifact.formal_verification is not None
        assert artifact.formal_verification.status == "pass"
        assert artifact.formal_verification.backend == "z3"
        assert artifact.formal_verification.max_depth == 8
        assert artifact.formal_verification.report_sha256 == "a" * 64

    def test_safety_critical_artifact_accepts_passing_lean4_formal_proof(
        self,
        artifact_path: Path,
        tmp_path: Path,
    ) -> None:
        report_root = tmp_path / "reports"
        report_path = report_root / "validation" / "reports" / "scpn_lean4_formal.json"
        report_path.parent.mkdir(parents=True)
        evidence = _passing_lean_evidence(artifact_path)
        _write_valid_lean_report(report_path, evidence)
        evidence = _passing_lean_evidence(artifact_path, report_path)
        proven_path = _write_mutated_artifact_raw(
            artifact_path,
            tmp_path / "proven-lean.scpnctl.json",
            lambda payload: payload.__setitem__("formal_verification", evidence),
        )

        artifact = load_artifact(
            str(proven_path),
            require_formal_verification=True,
            formal_report_root=report_root,
        )

        assert artifact.formal_verification is not None
        assert artifact.formal_verification.backend == "lean4"
        assert artifact.formal_verification.theorem_names == [
            "ScpnControl.PID.actuatorSaturationPreserved",
            "ScpnControl.SNN.markingBoundsPreserved",
        ]

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("lean_version", "", "lean_version"),
            ("solver", "Coq 8.19.0", "solver must identify Lean"),
            ("solver", "Lean 4.12.0", "solver must include lean_version"),
            ("lakefile_sha256", "not-a-sha", "lakefile_sha256"),
            ("proof_source_sha256", "not-a-sha", "proof_source_sha256"),
            ("theorem_names", [], "theorem_names"),
            ("theorem_names", ["bad theorem"], "theorem_names"),
            ("theorem_modules", ["../Escape"], "theorem_modules"),
            ("proved_contracts", ["pid.actuator_saturation"], "proved_contracts"),
            (
                "proved_contracts",
                ["pid.actuator_saturation", "snn.marking_bounds", "facility.full_certification"],
                "proved_contracts contains unsupported lean4 contracts",
            ),
            ("module_paths", ["../src/scpn_control/scpn/controller.py"], "module_paths"),
            ("safety_case_ids", ["bad id"], "safety_case_ids"),
            ("proof_assumptions", ["unbounded plant dynamics"], "proof_assumptions"),
            ("assumption_sha256", "not-a-sha", "assumption_sha256"),
            ("assumption_sha256", "0" * 64, "assumption_sha256"),
            (
                "theorem_names",
                ["ScpnControl.Transport.unrelated", "ScpnControl.SNN.markingBoundsPreserved"],
                "pid.actuator_saturation requires theorem_names",
            ),
            (
                "theorem_modules",
                ["ScpnControl.Transport", "ScpnControl.SNN"],
                "pid.actuator_saturation requires theorem_modules",
            ),
            (
                "theorem_names",
                [
                    "ScpnControl.PID.actuatorSaturationPreserved",
                    "ScpnControl.SNN.markingBoundsPreserved",
                    "ScpnControl.Transport.unrelatedProof",
                ],
                "theorem_names contains unsupported namespaces",
            ),
            (
                "module_paths",
                [
                    "src/scpn_control/control/pid_controller.py",
                    "src/scpn_control/scpn/geometry_neutral_replay.py",
                ],
                "module_paths missing required paths",
            ),
            (
                "safety_case_ids",
                ["SC-PID-ACTUATOR-SATURATION", "SC-UNRELATED-FORMAL-EVIDENCE"],
                "safety_case_ids missing required IDs",
            ),
        ],
    )
    def test_lean4_formal_proof_requires_machine_checkable_contract_metadata(
        self,
        artifact_path: Path,
        tmp_path: Path,
        field: str,
        value: object,
        match: str,
    ) -> None:
        def mutate(payload: dict[str, object]) -> None:
            evidence = _passing_lean_evidence(artifact_path)
            evidence[field] = value
            payload["formal_verification"] = evidence

        bad_path = _write_mutated_artifact_raw(artifact_path, tmp_path / f"bad-lean-{field}.scpnctl.json", mutate)

        with pytest.raises(ArtifactValidationError, match=match):
            load_artifact(str(bad_path), require_formal_verification=True)

    def test_lean4_formal_report_must_match_manifest_theorems(
        self,
        artifact_path: Path,
        tmp_path: Path,
    ) -> None:
        report_root = tmp_path / "reports"
        report_path = report_root / "validation" / "reports" / "scpn_lean4_formal.json"
        report_path.parent.mkdir(parents=True)
        report_evidence = _passing_lean_evidence(artifact_path)
        report_evidence["theorem_names"] = [
            "ScpnControl.PID.onlyPartial",
            "ScpnControl.SNN.markingBoundsPreserved",
        ]
        _write_valid_lean_report(report_path, report_evidence)
        evidence = _passing_lean_evidence(artifact_path, report_path)
        bad_path = _write_mutated_artifact_raw(
            artifact_path,
            tmp_path / "bad-lean-report.scpnctl.json",
            lambda payload: payload.__setitem__("formal_verification", evidence),
        )

        with pytest.raises(ArtifactValidationError, match="theorem_names"):
            load_artifact(str(bad_path), require_formal_verification=True, formal_report_root=report_root)

    def test_lean4_formal_report_rejects_duplicate_json_keys_under_report_root(
        self,
        artifact_path: Path,
        tmp_path: Path,
    ) -> None:
        report_root = tmp_path / "reports"
        report_path = report_root / "validation" / "reports" / "scpn_lean4_formal.json"
        report_path.parent.mkdir(parents=True)
        evidence = _passing_lean_evidence(artifact_path)
        _write_valid_lean_report(report_path, evidence)
        report_text = report_path.read_text(encoding="utf-8")
        report_path.write_text(
            report_text.replace('"status": "pass",', '"status": "pass",\n  "status": "fail",', 1),
            encoding="utf-8",
        )
        evidence = _passing_lean_evidence(artifact_path, report_path)
        bad_path = _write_mutated_artifact_raw(
            artifact_path,
            tmp_path / "duplicate-key-lean-report.scpnctl.json",
            lambda payload: payload.__setitem__("formal_verification", evidence),
        )

        with pytest.raises(ArtifactValidationError, match="duplicate JSON key: status"):
            load_artifact(str(bad_path), require_formal_verification=True, formal_report_root=report_root)

    def test_safety_critical_artifact_rejects_tampered_artifact_payload(
        self,
        artifact_path: Path,
        tmp_path: Path,
    ) -> None:
        def mutate(payload: dict[str, object]) -> None:
            payload["formal_verification"] = _passing_formal_evidence(artifact_path)
            payload["meta"]["stream_length"] = 128

        tampered_path = _write_mutated_artifact_raw(artifact_path, tmp_path / "tampered.scpnctl.json", mutate)

        with pytest.raises(ArtifactValidationError, match="artifact_sha256"):
            load_artifact(str(tampered_path), require_formal_verification=True)

    def test_safety_critical_artifact_verifies_report_digest_under_root(
        self,
        artifact_path: Path,
        tmp_path: Path,
    ) -> None:
        report_root = tmp_path / "reports"
        report_path = report_root / "validation" / "reports" / "scpn_z3_formal.json"
        report_path.parent.mkdir(parents=True)
        _write_valid_z3_report(report_path)
        proven_path = _write_mutated_artifact_raw(
            artifact_path,
            tmp_path / "proven-report.scpnctl.json",
            lambda payload: payload.__setitem__(
                "formal_verification",
                _passing_formal_evidence(artifact_path, report_path),
            ),
        )

        artifact = load_artifact(
            str(proven_path),
            require_formal_verification=True,
            formal_report_root=report_root,
        )

        assert artifact.formal_verification is not None
        assert artifact.formal_verification.report_sha256 != "a" * 64

    def test_safety_critical_artifact_rejects_report_digest_mismatch(
        self,
        artifact_path: Path,
        tmp_path: Path,
    ) -> None:
        report_root = tmp_path / "reports"
        report_path = report_root / "validation" / "reports" / "scpn_z3_formal.json"
        report_path.parent.mkdir(parents=True)
        _write_valid_z3_report(report_path)
        bad_path = _write_mutated_artifact_raw(
            artifact_path,
            tmp_path / "bad-report.scpnctl.json",
            lambda payload: payload.__setitem__("formal_verification", _passing_formal_evidence(artifact_path)),
        )

        with pytest.raises(ArtifactValidationError, match="report_sha256"):
            load_artifact(str(bad_path), require_formal_verification=True, formal_report_root=report_root)

    def test_safety_critical_artifact_rejects_z3_report_schema_mismatch(
        self,
        artifact_path: Path,
        tmp_path: Path,
    ) -> None:
        report_root = tmp_path / "reports"
        report_path = report_root / "validation" / "reports" / "scpn_z3_formal.json"
        report_path.parent.mkdir(parents=True)
        _write_valid_z3_report(report_path)

        def mutate(payload: dict[str, object]) -> None:
            evidence = _passing_formal_evidence(artifact_path, report_path)
            evidence["checked_specs"] = ["marking_bounds"]
            payload["formal_verification"] = evidence

        bad_path = _write_mutated_artifact_raw(
            artifact_path,
            tmp_path / "bad-z3-report-metadata.scpnctl.json",
            mutate,
        )

        with pytest.raises(ArtifactValidationError, match="checked_specs"):
            load_artifact(str(bad_path), require_formal_verification=True, formal_report_root=report_root)

    def test_z3_formal_report_rejects_duplicate_json_keys_under_report_root(
        self,
        artifact_path: Path,
        tmp_path: Path,
    ) -> None:
        report_root = tmp_path / "reports"
        report_path = report_root / "validation" / "reports" / "scpn_z3_formal.json"
        report_path.parent.mkdir(parents=True)
        _write_valid_z3_report(report_path)
        report_text = report_path.read_text(encoding="utf-8")
        report_path.write_text(
            report_text.replace('"status": "pass",', '"status": "pass",\n  "status": "fail",', 1),
            encoding="utf-8",
        )
        bad_path = _write_mutated_artifact_raw(
            artifact_path,
            tmp_path / "duplicate-key-z3-report.scpnctl.json",
            lambda payload: payload.__setitem__(
                "formal_verification",
                _passing_formal_evidence(artifact_path, report_path),
            ),
        )

        with pytest.raises(ArtifactValidationError, match="duplicate JSON key: status"):
            load_artifact(str(bad_path), require_formal_verification=True, formal_report_root=report_root)

    def test_z3_formal_report_rejects_unknown_fields_under_report_root(
        self,
        artifact_path: Path,
        tmp_path: Path,
    ) -> None:
        report_root = tmp_path / "reports"
        report_path = report_root / "validation" / "reports" / "scpn_z3_formal.json"
        report_path.parent.mkdir(parents=True)
        _write_valid_z3_report(report_path)
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        report_payload["foreign_attestation"] = "unrelated-proof-engine"
        _reseal_z3_report_payload(report_payload)
        report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        bad_path = _write_mutated_artifact_raw(
            artifact_path,
            tmp_path / "unknown-field-z3-report.scpnctl.json",
            lambda payload: payload.__setitem__(
                "formal_verification",
                _passing_formal_evidence(artifact_path, report_path),
            ),
        )

        with pytest.raises(ArtifactValidationError, match="unknown fields"):
            load_artifact(str(bad_path), require_formal_verification=True, formal_report_root=report_root)

    def test_z3_formal_report_rejects_unknown_violation_fields_under_report_root(
        self,
        artifact_path: Path,
        tmp_path: Path,
    ) -> None:
        report_root = tmp_path / "reports"
        report_path = report_root / "validation" / "reports" / "scpn_z3_formal.json"
        report_path.parent.mkdir(parents=True)
        _write_valid_z3_report(report_path)
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        report_payload["safety"]["violations"] = [
            {
                "foreign_counterexample": "padded-proof",
                "marking": {"sink": 1.0},
                "message": "sink exceeds admitted control envelope",
                "path": ["move"],
                "place": "sink",
                "property_name": "unsafe_bound",
                "transition": None,
            }
        ]
        _reseal_z3_report_payload(report_payload)
        report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        bad_path = _write_mutated_artifact_raw(
            artifact_path,
            tmp_path / "unknown-violation-field-z3-report.scpnctl.json",
            lambda payload: payload.__setitem__(
                "formal_verification",
                _passing_formal_evidence(artifact_path, report_path),
            ),
        )

        with pytest.raises(ArtifactValidationError, match="violation.*unknown fields"):
            load_artifact(str(bad_path), require_formal_verification=True, formal_report_root=report_root)

    def test_z3_formal_report_rejects_inconsistent_solver_status_under_report_root(
        self,
        artifact_path: Path,
        tmp_path: Path,
    ) -> None:
        report_root = tmp_path / "reports"
        report_path = report_root / "validation" / "reports" / "scpn_z3_formal.json"
        report_path.parent.mkdir(parents=True)
        _write_valid_z3_report(report_path)
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        report_payload["safety"]["solver_status"] = "sat"
        _reseal_z3_report_payload(report_payload)
        report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        bad_path = _write_mutated_artifact_raw(
            artifact_path,
            tmp_path / "inconsistent-solver-status-z3-report.scpnctl.json",
            lambda payload: payload.__setitem__(
                "formal_verification",
                _passing_formal_evidence(artifact_path, report_path),
            ),
        )

        with pytest.raises(ArtifactValidationError, match="sat.*must not hold"):
            load_artifact(str(bad_path), require_formal_verification=True, formal_report_root=report_root)

    def test_z3_formal_report_rejects_duplicate_section_specs_under_report_root(
        self,
        artifact_path: Path,
        tmp_path: Path,
    ) -> None:
        report_root = tmp_path / "reports"
        report_path = report_root / "validation" / "reports" / "scpn_z3_formal.json"
        report_path.parent.mkdir(parents=True)
        _write_valid_z3_report(report_path)
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        report_payload["temporal"]["checked_specs"] = ["move_eventually_fires", "move_eventually_fires"]
        _reseal_z3_report_payload(report_payload)
        report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        bad_path = _write_mutated_artifact_raw(
            artifact_path,
            tmp_path / "duplicate-section-spec-z3-report.scpnctl.json",
            lambda payload: payload.__setitem__(
                "formal_verification",
                _passing_formal_evidence(artifact_path, report_path),
            ),
        )

        with pytest.raises(ArtifactValidationError, match="checked_specs.*unique"):
            load_artifact(str(bad_path), require_formal_verification=True, formal_report_root=report_root)

    def test_formal_proof_manifest_rejects_unsafe_report_uri(
        self,
        artifact_path: Path,
        tmp_path: Path,
    ) -> None:
        def mutate(payload: dict[str, object]) -> None:
            evidence = _passing_formal_evidence(artifact_path)
            evidence["report_uri"] = "../outside.json"
            payload["formal_verification"] = evidence

        bad_path = _write_mutated_artifact_raw(artifact_path, tmp_path / "traversal-report.scpnctl.json", mutate)

        with pytest.raises(ArtifactValidationError, match="report_uri"):
            load_artifact(str(bad_path), require_formal_verification=True)

    def test_safety_critical_artifact_rejects_blocked_formal_proof(
        self,
        artifact_path: Path,
        tmp_path: Path,
    ) -> None:
        def mutate(payload: dict[str, object]) -> None:
            evidence = _passing_formal_evidence(artifact_path)
            evidence["status"] = "blocked"
            payload["formal_verification"] = evidence

        blocked_path = _write_mutated_artifact_raw(artifact_path, tmp_path / "blocked.scpnctl.json", mutate)

        with pytest.raises(ArtifactValidationError, match="requires passing"):
            load_artifact(str(blocked_path), require_formal_verification=True)

    def test_failed_formal_proof_manifest_must_store_counterexample_path(
        self,
        artifact_path: Path,
        tmp_path: Path,
    ) -> None:
        def mutate(payload: dict[str, object]) -> None:
            evidence = _passing_formal_evidence(artifact_path)
            evidence["status"] = "fail"
            payload["formal_verification"] = evidence

        failed_path = _write_mutated_artifact_raw(artifact_path, tmp_path / "failed-missing-path.scpnctl.json", mutate)

        with pytest.raises(ArtifactValidationError, match="counterexample path"):
            load_artifact(str(failed_path))

    def test_failed_formal_proof_manifest_preserves_counterexample_path(
        self,
        artifact_path: Path,
        tmp_path: Path,
    ) -> None:
        def mutate(payload: dict[str, object]) -> None:
            evidence = _passing_formal_evidence(artifact_path)
            evidence["status"] = "fail"
            evidence["counterexample_path"] = ["T0", "T1"]
            evidence["counterexample_property"] = "always_bounded_marking"
            payload["formal_verification"] = evidence

        failed_path = _write_mutated_artifact_raw(artifact_path, tmp_path / "failed-with-path.scpnctl.json", mutate)
        artifact = load_artifact(str(failed_path))

        assert artifact.formal_verification is not None
        assert artifact.formal_verification.status == "fail"
        assert artifact.formal_verification.counterexample_path == ["T0", "T1"]
        assert artifact.formal_verification.counterexample_property == "always_bounded_marking"

    def test_formal_proof_manifest_rejects_unbounded_claim_boundary(
        self,
        artifact_path: Path,
        tmp_path: Path,
    ) -> None:
        def mutate(payload: dict[str, object]) -> None:
            evidence = _passing_formal_evidence(artifact_path)
            evidence["claim_boundary"] = "unbounded controller safety proof"
            payload["formal_verification"] = evidence

        bad_path = _write_mutated_artifact_raw(artifact_path, tmp_path / "unbounded-claim.scpnctl.json", mutate)

        with pytest.raises(ArtifactValidationError, match="bounded proof boundary"):
            load_artifact(str(bad_path), require_formal_verification=True)

    def test_artifact_serialises_notes_and_noncompact_packed_weights(
        self,
        artifact_path: Path,
        tmp_path: Path,
    ) -> None:
        artifact = load_artifact(str(artifact_path))
        if artifact.weights.packed is None:
            pytest.skip("No packed weights in test artifact")
        artifact.meta.notes = "validation memo"
        out = tmp_path / "noted.scpnctl.json"

        save_artifact(artifact, str(out), compact_packed=False)
        payload = json.loads(out.read_text(encoding="utf-8"))

        assert payload["meta"]["notes"] == "validation memo"
        assert "data_u64" in payload["weights"]["packed"]["w_in_packed"]
