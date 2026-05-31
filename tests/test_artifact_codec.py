# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Artifact Codec Tests
"""Edge case tests for the u64 compact codec and artifact validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_control.scpn.artifact import (
    ArtifactValidationError,
    _decode_u64_compact,
    compute_artifact_payload_sha256,
    decode_u64_compact,
    encode_u64_compact,
    load_artifact,
    save_artifact,
)
from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.structure import StochasticPetriNet
from scpn_control.scpn.z3_model_checking import (
    Z3FormalVerificationReport,
    Z3ModelCheckingReport,
    build_z3_formal_report_payload,
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
