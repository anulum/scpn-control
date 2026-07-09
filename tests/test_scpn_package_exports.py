# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — SCPN Package Public Export Wrapper Tests
"""Exercise the lazy-import geometry-neutral replay wrappers re-exported from the
``scpn_control.scpn`` package root.

Each public wrapper in ``scpn_control/scpn/__init__.py`` defers to its
implementation in ``geometry_neutral_replay`` via a function-local import. Calling
them through the package root (rather than the implementation module directly) is
what drives those wrapper bodies, so these tests import exclusively from
``scpn_control.scpn``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_control.scpn import (
    GEOMETRY_NEUTRAL_REPLAY_EVIDENCE_SCHEMA_VERSION,
    GEOMETRY_NEUTRAL_REPLAY_SCHEMA_VERSION_V1_1,
    AERControlObservation,
    CertificateReplayResult,
    ControllerRuntimeBinding,
    FeatureAxisSpec,
    RuntimeTarget,
    SpikeBuffer,
    SpikeEvent,
    TimingEnvelope,
    assert_geometry_neutral_replay_claim_admissible,
    assert_geometry_neutral_v1_replay_loadable_under_v1_1_schema_bundle,
    assert_runtime_certificate_admissible,
    attach_geometry_neutral_aer_admission_metadata,
    build_geometry_neutral_aer_admission_metadata,
    compute_petri_topology_digest,
    decode_action_vector,
    feature_error_components,
    generate_geometry_neutral_report,
    geometry_neutral_replay_evidence,
    issue_runtime_safety_certificate,
    load_geometry_neutral_replay_evidence,
    load_geometry_neutral_replay_report,
    load_geometry_neutral_replay_schema,
    register_geometry_neutral_replay_v1_1_schema,
    render_geometry_neutral_markdown,
    replay_runtime_safety_certificate,
    save_geometry_neutral_replay_evidence,
    save_geometry_neutral_replay_report,
    validate_runtime_safety_certificate_payload,
    validate_geometry_neutral_report,
)


def _report() -> dict[str, Any]:
    return generate_geometry_neutral_report(steps=8, seed=41)


def _aer_admission_metadata() -> dict[str, Any]:
    buffer = SpikeBuffer(capacity=8)
    buffer.extend(
        [
            SpikeEvent(neuron_id=0, timestamp_ns=10),
            SpikeEvent(neuron_id=1, timestamp_ns=30),
            SpikeEvent(neuron_id=1, timestamp_ns=50),
        ]
    )
    observation = AERControlObservation(
        timestamp_ns=100,
        spike_stream=buffer,
        decode_window_ns=100,
        decode_strategy="rate",
        n_features=4,
        require_monotonic=True,
    )
    return build_geometry_neutral_aer_admission_metadata(
        admission_report=observation.admission_report(),
        decode_strategy="rate",
        decode_window_ns=100,
        n_features=4,
        feature_normalisation="unit",
        require_monotonic=True,
        feature_vector=observation.to_features(),
    )


class TestReportWrappers:
    def test_generate_then_validate_round_trips(self) -> None:
        report = _report()
        bench = report["geometry_neutral_replay"]
        assert bench["schema_version"].endswith(".v1")
        # validate raises on any violation; a freshly generated report must pass.
        validate_geometry_neutral_report(report)

    def test_generate_is_deterministic_for_same_seed(self) -> None:
        assert _report() == generate_geometry_neutral_report(steps=8, seed=41)

    def test_save_then_load_is_identity(self, tmp_path: Path) -> None:
        report = _report()
        path = save_geometry_neutral_replay_report(report, tmp_path / "replay.json")
        assert load_geometry_neutral_replay_report(path) == report

    def test_render_markdown_contains_replay_heading(self) -> None:
        markdown = render_geometry_neutral_markdown(_report())
        assert isinstance(markdown, str)
        assert "geometry" in markdown.lower()


class TestSchemaWrappers:
    def test_register_v1_1_matches_loaded_schema(self) -> None:
        registered = register_geometry_neutral_replay_v1_1_schema()
        assert registered["$id"] == GEOMETRY_NEUTRAL_REPLAY_SCHEMA_VERSION_V1_1
        assert load_geometry_neutral_replay_schema(GEOMETRY_NEUTRAL_REPLAY_SCHEMA_VERSION_V1_1) == registered

    def test_v1_report_loads_under_v1_1_schema_bundle(self) -> None:
        register_geometry_neutral_replay_v1_1_schema()
        loaded = assert_geometry_neutral_v1_replay_loadable_under_v1_1_schema_bundle(_report())
        assert isinstance(loaded, dict)


class TestAerAdmissionWrappers:
    def test_build_metadata_has_aer_admission_schema(self) -> None:
        metadata = _aer_admission_metadata()
        assert metadata["schema_version"].startswith("scpn-control.geometry-neutral-replay-aer-admission")
        assert metadata["monotonic_input"] is True

    def test_attach_metadata_binds_digest_into_manifest(self, tmp_path: Path) -> None:
        attached: Any = attach_geometry_neutral_aer_admission_metadata(_report(), _aer_admission_metadata())
        bench = attached["geometry_neutral_replay"]
        assert bench["aer_admission"]["decode_window_ns"] == 100
        assert bench["manifest"]["aer_admission_digest"]
        # the attached report must still survive a save/load round trip.
        path = save_geometry_neutral_replay_report(attached, tmp_path / "aer_replay.json")
        assert load_geometry_neutral_replay_report(path) == attached


class TestContractAndRuntimeSafetyExports:
    def test_feature_axis_and_kernel_exports_are_available(self) -> None:
        axis = FeatureAxisSpec(obs_key="beta_n", target=2.0, scale=0.5, pos_key="beta_pos", neg_key="beta_neg")
        pos, neg = feature_error_components([1.75], [axis.target], [axis.scale], axis_names=[axis.obs_key])
        previous = np.zeros(1, dtype=np.float64)

        decoded = decode_action_vector([0.9, 0.2], [0], [1], [10.0], [5.0], [100.0], 0.01, previous)

        assert float(pos[0]) == pytest.approx(0.5)
        assert float(neg[0]) == pytest.approx(0.0)
        assert decoded is previous
        assert float(decoded[0]) == pytest.approx(1.0)

    def test_runtime_safety_types_and_helpers_are_available(self) -> None:
        target = RuntimeTarget(name="pytest-target", architecture="x86_64", runtime="numpy", toolchain="pytest")
        envelope = TimingEnvelope(
            control_period_us=1000.0,
            worst_case_response_us=100.0,
            deadline_us=500.0,
            proof_firing_depth=4,
        )
        binding = ControllerRuntimeBinding(
            controller_id="export-test",
            controller_config={"runtime_backend": "numpy"},
            petri_topology_sha256="a" * 64,
            snn_parameters={"bitstream_length": 64},
            solver_mode="scpn-numpy",
            runtime_target=target,
            timing_envelope=envelope,
        )
        replay = CertificateReplayResult(True, True, True, True)

        assert target.digest()
        assert binding.digest()
        assert replay.passed is True
        assert callable(compute_petri_topology_digest)
        assert callable(issue_runtime_safety_certificate)
        assert callable(validate_runtime_safety_certificate_payload)
        assert callable(replay_runtime_safety_certificate)
        assert callable(assert_runtime_certificate_admissible)


class TestEvidenceWrappers:
    def test_evidence_round_trips_and_rejects_bounded_claim(self, tmp_path: Path) -> None:
        report = _report()
        evidence: Any = geometry_neutral_replay_evidence(report, generated_utc="2026-05-31T00:00:00Z")
        assert evidence.schema_version == GEOMETRY_NEUTRAL_REPLAY_EVIDENCE_SCHEMA_VERSION
        assert evidence.replay_report_sha256
        # a bounded (non-device) claim must be refused.
        with pytest.raises(ValueError, match="bounded-only"):
            assert_geometry_neutral_replay_claim_admissible(evidence)

        path = tmp_path / "evidence.json"
        save_geometry_neutral_replay_evidence(evidence, path)
        assert load_geometry_neutral_replay_evidence(path) == evidence

    def test_load_evidence_detects_tampering(self, tmp_path: Path) -> None:
        evidence: Any = geometry_neutral_replay_evidence(_report(), generated_utc="2026-05-31T00:00:00Z")
        path = tmp_path / "evidence.json"
        save_geometry_neutral_replay_evidence(evidence, path)
        payload = path.read_text(encoding="utf-8").replace('"deterministic": true', '"deterministic": false')
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(ValueError):
            load_geometry_neutral_replay_evidence(path)


def test_package_dir_includes_lazy_public_exports() -> None:
    """``dir(scpn_control)`` merges the module globals with the lazy ``__all__``.

    The package exposes public names lazily, so ``__dir__`` must union the live
    globals with the declared ``__all__`` to surface every public export to
    interactive tools.
    """
    import scpn_control

    listed = scpn_control.__dir__()
    assert set(scpn_control.__all__).issubset(set(listed))
    assert listed == sorted(set(listed))


def test_package_version_falls_back_when_distribution_metadata_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source-tree package import exposes the local development sentinel."""
    import scpn_control

    def missing_distribution(distribution_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(distribution_name)

    with monkeypatch.context() as patch:
        patch.setattr(importlib.metadata, "version", missing_distribution)
        importlib.reload(scpn_control)
        assert scpn_control.__version__ == "0.0.0.dev"

    importlib.reload(scpn_control)


def test_package_lazy_export_loads_and_caches_public_symbol() -> None:
    """A package-root public export loads through the declared lazy module map."""
    import scpn_control

    scpn_control.__dict__.pop("TokamakConfig", None)
    exported = scpn_control.TokamakConfig
    from scpn_control.core import TokamakConfig

    assert exported is TokamakConfig
    assert scpn_control.__dict__["TokamakConfig"] is TokamakConfig


def test_package_lazy_export_rejects_unknown_symbol() -> None:
    """Unknown package-root attributes fail with the normal AttributeError contract."""
    import scpn_control

    with pytest.raises(AttributeError, match="definitely_missing"):
        scpn_control.__getattr__("definitely_missing")
