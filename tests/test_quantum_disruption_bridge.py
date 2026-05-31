# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Quantum Disruption Bridge Tests

"""Behavioural tests for the optional quantum disruption bridge."""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest


def _control_features() -> np.ndarray:
    return np.array([1.2, 2.4, 3.4, 0.7, 0.9, 0.15, 0.004, 0.2], dtype=np.float64)


def _extra_iter_features() -> dict[str, float]:
    return {
        "P_rad": 22.0,
        "V_loop": 0.6,
        "W_stored": 110.0,
        "kappa": 1.75,
        "dIp_dt": -0.4,
    }


def test_quantum_disruption_bridge_import_does_not_import_quantum_package() -> None:
    sys.modules.pop("scpn_quantum_control", None)
    module = importlib.import_module("scpn_control.control.quantum_disruption_bridge")

    assert module.QUANTUM_BACKEND_OWNER == "scpn-quantum-control"
    assert "scpn_quantum_control" not in sys.modules


def test_quantum_disruption_dependency_contract_advertises_quantum_owner_surface() -> None:
    from scpn_control.control.quantum_disruption_bridge import (
        quantum_disruption_dependency_contract,
        validate_quantum_disruption_dependency_contract,
    )

    contract = quantum_disruption_dependency_contract()
    simulator_dependency = "qiskit-" + "a" + "er>=0.15,<1.0"
    misspelled_dependency = "qiskit-" + "a" + "re>=0.15,<1.0"

    assert contract["control_facade_owner"] == "scpn-control"
    assert contract["quantum_backend_owner"] == "scpn-quantum-control"
    assert contract["quantum_module"] == "scpn_quantum_control.control.q_disruption_iter"
    assert contract["required_public_surface"]["classifier_class"] == "QuantumDisruptionClassifier"
    assert contract["required_public_surface"]["predict_input"]["feature_names"][0] == "I_p"
    assert contract["required_public_surface"]["predict_input"]["normalised_range"] == [0.0, 1.0]
    assert contract["required_public_surface"]["predict_output"]["range"] == [0.0, 1.0]
    assert "qiskit>=2.2,<3.0" in contract["dependency_groups"]["quantum_core"]
    assert simulator_dependency in contract["dependency_groups"]["quantum_core"]
    assert misspelled_dependency not in contract["dependency_groups"]["quantum_core"]
    assert "pennylane>=0.40,<1.0" in contract["dependency_groups"]["quantum_optional_providers"]
    assert "qiskit-ibm-runtime>=0.40,<1.0" in contract["dependency_groups"]["quantum_optional_providers"]
    assert "do_not_admit_control_action" in contract["required_downstream_policy"]
    assert len(contract["contract_sha256"]) == 64
    assert validate_quantum_disruption_dependency_contract(contract) == contract


def test_quantum_disruption_dependency_contract_rejects_dependency_drift() -> None:
    from scpn_control.control.quantum_disruption_bridge import (
        quantum_disruption_dependency_contract,
        validate_quantum_disruption_dependency_contract,
    )

    contract = quantum_disruption_dependency_contract()
    drifted = {
        **contract,
        "dependency_groups": {
            **contract["dependency_groups"],
            "quantum_core": ["qiskit>=2.2,<3.0"],
        },
    }

    with pytest.raises(ValueError, match="quantum_core"):
        validate_quantum_disruption_dependency_contract(drifted)


def test_quantum_disruption_bridge_rejects_missing_iter_features_without_defaults() -> None:
    from scpn_control.control.quantum_disruption_bridge import map_control_features_to_iter

    with pytest.raises(ValueError, match="missing required ITER features"):
        map_control_features_to_iter(_control_features())


def test_quantum_disruption_bridge_records_declared_defaults_as_bounded_model() -> None:
    from scpn_control.control.quantum_disruption_bridge import (
        QuantumDisruptionBridgeConfig,
        map_control_features_to_iter,
    )

    mapped = map_control_features_to_iter(
        _control_features(),
        config=QuantumDisruptionBridgeConfig(allow_center_defaults=True),
    )

    assert mapped.claim_status == "bounded_model"
    assert mapped.publication_safe is False
    assert mapped.defaults_used == ("P_rad", "V_loop", "W_stored", "kappa", "dIp_dt")
    assert mapped.raw_iter_features.shape == (11,)
    assert mapped.normalized_iter_features.shape == (11,)
    assert np.all((mapped.normalized_iter_features >= 0.0) & (mapped.normalized_iter_features <= 1.0))


def test_quantum_disruption_kernel_matrix_is_symmetric_bounded_and_digestible() -> None:
    from scpn_control.control.quantum_disruption_bridge import (
        QuantumDisruptionBridgeConfig,
        quantum_disruption_kernel_matrix,
        validate_quantum_disruption_kernel_report,
    )

    samples = np.vstack([_control_features(), _control_features() * np.array([1.1, 0.9, 1, 1, 1, 1, 0.8, 1])])

    report = quantum_disruption_kernel_matrix(
        samples,
        config=QuantumDisruptionBridgeConfig(allow_center_defaults=True),
    )

    kernel = np.asarray(report["kernel_matrix"], dtype=np.float64)
    assert kernel.shape == (2, 2)
    np.testing.assert_allclose(kernel, kernel.T, atol=1.0e-12)
    np.testing.assert_allclose(np.diag(kernel), np.ones(2), atol=1.0e-12)
    assert np.all((kernel >= 0.0) & (kernel <= 1.0))
    assert report["dependency_contract"]["quantum_module"] == "scpn_quantum_control.control.q_disruption_iter"
    certificate = report["report_certificate"]
    assert certificate["report_kind"] == "kernel-advisory"
    assert certificate["report_schema_version"] == report["schema_version"]
    assert certificate["dependency_contract_schema_version"] == report["dependency_contract"]["schema_version"]
    assert certificate["dependency_contract_sha256"] == report["dependency_contract"]["contract_sha256"]
    assert certificate["admitted_for_control"] is False
    assert certificate["external_validation_required"] is True
    assert len(certificate["content_sha256"]) == 64
    assert len(certificate["certificate_sha256"]) == 64
    assert validate_quantum_disruption_kernel_report(report) == report


def test_quantum_disruption_bridge_fails_closed_when_quantum_dependency_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scpn_control.control.quantum_disruption_bridge import (
        QuantumDisruptionBridgeConfig,
        run_quantum_disruption_bridge,
        validate_quantum_disruption_bridge_report,
    )

    real_import = importlib.import_module

    def guarded_import(name: str, package: str | None = None) -> types.ModuleType:
        if name == "scpn_quantum_control.control.q_disruption_iter":
            raise ModuleNotFoundError(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import)

    report = run_quantum_disruption_bridge(
        _control_features(),
        extra_iter_features=_extra_iter_features(),
        config=QuantumDisruptionBridgeConfig(require_quantum_backend=False),
    )

    assert report["status"] == "quantum-unavailable"
    assert report["quantum_available"] is False
    assert report["quantum_score"] is None
    assert report["admitted_for_control"] is False
    assert 0.0 <= report["classical_baseline_score"] <= 1.0
    assert validate_quantum_disruption_bridge_report(report) == report


def test_quantum_disruption_bridge_calls_quantum_owner_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    from scpn_control.control.quantum_disruption_bridge import (
        QuantumDisruptionBridgeConfig,
        run_quantum_disruption_bridge,
        validate_quantum_disruption_bridge_report,
    )

    class FakeClassifier:
        def __init__(self, seed: int) -> None:
            self.seed = seed

        def predict(self, features: np.ndarray) -> float:
            assert features.shape == (11,)
            assert np.all((features >= 0.0) & (features <= 1.0))
            return 0.73

    fake_module = types.SimpleNamespace(QuantumDisruptionClassifier=FakeClassifier)
    real_import = importlib.import_module

    def guarded_import(name: str, package: str | None = None) -> types.ModuleType:
        if name == "scpn_quantum_control.control.q_disruption_iter":
            return fake_module  # type: ignore[return-value]
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import)

    report = run_quantum_disruption_bridge(
        _control_features(),
        extra_iter_features=_extra_iter_features(),
        config=QuantumDisruptionBridgeConfig(seed=123),
    )

    assert report["status"] == "advisory"
    assert report["quantum_available"] is True
    assert report["quantum_score"] == 0.73
    assert report["quantum_backend_owner"] == "scpn-quantum-control"
    assert report["control_facade_owner"] == "scpn-control"
    assert report["admitted_for_control"] is False
    assert report["admission_evidence"]["decision"] == "advisory_only"
    assert report["admission_evidence"]["defaults_used"] == []
    assert "external_validation_required" in report["admission_evidence"]["reasons"]
    assert len(report["admission_evidence"]["control_features_sha256"]) == 64
    assert len(report["admission_evidence"]["feature_mapping_sha256"]) == 64
    assert report["dependency_contract"]["quantum_module"] == "scpn_quantum_control.control.q_disruption_iter"
    certificate = report["report_certificate"]
    assert certificate["report_kind"] == "bridge-advisory"
    assert certificate["report_schema_version"] == report["schema_version"]
    assert certificate["control_facade_owner"] == "scpn-control"
    assert certificate["quantum_backend_owner"] == "scpn-quantum-control"
    assert certificate["dependency_contract_schema_version"] == report["dependency_contract"]["schema_version"]
    assert certificate["dependency_contract_sha256"] == report["dependency_contract"]["contract_sha256"]
    assert certificate["publication_safe"] is False
    assert certificate["admitted_for_control"] is False
    assert "require_external_evidence" in certificate["required_downstream_policy"]
    assert len(certificate["content_sha256"]) == 64
    assert len(certificate["certificate_sha256"]) == 64
    assert validate_quantum_disruption_bridge_report(report) == report


def test_quantum_disruption_bridge_report_rejects_certificate_kind_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scpn_control.control.quantum_disruption_bridge import (
        QuantumDisruptionBridgeConfig,
        quantum_disruption_kernel_matrix,
        run_quantum_disruption_bridge,
        validate_quantum_disruption_bridge_report,
    )

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, package=None: (
            types.SimpleNamespace(
                QuantumDisruptionClassifier=type(
                    "FakeClassifier",
                    (),
                    {"__init__": lambda self, seed: None, "predict": lambda self, features: 0.44},
                )
            )
            if name == "scpn_quantum_control.control.q_disruption_iter"
            else importlib.import_module(name, package)
        ),
    )

    bridge_report = run_quantum_disruption_bridge(
        _control_features(),
        extra_iter_features=_extra_iter_features(),
        config=QuantumDisruptionBridgeConfig(seed=11),
    )
    kernel_report = quantum_disruption_kernel_matrix(
        np.vstack([_control_features(), _control_features() * np.array([1.05, 1, 1, 1, 1, 1, 1, 1])]),
        config=QuantumDisruptionBridgeConfig(allow_center_defaults=True, seed=11),
    )
    replayed = dict(bridge_report)
    replayed["report_certificate"] = kernel_report["report_certificate"]
    replayed["payload_sha256"] = bridge_report["payload_sha256"]

    with pytest.raises(ValueError, match="report_certificate"):
        validate_quantum_disruption_bridge_report(replayed)


def test_quantum_disruption_bridge_report_rejects_dependency_contract_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scpn_control.control.quantum_disruption_bridge import (
        QuantumDisruptionBridgeConfig,
        run_quantum_disruption_bridge,
        validate_quantum_disruption_bridge_report,
    )

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, package=None: (
            types.SimpleNamespace(
                QuantumDisruptionClassifier=type(
                    "FakeClassifier",
                    (),
                    {"__init__": lambda self, seed: None, "predict": lambda self, features: 0.45},
                )
            )
            if name == "scpn_quantum_control.control.q_disruption_iter"
            else importlib.import_module(name, package)
        ),
    )

    report = run_quantum_disruption_bridge(
        _control_features(),
        extra_iter_features=_extra_iter_features(),
        config=QuantumDisruptionBridgeConfig(seed=13),
    )
    replayed = dict(report)
    replayed["dependency_contract"] = {
        **report["dependency_contract"],
        "quantum_module": "scpn_quantum_control.control.replayed_backend",
    }
    replayed["payload_sha256"] = report["payload_sha256"]

    with pytest.raises(ValueError, match="dependency contract quantum_module"):
        validate_quantum_disruption_bridge_report(replayed)


def test_quantum_disruption_bridge_report_rejects_certificate_dependency_digest_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scpn_control.control.quantum_disruption_bridge import (
        QuantumDisruptionBridgeConfig,
        run_quantum_disruption_bridge,
        validate_quantum_disruption_bridge_report,
    )

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, package=None: (
            types.SimpleNamespace(
                QuantumDisruptionClassifier=type(
                    "FakeClassifier",
                    (),
                    {"__init__": lambda self, seed: None, "predict": lambda self, features: 0.46},
                )
            )
            if name == "scpn_quantum_control.control.q_disruption_iter"
            else importlib.import_module(name, package)
        ),
    )

    report = run_quantum_disruption_bridge(
        _control_features(),
        extra_iter_features=_extra_iter_features(),
        config=QuantumDisruptionBridgeConfig(seed=14),
    )
    replayed = dict(report)
    replayed["report_certificate"] = {
        **report["report_certificate"],
        "dependency_contract_sha256": "f" * 64,
    }
    replayed["payload_sha256"] = report["payload_sha256"]

    with pytest.raises(ValueError, match="dependency_contract_sha256"):
        validate_quantum_disruption_bridge_report(replayed)


def test_quantum_disruption_bridge_report_marks_center_defaults_as_bounded_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scpn_control.control.quantum_disruption_bridge import (
        QuantumDisruptionBridgeConfig,
        run_quantum_disruption_bridge,
        validate_quantum_disruption_bridge_report,
    )

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, package=None: (
            types.SimpleNamespace(
                QuantumDisruptionClassifier=type(
                    "FakeClassifier",
                    (),
                    {"__init__": lambda self, seed: None, "predict": lambda self, features: 0.51},
                )
            )
            if name == "scpn_quantum_control.control.q_disruption_iter"
            else importlib.import_module(name, package)
        ),
    )

    report = run_quantum_disruption_bridge(
        _control_features(),
        config=QuantumDisruptionBridgeConfig(allow_center_defaults=True),
    )

    assert report["admission_evidence"]["decision"] == "advisory_only"
    assert report["admission_evidence"]["defaults_used"] == ["P_rad", "V_loop", "W_stored", "kappa", "dIp_dt"]
    assert "center_defaults_used" in report["admission_evidence"]["reasons"]
    assert report["admission_evidence"]["publication_safe"] is False
    assert validate_quantum_disruption_bridge_report(report) == report


def test_quantum_disruption_bridge_report_rejects_admission_digest_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scpn_control.control.quantum_disruption_bridge import (
        QuantumDisruptionBridgeConfig,
        run_quantum_disruption_bridge,
        validate_quantum_disruption_bridge_report,
    )

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, package=None: (
            types.SimpleNamespace(
                QuantumDisruptionClassifier=type(
                    "FakeClassifier",
                    (),
                    {"__init__": lambda self, seed: None, "predict": lambda self, features: 0.42},
                )
            )
            if name == "scpn_quantum_control.control.q_disruption_iter"
            else importlib.import_module(name, package)
        ),
    )

    report = run_quantum_disruption_bridge(
        _control_features(),
        extra_iter_features=_extra_iter_features(),
        config=QuantumDisruptionBridgeConfig(seed=7),
    )
    replayed = dict(report)
    replayed["admission_evidence"] = {
        **report["admission_evidence"],
        "feature_mapping_sha256": "f" * 64,
    }
    replayed["payload_sha256"] = report["payload_sha256"]

    with pytest.raises(ValueError, match="feature_mapping_sha256"):
        validate_quantum_disruption_bridge_report(replayed)


def test_quantum_disruption_bridge_report_rejects_tampering(monkeypatch: pytest.MonkeyPatch) -> None:
    from scpn_control.control.quantum_disruption_bridge import (
        QuantumDisruptionBridgeConfig,
        run_quantum_disruption_bridge,
        validate_quantum_disruption_bridge_report,
    )

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, package=None: (
            types.SimpleNamespace(
                QuantumDisruptionClassifier=type(
                    "FakeClassifier",
                    (),
                    {"__init__": lambda self, seed: None, "predict": lambda self, features: 0.42},
                )
            )
            if name == "scpn_quantum_control.control.q_disruption_iter"
            else importlib.import_module(name, package)
        ),
    )
    report = run_quantum_disruption_bridge(
        _control_features(),
        extra_iter_features=_extra_iter_features(),
        config=QuantumDisruptionBridgeConfig(seed=321),
    )
    tampered = dict(report)
    tampered["quantum_score"] = 0.99

    with pytest.raises(ValueError, match="report_certificate content_sha256"):
        validate_quantum_disruption_bridge_report(tampered)
