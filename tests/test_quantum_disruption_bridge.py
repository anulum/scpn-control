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
import math
import sys
import types
from copy import deepcopy
from typing import Any, cast

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


def _unavailable_report() -> dict[str, Any]:
    from scpn_control.control.quantum_disruption_bridge import (
        QuantumDisruptionBridgeConfig,
        run_quantum_disruption_bridge,
    )

    return cast(
        dict[str, Any],
        run_quantum_disruption_bridge(
            _control_features(),
            extra_iter_features=_extra_iter_features(),
            config=QuantumDisruptionBridgeConfig(quantum_module="missing.quantum.backend"),
        ),
    )


def _kernel_report() -> dict[str, Any]:
    from scpn_control.control.quantum_disruption_bridge import (
        QuantumDisruptionBridgeConfig,
        quantum_disruption_kernel_matrix,
    )

    return cast(
        dict[str, Any],
        quantum_disruption_kernel_matrix(
            np.vstack([_control_features(), _control_features()]),
            config=QuantumDisruptionBridgeConfig(allow_center_defaults=True),
        ),
    )


def test_quantum_disruption_bridge_import_does_not_import_quantum_package() -> None:
    sys.modules.pop("scpn_quantum_control", None)
    module = importlib.import_module("scpn_control.control.quantum_disruption_bridge")

    assert module.QUANTUM_BACKEND_OWNER == "scpn-quantum-control"
    assert "scpn_quantum_control" not in sys.modules


def test_quantum_disruption_config_rejects_non_contractual_values() -> None:
    from scpn_control.control.quantum_disruption_bridge import QuantumDisruptionBridgeConfig

    invalid_configs: tuple[dict[str, Any], ...] = (
        {"allow_center_defaults": "yes"},
        {"require_quantum_backend": 1},
        {"seed": True},
        {"claim_status": "facility_validated"},
        {"backend_profile": ""},
        {"quantum_module": " "},
        {"source_mode": ""},
    )

    for kwargs in invalid_configs:
        with pytest.raises(ValueError):
            QuantumDisruptionBridgeConfig(**kwargs)


def test_quantum_disruption_feature_mapping_rejects_bad_extra_features() -> None:
    from scpn_control.control.quantum_disruption_bridge import (
        QuantumDisruptionBridgeConfig,
        map_control_features_to_iter,
    )

    config = QuantumDisruptionBridgeConfig(allow_center_defaults=True)
    with pytest.raises(ValueError, match="unsupported extra ITER feature"):
        map_control_features_to_iter(_control_features(), extra_iter_features={"bad": 1.0}, config=config)
    with pytest.raises(ValueError, match="must be finite"):
        map_control_features_to_iter(_control_features(), extra_iter_features={"P_rad": np.inf}, config=config)
    with pytest.raises(ValueError, match="8 values"):
        map_control_features_to_iter([1.0, 2.0], config=config)


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
    decision = report["advisory_decision"]
    assert decision["schema_version"] == "scpn-control.quantum-disruption-advisory-decision.v1"
    assert decision["score_basis"] == "classical_baseline_score"
    assert decision["control_action"] == "blocked"
    assert decision["admitted_for_control"] is False
    assert decision["backend_contract_validated"] is False
    assert "quantum_backend_unavailable" in decision["reasons"]
    assert report["report_certificate"]["advisory_decision_sha256"] == decision["decision_sha256"]
    assert report["backend_contract_attestation"]["status"] == "backend_unavailable"
    assert report["backend_contract_attestation"]["backend_contract_validated"] is False
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

    from scpn_control.control.quantum_disruption_bridge import quantum_disruption_dependency_contract

    fake_module = types.SimpleNamespace(
        QuantumDisruptionClassifier=FakeClassifier,
        scpn_control_bridge_dependency_contract=quantum_disruption_dependency_contract,
    )
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
    decision = report["advisory_decision"]
    assert decision["risk_score"] == 0.73
    assert decision["score_basis"] == "quantum_score"
    assert decision["risk_band"] == "high"
    assert decision["thresholds"] == {"elevated": 0.4, "high": 0.7}
    assert decision["backend_contract_validated"] is True
    assert decision["control_action"] == "blocked"
    assert decision["admitted_for_control"] is False
    assert report["quantum_backend_owner"] == "scpn-quantum-control"
    assert report["control_facade_owner"] == "scpn-control"
    assert report["admitted_for_control"] is False
    assert report["admission_evidence"]["decision"] == "advisory_only"
    assert report["admission_evidence"]["defaults_used"] == []
    assert "external_validation_required" in report["admission_evidence"]["reasons"]
    assert len(report["admission_evidence"]["control_features_sha256"]) == 64
    assert len(report["admission_evidence"]["feature_mapping_sha256"]) == 64
    assert report["dependency_contract"]["quantum_module"] == "scpn_quantum_control.control.q_disruption_iter"
    assert report["backend_contract_attestation"]["status"] == "matched"
    assert report["backend_contract_attestation"]["backend_contract_validated"] is True
    assert (
        report["backend_contract_attestation"]["observed_contract_sha256"]
        == report["dependency_contract"]["contract_sha256"]
    )
    certificate = report["report_certificate"]
    assert certificate["report_kind"] == "bridge-advisory"
    assert certificate["report_schema_version"] == report["schema_version"]
    assert certificate["control_facade_owner"] == "scpn-control"
    assert certificate["quantum_backend_owner"] == "scpn-quantum-control"
    assert certificate["dependency_contract_schema_version"] == report["dependency_contract"]["schema_version"]
    assert certificate["dependency_contract_sha256"] == report["dependency_contract"]["contract_sha256"]
    assert (
        certificate["backend_contract_attestation_sha256"]
        == report["backend_contract_attestation"]["attestation_sha256"]
    )
    assert certificate["advisory_decision_sha256"] == decision["decision_sha256"]
    assert certificate["publication_safe"] is False
    assert certificate["admitted_for_control"] is False
    assert "require_external_evidence" in certificate["required_downstream_policy"]
    assert len(certificate["content_sha256"]) == 64
    assert len(certificate["certificate_sha256"]) == 64
    assert validate_quantum_disruption_bridge_report(report) == report


def test_quantum_disruption_bridge_records_available_backend_without_contract(
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
                    {"__init__": lambda self, seed: None, "predict": lambda self, features: 0.41},
                )
            )
            if name == "scpn_quantum_control.control.q_disruption_iter"
            else importlib.import_module(name, package)
        ),
    )

    report = run_quantum_disruption_bridge(
        _control_features(),
        extra_iter_features=_extra_iter_features(),
        config=QuantumDisruptionBridgeConfig(seed=15),
    )

    assert report["status"] == "advisory"
    assert report["backend_contract_attestation"]["status"] == "not_exposed"
    assert report["backend_contract_attestation"]["backend_contract_validated"] is False
    assert report["backend_contract_attestation"]["observed_contract_sha256"] is None
    assert "backend_contract_not_exposed" in report["backend_contract_attestation"]["reasons"]
    assert validate_quantum_disruption_bridge_report(report) == report


def test_quantum_disruption_bridge_fails_closed_on_backend_contract_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scpn_control.control.quantum_disruption_bridge import (
        QuantumDisruptionBridgeConfig,
        quantum_disruption_dependency_contract,
        run_quantum_disruption_bridge,
    )

    expected = quantum_disruption_dependency_contract()
    drifted = {**expected, "quantum_module": "scpn_quantum_control.control.drifted"}

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, package=None: (
            types.SimpleNamespace(
                QuantumDisruptionClassifier=type(
                    "FakeClassifier",
                    (),
                    {"__init__": lambda self, seed: None, "predict": lambda self, features: 0.43},
                ),
                scpn_control_bridge_dependency_contract=lambda: drifted,
            )
            if name == "scpn_quantum_control.control.q_disruption_iter"
            else importlib.import_module(name, package)
        ),
    )

    with pytest.raises(RuntimeError, match="backend contract"):
        run_quantum_disruption_bridge(
            _control_features(),
            extra_iter_features=_extra_iter_features(),
            config=QuantumDisruptionBridgeConfig(seed=16),
        )


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


def test_quantum_disruption_bridge_report_rejects_backend_attestation_digest_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scpn_control.control.quantum_disruption_bridge import (
        QuantumDisruptionBridgeConfig,
        quantum_disruption_dependency_contract,
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
                    {"__init__": lambda self, seed: None, "predict": lambda self, features: 0.47},
                ),
                scpn_control_bridge_dependency_contract=quantum_disruption_dependency_contract,
            )
            if name == "scpn_quantum_control.control.q_disruption_iter"
            else importlib.import_module(name, package)
        ),
    )

    report = run_quantum_disruption_bridge(
        _control_features(),
        extra_iter_features=_extra_iter_features(),
        config=QuantumDisruptionBridgeConfig(seed=17),
    )
    replayed = dict(report)
    replayed["report_certificate"] = {
        **report["report_certificate"],
        "backend_contract_attestation_sha256": "f" * 64,
    }
    replayed["payload_sha256"] = report["payload_sha256"]

    with pytest.raises(ValueError, match="backend_contract_attestation_sha256"):
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


def test_quantum_disruption_bridge_report_rejects_advisory_decision_replay(
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
                    {"__init__": lambda self, seed: None, "predict": lambda self, features: 0.82},
                )
            )
            if name == "scpn_quantum_control.control.q_disruption_iter"
            else importlib.import_module(name, package)
        ),
    )

    report = run_quantum_disruption_bridge(
        _control_features(),
        extra_iter_features=_extra_iter_features(),
        config=QuantumDisruptionBridgeConfig(seed=18),
    )
    replayed = dict(report)
    replayed["advisory_decision"] = {
        **report["advisory_decision"],
        "risk_band": "low",
    }
    replayed["payload_sha256"] = report["payload_sha256"]

    with pytest.raises(ValueError, match="advisory_decision"):
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


def test_quantum_disruption_bridge_requires_quantum_backend_when_configured() -> None:
    from scpn_control.control.quantum_disruption_bridge import (
        QuantumDisruptionBridgeConfig,
        run_quantum_disruption_bridge,
    )

    with pytest.raises(RuntimeError, match="quantum disruption backend is required"):
        run_quantum_disruption_bridge(
            _control_features(),
            extra_iter_features=_extra_iter_features(),
            config=QuantumDisruptionBridgeConfig(
                require_quantum_backend=True,
                quantum_module="missing.quantum.backend",
            ),
        )


def test_quantum_disruption_dependency_contract_rejects_top_level_tampering() -> None:
    from scpn_control.control.quantum_disruption_bridge import (
        quantum_disruption_dependency_contract,
        validate_quantum_disruption_dependency_contract,
    )

    contract = quantum_disruption_dependency_contract()
    cases: tuple[tuple[str, object, str], ...] = (
        ("schema_version", "bad", "schema_version"),
        ("control_facade_owner", "other", "control_facade_owner"),
        ("quantum_backend_owner", "other", "quantum_backend_owner"),
        ("quantum_module", "other.module", "quantum_module"),
        ("claim_boundary", "unsafe", "claim_boundary"),
        ("admitted_for_control", True, "admitted_for_control"),
        ("publication_safe", True, "publication_safe"),
        ("required_downstream_policy", ["do_not_admit_control_action"], "missing downstream policy"),
        ("contract_sha256", "a" * 64, "contract_sha256 does not match"),
    )

    for key, value, message in cases:
        tampered = deepcopy(contract)
        tampered[key] = value
        with pytest.raises(ValueError, match=message):
            validate_quantum_disruption_dependency_contract(tampered)


def test_quantum_disruption_dependency_contract_rejects_nested_surface_drift() -> None:
    from scpn_control.control.quantum_disruption_bridge import (
        quantum_disruption_dependency_contract,
        validate_quantum_disruption_dependency_contract,
    )

    contract = quantum_disruption_dependency_contract()
    cases: tuple[tuple[tuple[str, ...], object, str], ...] = (
        (("report_schema_versions",), [], "report_schema_versions"),
        (("report_schema_versions", "bridge"), "bad", "report_schema_versions"),
        (("required_public_surface",), [], "required_public_surface"),
        (("required_public_surface", "classifier_class"), "OtherClassifier", "classifier_class"),
        (("required_public_surface", "constructor_kwargs"), [], "constructor_kwargs"),
        (("required_public_surface", "predict_method"), "score", "predict_method"),
        (("required_public_surface", "predict_input"), [], "predict_input"),
        (("required_public_surface", "predict_input", "shape"), [8], "predict_input shape"),
        (("required_public_surface", "predict_input", "feature_names"), [], "feature_names"),
        (("required_public_surface", "predict_input", "normalised_range"), [0.0, 2.0], "normalised_range"),
        (("required_public_surface", "predict_input", "dtype"), "float32", "dtype"),
        (("required_public_surface", "predict_output"), [], "predict_output"),
        (("required_public_surface", "predict_output", "type"), "vector", "predict_output type"),
        (("required_public_surface", "predict_output", "range"), [0.0, 2.0], "predict_output range"),
        (("feature_contract",), [], "feature_contract"),
        (("feature_contract", "control_feature_names"), [], "control_feature_names"),
        (("feature_contract", "iter_feature_names"), [], "iter_feature_names"),
        (("feature_contract", "extra_iter_features"), [], "extra_iter_features"),
        (("feature_contract", "centre_defaults_allowed_only_when_declared"), False, "centre default policy"),
        (("dependency_groups",), [], "dependency_groups"),
        (("dependency_groups", "control_runtime"), [], "control_runtime"),
        (("dependency_groups", "quantum_core"), [], "quantum_core"),
        (("dependency_groups", "quantum_optional_providers"), [], "quantum_optional_providers"),
    )

    for path, value, message in cases:
        tampered = deepcopy(contract)
        cursor: dict[str, Any] = tampered
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value
        with pytest.raises(ValueError, match=message):
            validate_quantum_disruption_dependency_contract(tampered)


def test_quantum_disruption_bridge_report_rejects_top_level_tampering() -> None:
    from scpn_control.control.quantum_disruption_bridge import validate_quantum_disruption_bridge_report

    report = _unavailable_report()
    cases: tuple[tuple[str, object, str], ...] = (
        ("schema_version", "bad", "schema_version"),
        ("status", "published", "status"),
        ("claim_boundary", "unsafe", "claim_boundary"),
        ("control_facade_owner", "other", "control_facade_owner"),
        ("quantum_backend_owner", "other", "quantum_backend_owner"),
        ("created_at", "", "created_at"),
        ("claim_status", "facility_validated", "claim_status"),
        ("quantum_available", "false", "quantum_available"),
        ("admitted_for_control", True, "admit control action"),
        ("human_review_required", False, "human review"),
        ("config", [], "config"),
        ("quantum_module", "", "quantum_module"),
        ("payload_sha256", "a" * 64, "payload_sha256 does not match"),
    )

    for key, value, message in cases:
        tampered = deepcopy(report)
        tampered[key] = value
        with pytest.raises(ValueError, match=message):
            validate_quantum_disruption_bridge_report(tampered)

    conflicts = deepcopy(report)
    conflicts["quantum_available"] = True
    with pytest.raises(ValueError, match="unavailable status conflicts"):
        validate_quantum_disruption_bridge_report(conflicts)


def test_quantum_disruption_bridge_report_rejects_feature_and_admission_tampering() -> None:
    from scpn_control.control.quantum_disruption_bridge import validate_quantum_disruption_bridge_report

    report = _unavailable_report()
    cases: tuple[tuple[tuple[str, ...], object, str], ...] = (
        (("feature_mapping",), [], "feature_mapping"),
        (("feature_mapping", "normalized_iter_features"), [2.0] * 11, "normalized features"),
        (("feature_mapping", "control_feature_names"), [], "control feature names"),
        (("feature_mapping", "iter_feature_names"), [], "ITER feature names"),
        (("feature_mapping", "claim_status"), "facility_validated", "claim_status"),
        (("feature_mapping", "publication_safe"), True, "publication_safe"),
        (("admission_evidence",), [], "admission_evidence"),
        (("admission_evidence", "decision"), "allow", "decision"),
        (("admission_evidence", "publication_safe"), True, "publication_safe"),
        (("admission_evidence", "admitted_for_control"), True, "admitted_for_control"),
        (("admission_evidence", "defaults_used"), ["drifted"], "defaults_used must match"),
        (("admission_evidence", "reasons"), [], "missing reason"),
        (("admission_evidence", "required_external_evidence"), [], "missing external evidence"),
        (("admission_evidence", "control_features_sha256"), "bad", "SHA-256"),
        (("admission_evidence", "normalized_iter_features_sha256"), "a" * 64, "normalized_iter_features_sha256"),
        (("admission_evidence", "feature_mapping_sha256"), "a" * 64, "feature_mapping_sha256"),
        (("classical_baseline_score",), 2.0, "classical_baseline_score"),
        (("risk_score",), math.inf, "risk_score"),
    )

    for path, value, message in cases:
        tampered = deepcopy(report)
        cursor: dict[str, Any] = tampered
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value
        with pytest.raises(ValueError, match=message):
            validate_quantum_disruption_bridge_report(tampered)


def test_quantum_disruption_kernel_report_rejects_matrix_and_certificate_tampering() -> None:
    from scpn_control.control.quantum_disruption_bridge import validate_quantum_disruption_kernel_report

    report = _kernel_report()
    cases: tuple[tuple[tuple[str, ...], object, str], ...] = (
        (("schema_version",), "bad", "schema_version"),
        (("status",), "published", "status"),
        (("claim_boundary",), "unsafe", "claim_boundary"),
        (("control_facade_owner",), "other", "control_facade_owner"),
        (("quantum_backend_owner",), "other", "quantum_backend_owner"),
        (("admitted_for_control",), True, "admit control action"),
        (("kernel_matrix",), [[1.0, 0.2, 0.1]], "matrix shape"),
        (("kernel_matrix",), [[1.0, float("nan")], [0.0, 1.0]], "must be finite"),
        (("kernel_matrix",), [[1.0, 2.0], [2.0, 1.0]], r"\[0, 1\]"),
        (("kernel_matrix",), [[1.0, 0.2], [0.3, 1.0]], "symmetric"),
        (("kernel_matrix",), [[0.9, 0.2], [0.2, 1.0]], "diagonal"),
        (("payload_sha256",), "a" * 64, "payload_sha256 does not match"),
        (("report_certificate",), [], "report_certificate"),
        (("report_certificate", "schema_version"), "bad", "schema_version"),
        (("report_certificate", "report_kind"), "bridge-advisory", "report_kind"),
        (("report_certificate", "control_facade_owner"), "other", "control_facade_owner"),
        (("report_certificate", "dependency_contract_sha256"), "bad", "SHA-256"),
        (("report_certificate", "backend_contract_attestation_sha256"), "bad", "SHA-256"),
        (("report_certificate", "admitted_for_control"), True, "admitted_for_control"),
        (("report_certificate", "publication_safe"), True, "publication_safe"),
        (("report_certificate", "external_validation_required"), False, "external_validation_required"),
        (("report_certificate", "required_downstream_policy"), [], "missing downstream policy"),
        (("report_certificate", "claim_boundary_sha256"), "bad", "SHA-256"),
        (("report_certificate", "content_sha256"), "bad", "SHA-256"),
        (("report_certificate", "certificate_sha256"), "bad", "SHA-256"),
    )

    for path, value, message in cases:
        tampered = deepcopy(report)
        cursor: dict[str, Any] = tampered
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value
        with pytest.raises(ValueError, match=message):
            validate_quantum_disruption_kernel_report(tampered)


def test_quantum_disruption_kernel_and_mapping_reject_bad_numeric_shapes() -> None:
    from scpn_control.control.quantum_disruption_bridge import (
        QuantumDisruptionBridgeConfig,
        normalize_iter_features,
        quantum_disruption_kernel_matrix,
    )

    config = QuantumDisruptionBridgeConfig(allow_center_defaults=True)
    with pytest.raises(ValueError, match="samples_a must have shape"):
        quantum_disruption_kernel_matrix([[1.0, 2.0, 3.0]], config=config)
    with pytest.raises(ValueError, match="samples_a must be finite"):
        quantum_disruption_kernel_matrix([[float("nan")] * 8], config=config)
    with pytest.raises(ValueError, match="raw_iter_features must contain 11 values"):
        normalize_iter_features([1.0, 2.0])


def test_quantum_disruption_bridge_report_rejects_backend_attestation_tampering() -> None:
    from scpn_control.control.quantum_disruption_bridge import validate_quantum_disruption_bridge_report

    report = _unavailable_report()
    valid_digest = report["dependency_contract"]["contract_sha256"]
    cases: tuple[tuple[object, str], ...] = (
        ([], "backend_contract_attestation"),
        ({**report["backend_contract_attestation"], "status": "unknown"}, "status"),
        (
            {**report["backend_contract_attestation"], "backend_contract_validated": "false"},
            "backend_contract_validated",
        ),
        ({**report["backend_contract_attestation"], "expected_contract_sha256": "bad"}, "expected_contract_sha256"),
        (
            {**report["backend_contract_attestation"], "expected_contract_sha256": "a" * 64},
            "expected_contract_sha256 mismatch",
        ),
        ({**report["backend_contract_attestation"], "observed_contract_sha256": "bad"}, "observed_contract_sha256"),
        ({**report["backend_contract_attestation"], "backend_contract_validated": True}, "non-matched status"),
        ({**report["backend_contract_attestation"], "observed_contract_sha256": valid_digest}, "non-matched status"),
        ({**report["backend_contract_attestation"], "reasons": []}, "required reason"),
        ({**report["backend_contract_attestation"], "attestation_sha256": "bad"}, "attestation_sha256"),
        ({**report["backend_contract_attestation"], "attestation_sha256": "a" * 64}, "attestation_sha256 mismatch"),
    )

    for value, message in cases:
        tampered = deepcopy(report)
        tampered["backend_contract_attestation"] = value
        with pytest.raises(ValueError, match=message):
            validate_quantum_disruption_bridge_report(tampered)


def test_quantum_disruption_bridge_report_rejects_advisory_decision_tampering() -> None:
    from scpn_control.control.quantum_disruption_bridge import validate_quantum_disruption_bridge_report

    report = _unavailable_report()
    cases: tuple[tuple[object, str], ...] = (
        ([], "advisory_decision"),
        ({**report["advisory_decision"], "schema_version": "bad"}, "schema_version"),
        ({**report["advisory_decision"], "risk_score": 0.0}, "risk_score mismatch"),
        ({**report["advisory_decision"], "score_basis": "quantum_score"}, "score_basis"),
        ({**report["advisory_decision"], "risk_band": "high"}, "risk_band"),
        ({**report["advisory_decision"], "thresholds": {"elevated": 0.5}}, "thresholds"),
        ({**report["advisory_decision"], "control_action": "allow"}, "control_action"),
        ({**report["advisory_decision"], "admitted_for_control": True}, "admitted_for_control"),
        ({**report["advisory_decision"], "publication_safe": True}, "publication_safe"),
        ({**report["advisory_decision"], "human_review_required": False}, "human_review_required"),
        ({**report["advisory_decision"], "external_validation_required": False}, "external_validation_required"),
        ({**report["advisory_decision"], "backend_contract_validated": True}, "backend_contract_validated"),
        ({**report["advisory_decision"], "reasons": []}, "missing reason"),
        (
            {
                **report["advisory_decision"],
                "reasons": ["advisory_only", "external_validation_required", "control_admission_blocked"],
            },
            "quantum_score_unavailable",
        ),
        ({**report["advisory_decision"], "decision_sha256": "bad"}, "decision_sha256"),
        ({**report["advisory_decision"], "decision_sha256": "a" * 64}, "decision_sha256 mismatch"),
    )

    for value, message in cases:
        tampered = deepcopy(report)
        tampered["advisory_decision"] = value
        with pytest.raises(ValueError, match=message):
            validate_quantum_disruption_bridge_report(tampered)


def test_quantum_disruption_kernel_report_rejects_dependency_contract_object_drift() -> None:
    from scpn_control.control.quantum_disruption_bridge import validate_quantum_disruption_kernel_report

    report = _kernel_report()
    tampered = deepcopy(report)
    tampered["dependency_contract"] = []
    with pytest.raises(ValueError, match="dependency_contract"):
        validate_quantum_disruption_kernel_report(tampered)

    invalid_digest = deepcopy(report)
    invalid_digest["payload_sha256"] = "bad"
    with pytest.raises(ValueError, match="payload_sha256"):
        validate_quantum_disruption_kernel_report(invalid_digest)


def test_quantum_disruption_public_validators_reject_non_object_payloads() -> None:
    from scpn_control.control.quantum_disruption_bridge import (
        validate_quantum_disruption_bridge_report,
        validate_quantum_disruption_dependency_contract,
        validate_quantum_disruption_kernel_report,
    )

    with pytest.raises(ValueError, match="dependency contract must be an object"):
        validate_quantum_disruption_dependency_contract(cast(dict[str, Any], []))
    with pytest.raises(ValueError, match="bridge report must be an object"):
        validate_quantum_disruption_bridge_report(cast(dict[str, Any], []))
    with pytest.raises(ValueError, match="kernel report must be an object"):
        validate_quantum_disruption_kernel_report(cast(dict[str, Any], []))


def test_quantum_disruption_dependency_contract_rejects_policy_and_digest_shape() -> None:
    from scpn_control.control.quantum_disruption_bridge import (
        quantum_disruption_dependency_contract,
        validate_quantum_disruption_dependency_contract,
    )

    contract = quantum_disruption_dependency_contract()
    policy_shape = deepcopy(contract)
    policy_shape["required_downstream_policy"] = "blocked"
    with pytest.raises(ValueError, match="required_downstream_policy"):
        validate_quantum_disruption_dependency_contract(policy_shape)

    digest_shape = deepcopy(contract)
    digest_shape["contract_sha256"] = "bad"
    with pytest.raises(ValueError, match="contract_sha256 must be"):
        validate_quantum_disruption_dependency_contract(digest_shape)
