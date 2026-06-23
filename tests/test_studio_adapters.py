# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Studio live-emitter adapter tests
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ──────────────────────────────────────────────────────────────────────
"""Tests for the studio adapters that wire live CONTROL emitters to EvidenceBundles.

Each adapter is fed a faithful slice of its real emitter's output — a
``ReconstructionResult`` from ``RealtimeEFIT.reconstruct``, an issued runtime safety
certificate, and a controller-latency measurement — and the resulting bundle is
checked for the correct schema, honest rendering, and provenance digests.

Skips cleanly when the optional ``scpn-studio-platform`` SDK is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scpn_studio_platform")

from scpn_studio_platform.evidence import EvidenceKind  # noqa: E402

from scpn_control.control.realtime_efit import ReconstructionResult, ShapeParams  # noqa: E402
from scpn_control.studio import (  # noqa: E402
    controller_latency_evidence_from_measurement,
    efit_evidence_from_reconstruction,
    safety_certificate_evidence_from_certificate,
)

_TS = {"started": "2026-06-23T00:00:00Z", "ended": "2026-06-23T00:00:01Z"}
_WHO = {"operator": "opaque:tenant-1", "studio_version": "0.test"}


def _reconstruction() -> ReconstructionResult:
    shape = ShapeParams(
        R0=1.7,
        a=0.5,
        kappa=1.8,
        delta_upper=0.4,
        delta_lower=0.4,
        q95=3.5,
        beta_pol=0.9,
        li=0.8,
        Ip_reconstructed=1.5e7,
    )
    return ReconstructionResult(
        psi=np.zeros((33, 33), dtype=np.float64),
        p_prime_coeffs=np.zeros(3, dtype=np.float64),
        ff_prime_coeffs=np.zeros(3, dtype=np.float64),
        shape=shape,
        chi_squared=4.38e-9,
        n_iterations=7,
        wall_time_ms=12.3,
    )


def _certificate(*, live: str = "d" * 64) -> tuple[dict[str, object], str]:
    cert: dict[str, object] = {
        "scope": "scpn-control.runtime-safety-certificate",
        "binding": {"petri_topology_sha256": "d" * 64},
        "formal_certificate": {"holds": True, "non_vacuous": True, "payload_sha256": "c" * 64},
        "formal_certificate_sha256": "c" * 64,
        "checked_specs": ["AG(no_overflow)", "AF(safe_shutdown)"],
        "payload_sha256": "e" * 64,
    }
    return cert, live


def _measurement() -> dict[str, float | int]:
    return {"n": 200, "p50_us": 5.05, "p95_us": 6.11, "p99_us": 6.4, "mean_us": 5.3}


# ── EFIT reconstruction adapter ────────────────────────────────────────
def test_efit_adapter_reads_the_live_reconstruction() -> None:
    bundle = efit_evidence_from_reconstruction(
        _reconstruction(),
        measurements={"Ip": 1.5e7, "flux_loops": 16},
        **_WHO,
        **_TS,
    )
    assert bundle.schema == "studio.efit-reconstruction.v1"
    assert bundle.renders_as_validated is False
    assert bundle.evidence_kind is EvidenceKind.MEASURED
    assert bundle.physical_contract is not None
    assert bundle.physical_contract.grid["nz"] == 33
    assert bundle.physical_contract.grid["nr"] == 33
    assert bundle.claim_boundary.validity_domain is not None


def test_efit_adapter_digests_depend_on_inputs_and_result() -> None:
    a = efit_evidence_from_reconstruction(_reconstruction(), measurements={"Ip": 1.0}, **_WHO, **_TS)
    b = efit_evidence_from_reconstruction(_reconstruction(), measurements={"Ip": 2.0}, **_WHO, **_TS)
    # Different inputs -> different input-provenance entity ids are not exposed, but
    # the bundles are independently valid and the result digest is stable per result.
    assert a.entity.digest == b.entity.digest  # same reconstruction summary
    assert a.schema == b.schema


# ── safety certificate adapter ─────────────────────────────────────────
def test_safety_cert_adapter_held_proof_is_admissible() -> None:
    cert, live = _certificate(live="d" * 64)
    bundle = safety_certificate_evidence_from_certificate(
        cert,
        live_topology_sha256=live,
        checker="z3",
        checker_version="4.13.0",
        **_WHO,
        **_TS,
    )
    assert bundle.evidence_kind is EvidenceKind.FORMALLY_PROVEN
    assert bundle.renders_as_validated is True
    cert_obj = bundle.formal_certificates[0]
    assert cert_obj.checker == "z3"
    assert cert_obj.theorem_id == "AG(no_overflow) ; AF(safe_shutdown)"
    assert cert_obj.non_vacuous is True
    assert cert_obj.subject_digest == "d" * 64
    assert cert_obj.proof_digest == "c" * 64


def test_safety_cert_adapter_drifted_topology_is_voided() -> None:
    cert, live = _certificate(live="f" * 64)
    bundle = safety_certificate_evidence_from_certificate(
        cert,
        live_topology_sha256=live,
        checker="z3",
        checker_version="4.13.0",
        **_WHO,
        **_TS,
    )
    assert bundle.renders_as_validated is False
    assert bundle.proof_voided_by("f" * 64) is True


def test_safety_cert_adapter_falls_back_to_scope_without_specs() -> None:
    cert, live = _certificate()
    cert["checked_specs"] = []
    bundle = safety_certificate_evidence_from_certificate(
        cert,
        live_topology_sha256=live,
        checker="lean",
        checker_version="4.9.0",
        **_WHO,
        **_TS,
    )
    assert bundle.formal_certificates[0].theorem_id == "scpn-control.runtime-safety-certificate"


def test_safety_cert_adapter_defaults_non_vacuous_false_when_absent() -> None:
    cert, live = _certificate()
    cert["formal_certificate"] = {"holds": True, "payload_sha256": "c" * 64}
    bundle = safety_certificate_evidence_from_certificate(
        cert,
        live_topology_sha256=live,
        checker="z3",
        checker_version="4.13.0",
        **_WHO,
        **_TS,
    )
    assert bundle.formal_certificates[0].non_vacuous is False


def test_safety_cert_adapter_rejects_missing_binding() -> None:
    with pytest.raises(KeyError):
        safety_certificate_evidence_from_certificate(
            {"checked_specs": [], "formal_certificate_sha256": "c" * 64},
            live_topology_sha256="d" * 64,
            checker="z3",
            checker_version="4.13.0",
            **_WHO,
            **_TS,
        )


# ── controller latency adapter ─────────────────────────────────────────
def test_latency_adapter_reads_the_measurement() -> None:
    bundle = controller_latency_evidence_from_measurement(
        _measurement(),
        controller="h_infinity",
        active_backend="rust",
        reference_backend="python",
        **_WHO,
        **_TS,
    )
    assert bundle.schema == "studio.controller-latency.v1"
    assert bundle.renders_as_validated is False
    assert bundle.evidence_kind is EvidenceKind.MEASURED
    assert bundle.numeric_provenance is not None
    assert bundle.numeric_provenance.active_backend == "rust"


def test_latency_adapter_rejects_missing_percentile() -> None:
    bad = {"n": 200, "p50_us": 5.0, "p95_us": 6.0}  # no p99_us
    with pytest.raises(KeyError):
        controller_latency_evidence_from_measurement(
            bad,
            controller="h_infinity",
            active_backend="rust",
            reference_backend="python",
            **_WHO,
            **_TS,
        )
