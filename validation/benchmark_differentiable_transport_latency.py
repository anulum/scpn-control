# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable transport latency benchmark
"""Publish bounded audited gradient-latency evidence for transport tuning."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scpn_control.core.differentiable_transport import (
    benchmark_transport_parameter_gradient_latency,
    benchmark_transport_rollout_source_gradient_latency,
    differentiable_transport_rollout,
    has_jax,
    save_transport_gradient_latency_report,
    save_transport_rollout_gradient_latency_report,
    transport_campaign_metadata,
    transport_full_fidelity_readiness_evidence,
)

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "differentiable_transport_latency.json"
MD_REPORT = REPORT_DIR / "differentiable_transport_latency.md"
ROLLOUT_JSON_REPORT = REPORT_DIR / "differentiable_transport_rollout_latency.json"
ROLLOUT_MD_REPORT = REPORT_DIR / "differentiable_transport_rollout_latency.md"
READINESS_JSON_REPORT = REPORT_DIR / "differentiable_transport_full_fidelity_readiness.json"
READINESS_MD_REPORT = REPORT_DIR / "differentiable_transport_full_fidelity_readiness.md"


def _profiles(rho: np.ndarray) -> np.ndarray:
    te = 8.0 * np.exp(-((rho - 0.35) ** 2) / 0.03) + 0.2
    ti = 6.0 * np.exp(-((rho - 0.42) ** 2) / 0.04) + 0.2
    ne = 4.0 + 0.8 * (1.0 - rho**2)
    nz = 0.03 + 0.02 * np.exp(-((rho - 0.65) ** 2) / 0.02)
    return np.stack([te, ti, ne, nz])


def main() -> None:
    """Run local audited transport-gradient latency benchmark and write reports."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not has_jax():
        payload = {
            "status": "blocked",
            "reason": "JAX is required for audited differentiable transport gradient-latency evidence",
            "claim_status": "no latency claim; JAX gradient backend unavailable in this environment",
            "schema_version": 1,
        }
        JSON_REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        ROLLOUT_JSON_REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        READINESS_JSON_REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        MD_REPORT.write_text(
            "\n".join(
                [
                    "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
                    "<!-- Commercial license available -->",
                    "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
                    "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
                    "<!-- ORCID: 0009-0009-3560-0851 -->",
                    "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
                    "<!-- SCPN Control — Differentiable transport latency benchmark report -->",
                    "",
                    "# Differentiable Transport Gradient-Latency Benchmark",
                    "",
                    "Status: `blocked`.",
                    "",
                    "JAX is required for audited differentiable transport gradient-latency",
                    "evidence. This environment does not provide the JAX gradient backend,",
                    "so no latency claim is made.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        ROLLOUT_MD_REPORT.write_text(
            "\n".join(
                [
                    "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
                    "<!-- Commercial license available -->",
                    "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
                    "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
                    "<!-- ORCID: 0009-0009-3560-0851 -->",
                    "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
                    "<!-- SCPN Control — Differentiable transport rollout latency benchmark report -->",
                    "",
                    "# Differentiable Transport Rollout Gradient-Latency Benchmark",
                    "",
                    "Status: `blocked`.",
                    "",
                    "JAX is required for audited differentiable transport rollout",
                    "source-gradient latency evidence. This environment does not",
                    "provide the JAX gradient backend, so no latency claim is made.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        READINESS_MD_REPORT.write_text(
            "\n".join(
                [
                    "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
                    "<!-- Commercial license available -->",
                    "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
                    "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
                    "<!-- ORCID: 0009-0009-3560-0851 -->",
                    "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
                    "<!-- SCPN Control — Differentiable transport full-fidelity readiness report -->",
                    "",
                    "# Differentiable Transport Full-Fidelity Readiness",
                    "",
                    "Status: `blocked`.",
                    "",
                    "JAX is required before readiness evidence can bind gradient",
                    "latency, rollout latency, campaign metadata, and external",
                    "reference admission artifacts.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return
    rho = np.linspace(0.05, 1.0, 21)
    profiles = _profiles(rho)
    chi = np.stack(
        [
            0.18 + 0.02 * rho,
            0.15 + 0.02 * rho,
            0.04 + 0.004 * rho,
            0.012 + 0.001 * rho,
        ]
    )
    sources = np.zeros_like(profiles)
    sources[0, 4:8] = 0.03
    sources[2, 7:11] = -0.01
    target = profiles.copy()
    target[0, 6:13] += 0.02
    target[1, 5:12] -= 0.015
    target[2, 4:10] += 0.01
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    report = benchmark_transport_parameter_gradient_latency(
        profiles,
        chi,
        sources,
        target,
        rho,
        8.0e-4,
        edge_values,
        weights=np.array([1.0, 0.75, 0.25, 0.1]),
        epsilon=5.0e-5,
        tolerance=2.0e-3,
        sample_indices=((0, 5), (1, 10), (2, 7), (3, 12)),
        warmup_runs=1,
        timed_runs=5,
    )
    payload = asdict(report)
    save_transport_gradient_latency_report(report, JSON_REPORT)
    source_sequence = np.repeat(sources[None, :, :], 4, axis=0)
    desired_sources = source_sequence.copy()
    desired_sources[:, 0, 5:10] += 0.01
    desired_sources[:, 1, 4:9] -= 0.006
    desired_sources[:, 2, 8:12] += 0.004
    target_history = np.asarray(
        differentiable_transport_rollout(
            profiles,
            chi,
            desired_sources,
            rho,
            8.0e-4,
            edge_values,
            use_jax=False,
        ),
        dtype=np.float64,
    )
    rollout_report = benchmark_transport_rollout_source_gradient_latency(
        profiles,
        chi,
        source_sequence,
        target_history,
        rho,
        8.0e-4,
        edge_values,
        weights=np.array([1.0, 0.75, 0.25, 0.1]),
        epsilon=5.0e-5,
        tolerance=2.0e-3,
        sample_indices=((0, 0, 5), (1, 1, 8), (2, 2, 10), (3, 3, 12)),
        warmup_runs=1,
        timed_runs=5,
    )
    rollout_payload = asdict(rollout_report)
    save_transport_rollout_gradient_latency_report(rollout_report, ROLLOUT_JSON_REPORT)
    equilibrium_psi = np.tile(np.linspace(0.0, 1.0, rho.size), (rho.size, 1))
    campaign_metadata = transport_campaign_metadata(
        profiles,
        chi,
        sources,
        rho,
        8.0e-4,
        edge_values,
        backend="jax",
        gradient_tolerance=2.0e-3,
        equilibrium_psi=equilibrium_psi,
    )
    readiness = transport_full_fidelity_readiness_evidence(
        campaign_metadata,
        report,
        rollout_report=rollout_report,
    )
    readiness_payload = asdict(readiness)
    READINESS_JSON_REPORT.write_text(
        json.dumps(readiness_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MD_REPORT.write_text(
        "\n".join(
            [
                "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
                "<!-- Commercial license available -->",
                "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- ORCID: 0009-0009-3560-0851 -->",
                "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
                "<!-- SCPN Control — Differentiable transport latency benchmark report -->",
                "",
                "# Differentiable Transport Gradient-Latency Benchmark",
                "",
                "This report measures the local audited gradient-admission path for",
                "controller-tuning studies. It is not a real-time control-loop guarantee.",
                "",
                f"- Backend: `{payload['backend']}`",
                f"- dtype: `{payload['dtype']}`",
                f"- Runtime platform: `{payload['runtime_metadata']['platform']}`",
                f"- Runtime machine: `{payload['runtime_metadata']['machine']}`",
                f"- JAX default backend: `{payload['runtime_metadata']['jax_default_backend']}`",
                f"- JAX devices: `{', '.join(payload['runtime_metadata']['jax_devices'])}`",
                f"- JAX x64 enabled: `{payload['runtime_metadata']['jax_enable_x64']}`",
                f"- Radial points: `{payload['n_rho']}`",
                f"- Timed runs: `{payload['timed_runs']}`",
                f"- Audit passed: `{payload['audit']['passed']}`",
                f"- P50 latency [ms]: `{payload['p50_ms']:.6f}`",
                f"- P95 latency [ms]: `{payload['p95_ms']:.6f}`",
                f"- Max latency [ms]: `{payload['max_ms']:.6f}`",
                f"- Claim boundary: `{payload['claim_status']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    ROLLOUT_MD_REPORT.write_text(
        "\n".join(
            [
                "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
                "<!-- Commercial license available -->",
                "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- ORCID: 0009-0009-3560-0851 -->",
                "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
                "<!-- SCPN Control — Differentiable transport rollout latency benchmark report -->",
                "",
                "# Differentiable Transport Rollout Gradient-Latency Benchmark",
                "",
                "This report measures the local audited multi-step source-rollout",
                "gradient-admission path for controller-tuning studies. It is not",
                "a real-time control-loop guarantee.",
                "",
                f"- Backend: `{rollout_payload['backend']}`",
                f"- dtype: `{rollout_payload['dtype']}`",
                f"- Runtime platform: `{rollout_payload['runtime_metadata']['platform']}`",
                f"- Runtime machine: `{rollout_payload['runtime_metadata']['machine']}`",
                f"- JAX default backend: `{rollout_payload['runtime_metadata']['jax_default_backend']}`",
                f"- JAX devices: `{', '.join(rollout_payload['runtime_metadata']['jax_devices'])}`",
                f"- JAX x64 enabled: `{rollout_payload['runtime_metadata']['jax_enable_x64']}`",
                f"- Radial points: `{rollout_payload['n_rho']}`",
                f"- Rollout steps: `{rollout_payload['n_steps']}`",
                f"- Timed runs: `{rollout_payload['timed_runs']}`",
                f"- Audit passed: `{rollout_payload['audit']['passed']}`",
                f"- P50 latency [ms]: `{rollout_payload['p50_ms']:.6f}`",
                f"- P95 latency [ms]: `{rollout_payload['p95_ms']:.6f}`",
                f"- Max latency [ms]: `{rollout_payload['max_ms']:.6f}`",
                f"- Claim boundary: `{rollout_payload['claim_status']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    READINESS_MD_REPORT.write_text(
        "\n".join(
            [
                "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
                "<!-- Commercial license available -->",
                "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- ORCID: 0009-0009-3560-0851 -->",
                "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
                "<!-- SCPN Control — Differentiable transport full-fidelity readiness report -->",
                "",
                "# Differentiable Transport Full-Fidelity Readiness",
                "",
                "This report binds local differentiable-transport latency evidence",
                "to campaign metadata and records why a full-fidelity claim is or",
                "is not admissible.",
                "",
                f"- Backend: `{readiness_payload['backend']}`",
                f"- Radial points: `{readiness_payload['n_rho']}`",
                f"- Equilibrium coupled: `{readiness_payload['equilibrium_coupled']}`",
                f"- Rollout steps: `{readiness_payload['rollout_steps']}`",
                f"- Full-fidelity admissible: `{readiness_payload['full_fidelity_claim_admissible']}`",
                f"- Blocked reasons: `{', '.join(readiness_payload['blocked_reasons'])}`",
                f"- Claim boundary: `{readiness_payload['claim_status']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
