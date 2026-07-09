# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public inventory and coverage claim tests.
"""Regression tests for public inventory and coverage claims."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final, cast

ROOT: Final = Path(__file__).resolve().parents[1]
MANIFEST: Final = ROOT / "docs" / "_generated" / "capability_manifest.json"


def _manifest_counts() -> dict[str, int]:
    """Return generated capability inventory counts."""

    raw: Any = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest = cast(dict[str, Any], raw)
    return cast(dict[str, int], manifest["counts"])


def _coverage_gate() -> int:
    """Return the configured package coverage gate from ``pyproject.toml``."""

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"(?m)^fail_under = (?P<gate>\d+)$", pyproject)
    assert match is not None
    return int(match.group("gate"))


def _rust_snn_ci_latency() -> tuple[float, float]:
    """Return Rust SNN CI P50 and P95 latencies from the committed report."""

    raw: Any = json.loads((ROOT / "validation" / "reports" / "controller_latency.json").read_text(encoding="utf-8"))
    report = cast(dict[str, Any], raw)
    controllers = cast(list[dict[str, Any]], report["controllers"])
    for controller in controllers:
        if controller["name"] == "SNN" and controller["backend"] == "rust":
            stats = cast(dict[str, float], controller["stats"])
            return stats["p50_us"], stats["p95_us"]
    raise AssertionError("Rust SNN controller latency row is missing")


def _native_handoff_report() -> dict[str, Any]:
    """Return the committed native handoff comparison report."""

    raw: Any = json.loads(
        (ROOT / "validation" / "reports" / "native_handoff_comparison.json").read_text(encoding="utf-8")
    )
    return cast(dict[str, Any], raw)


def _normalized_prose(path: Path) -> str:
    """Return Markdown prose with line wrapping collapsed for claim checks."""

    return " ".join(path.read_text(encoding="utf-8").split())


def test_pitch_inventory_claim_matches_generated_manifest() -> None:
    """The pitch inventory summary must match generated capability counts."""

    counts = _manifest_counts()
    pitch = (ROOT / "docs" / "pitch.md").read_text(encoding="utf-8")

    assert f"{counts['source_module_count']} Python control/physics modules" in pitch
    assert f"{counts['rust_source_file_count']} Rust source files" in pitch
    assert f"{counts['python_test_file_count']} Python test files" in pitch
    assert f"{counts['workflow_count']} GitHub Actions workflows" in pitch


def test_readme_tree_test_count_matches_generated_manifest() -> None:
    """The README tree summary must match generated test-file inventory."""

    counts = _manifest_counts()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert f"tests/                 # {counts['python_test_file_count']} Python test files" in readme


def test_architecture_maps_are_bound_to_generated_inventory() -> None:
    """Architecture maps must identify the generated manifest as authoritative."""

    counts = _manifest_counts()
    checked_paths = (
        ROOT / "ARCHITECTURE.md",
        ROOT / "docs" / "architecture.md",
    )
    required_fragments = (
        "illustrative subset",
        "docs/_generated/capability_manifest.json",
        "tools/capability_manifest.py",
        "python tools/capability_manifest.py --check",
    )
    stale_fragments = (
        "3,300+ tests (235 files)",
        "154 Python modules (149 non-init)",
        "69 non-init modules",
        "51 non-init modules",
        "6 notebooks + 3 scripts",
        "99% coverage gate",
    )

    assert counts["source_module_count"] >= 1
    assert counts["python_test_file_count"] >= 1

    for path in checked_paths:
        text = _normalized_prose(path)
        for fragment in required_fragments:
            assert fragment in text, f"{path.relative_to(ROOT)} missing {fragment!r}"
        for fragment in stale_fragments:
            assert fragment not in text, f"{path.relative_to(ROOT)} contains {fragment!r}"


def test_architecture_native_latency_claim_cites_loopback_report() -> None:
    """Architecture native-latency wording must cite the loopback report."""

    report = _native_handoff_report()
    native_stats = cast(dict[str, float], cast(dict[str, Any], report["latency_stats"])["native"])
    parameters = cast(dict[str, Any], report["parameters"])
    comparison = cast(dict[str, Any], report["comparison"])
    architecture = _normalized_prose(ROOT / "docs" / "architecture.md")

    assert comparison["native_udp_sink_packets"] == parameters["steps"]
    assert parameters["transport_backend"] == "std"
    assert parameters["transport_endpoint"] == "127.0.0.1"

    required_fragments = (
        "validation/reports/native_handoff_comparison.json",
        f"{native_stats['p50_us']:.3f} µs P50",
        f"{native_stats['p95_us']:.3f} µs P95",
        f"{parameters['repeats']} repeats",
        f"{parameters['steps']:,} steps",
        "standard loopback-UDP transport",
        "`127.0.0.1`",
        f"port base `{parameters['transport_port_base']}`",
        "not fielded plant or PCS-cycle latency",
    )

    for fragment in required_fragments:
        assert fragment in architecture, f"docs/architecture.md missing {fragment!r}"

    assert "~5 µs P50 CI" not in architecture


def test_architecture_framing_uses_evidence_bound_language() -> None:
    """Architecture framing must describe evidence boundaries, not guarantees."""

    architecture = _normalized_prose(ROOT / "docs" / "architecture.md")
    required_fragments = (
        "separates Python orchestration from Rust/PyO3 hot paths",
        "module-level tests and claim-boundary metadata",
        "benchmark reports, runtime-admission records, and certificate evidence",
        "Architecture review workflow",
        "Use this map as a review checklist",
        "keep claim levels local unless a validator admits promotion",
        "without implying deployment readiness by itself",
    )
    stale_fragments = (
        "practical guarantee",
        "scientific experimentation stays ergonomic",
        "Enterprise onboarding",
        "timing guarantees",
        "maximize iteration speed",
        "practical control plane",
        "speed and governance",
    )

    for fragment in required_fragments:
        assert fragment in architecture, f"docs/architecture.md missing {fragment!r}"
    for fragment in stale_fragments:
        assert fragment not in architecture, f"docs/architecture.md contains {fragment!r}"


def test_public_coverage_gate_claims_match_configuration() -> None:
    """Current public coverage-gate claims must match ``pyproject.toml``."""

    gate = _coverage_gate()
    pitch = (ROOT / "docs" / "pitch.md").read_text(encoding="utf-8")
    paper = (ROOT / "paper.md").read_text(encoding="utf-8")

    assert f"{gate}% configured coverage gate" in pitch
    assert f"{gate}% package-coverage gate" in paper


def test_current_surfaces_do_not_use_stale_live_inventory_counts() -> None:
    """Live public inventory summaries must not repeat stale current counts."""

    checked_paths = (
        ROOT / "README.md",
        ROOT / "docs" / "pitch.md",
        ROOT / "paper.md",
    )
    stale_fragments = (
        "153 Python modules",
        "4,000+ collected Python tests",
        "20 CI jobs",
        "3,700+ collected Python tests",
        "170+ more test files",
        "99% coverage gate",
        "99% package-coverage gate",
    )

    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        for fragment in stale_fragments:
            assert fragment not in text, f"{path.relative_to(ROOT)} contains {fragment}"


def test_safe_rl_paper_claim_matches_implementation_surface() -> None:
    """Paper safe-RL wording must match the implemented controller surface."""

    expected = "safe RL (CPO-formulated Lagrangian constraints with control barrier functions, Ames 2017)"
    stale = "safe RL (PPO with MHD constraint veto)"
    checked_paths = (
        ROOT / "paper.md",
        ROOT / "docs" / "joss_paper.md",
    )

    for path in checked_paths:
        text = path.read_text(encoding="utf-8").replace("\n  ", " ")
        assert expected in text, f"{path.relative_to(ROOT)} missing safe-RL implementation wording"
        assert stale not in text, f"{path.relative_to(ROOT)} contains stale safe-RL veto wording"


def test_neural_equilibrium_latency_claim_is_not_uncited() -> None:
    """Public neural-equilibrium docs must not repeat the uncited 0.39 ms claim."""

    checked_paths = (
        ROOT / "docs" / "architecture.md",
        ROOT / "docs" / "pitch.md",
    )

    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        assert "0.39 ms" not in text
        assert "0.39ms" not in text

    pitch = (ROOT / "docs" / "pitch.md").read_text(encoding="utf-8")
    assert "validation/reports/neural_equilibrium_pretraining.json" in pitch
    assert "real EFIT/P-EFIT" in pitch
    assert "latency or accuracy claims remain blocked" in pitch


def test_faq_rust_snn_latency_claim_matches_report() -> None:
    """The FAQ Rust SNN latency claim must match the committed CI report."""

    p50_us, p95_us = _rust_snn_ci_latency()
    faq = (ROOT / "docs" / "faq.md").read_text(encoding="utf-8")

    assert f"{p50_us:.3f} µs P50" in faq
    assert f"{p95_us:.3f} µs P95" in faq
    assert "0.92 µs (P50, CI)" not in faq


def test_faq_controller_registry_guidance_matches_code() -> None:
    """The FAQ controller-extension guidance must name the real registry."""

    faq = (ROOT / "docs" / "faq.md").read_text(encoding="utf-8")
    comparison = (ROOT / "validation" / "controller_comparison.py").read_text(encoding="utf-8")
    flight_sim = (ROOT / "src" / "scpn_control" / "control" / "tokamak_flight_sim.py").read_text(encoding="utf-8")

    assert "CONTROLLERS = {" in comparison
    assert "CONTROLLERS =" not in flight_sim
    assert "validation/controller_comparison.py" in faq
    assert "`tokamak_flight_sim.py` provides the" in faq
    assert "within `tokamak_flight_sim.py` or" not in faq


def test_public_physics_monitor_claims_keep_open_issue_caveats() -> None:
    """EM and Lyapunov public claims must carry the live research boundary."""

    checked_paths = (
        ROOT / "ROADMAP.md",
        ROOT / "paper.md",
        ROOT / "docs" / "joss_paper.md",
    )
    required_fragments = (
        "research-prototype",
        "warm-up and consecutive-window fail-open",
        "not a standalone fail-closed safety interlock",
        "external-code revalidation",
    )

    for path in checked_paths:
        text = _normalized_prose(path)
        for fragment in required_fragments:
            assert fragment in text, f"{path.relative_to(ROOT)} missing {fragment!r}"


def test_joss_qlknn_claim_matches_neural_transport_admission_report() -> None:
    """JOSS neural-transport wording must match the public claim report."""

    raw: Any = json.loads(
        (ROOT / "validation" / "reports" / "neural_transport_claims.json").read_text(encoding="utf-8")
    )
    report = cast(dict[str, Any], raw)
    evidence = cast(dict[str, Any], report["claim_evidence"])

    assert evidence["surrogate_mode"] == "analytic_fallback"
    assert evidence["quantitative_claim_allowed"] is False

    checked_paths = (
        ROOT / "paper.md",
        ROOT / "docs" / "joss_paper.md",
    )
    required_fragments = (
        "QLKNN-style neural-transport facade",
        "analytic critical-gradient fallback",
        "current public neural-transport claim report is `analytic_fallback`",
        "quantitative QLKNN/QuaLiKiz claims remain blocked unless trained weights",
    )
    stale_fragments = (
        "QLKNN surrogates ($\\sim$24 ns)",
        "QLKNN surrogate [@plassche2020]",
    )

    for path in checked_paths:
        text = _normalized_prose(path)
        for fragment in required_fragments:
            assert fragment in text, f"{path.relative_to(ROOT)} missing {fragment!r}"
        for fragment in stale_fragments:
            assert fragment not in text, f"{path.relative_to(ROOT)} contains {fragment!r}"

    source = (ROOT / "src" / "scpn_control" / "core" / "neural_transport.py").read_text(encoding="utf-8")
    assert "Quantitative QLKNN or QuaLiKiz claims are" in source
    assert "reproduces gyrokinetic-level predictions" not in source


def test_joss_mu_synthesis_claim_matches_static_bound_report() -> None:
    """JOSS mu-synthesis wording must match the bounded static report."""

    raw: Any = json.loads((ROOT / "validation" / "reports" / "mu_synthesis_claims.json").read_text(encoding="utf-8"))
    report = cast(dict[str, Any], raw)

    assert report["claim_status"] == "bounded_static_mu_evidence"
    assert report["static_dc_analysis_only"] is True
    assert report["validated_claim_allowed"] is False

    checked_paths = (
        ROOT / "paper.md",
        ROOT / "docs" / "joss_paper.md",
    )
    required_fragments = (
        "D-scaling upper bound",
        "bounded static $\\mu$-analysis",
        "not validated full DK synthesis",
    )
    stale_fragments = (
        "$\\mu$-synthesis (D-K iteration)",
        "$\\mu$-synthesis (DK-iteration",
    )

    for path in checked_paths:
        text = _normalized_prose(path)
        for fragment in required_fragments:
            assert fragment in text, f"{path.relative_to(ROOT)} missing {fragment!r}"
        for fragment in stale_fragments:
            assert fragment not in text, f"{path.relative_to(ROOT)} contains {fragment!r}"
