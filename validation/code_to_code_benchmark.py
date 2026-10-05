# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Code-to-code diagnostic command and compatible aliases.

"""Run bounded CONTROL transport diagnostics and optional TORAX comparisons.

Initial axis-to-edge profiles and fixed dt are mapped from one declared
scenario. The configured equilibrium/current, source and transport closures
remain different; a finite comparison cannot admit a physical reference.
Use explicit temporary outputs for local observations. Persistent repository
evidence roots require recorded-runner custody.

Usage: python validation/code_to_code_benchmark.py --with-torax
       --json-out /tmp/comparison.json --markdown-out /tmp/comparison.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_repo_src_on_path() -> None:
    """Add the defining repository and src to support direct source execution."""
    root = Path(__file__).resolve().parents[1]
    for path in (str(root), str(root / "src")):
        if path not in sys.path:
            sys.path.insert(0, path)


_ensure_repo_src_on_path()

from validation.code_to_code_comparison import (
    _benchmark_numeric_payload_is_finite as _benchmark_numeric_payload_is_finite,
)
from validation.code_to_code_comparison import (
    _canonical_json as _canonical_json,
)
from validation.code_to_code_comparison import (
    _compare_results as _compare_results,
)
from validation.code_to_code_comparison import (
    _finite_number as _finite_number,
)
from validation.code_to_code_comparison import (
    _finite_vector as _finite_vector,
)
from validation.code_to_code_comparison import (
    _payload_without_digest as _payload_without_digest,
)
from validation.code_to_code_comparison import (
    _sha256_payload as _sha256_payload,
)
from validation.code_to_code_comparison import (
    _verify_payload_digest as _verify_payload_digest,
)
from validation.code_to_code_comparison import (
    compare_transport_profiles as compare_transport_profiles,
)
from validation.code_to_code_comparison import (
    verify_payload_digest as verify_payload_digest,
)
from validation.code_to_code_local import (
    _run_scpn_control as _run_scpn_control,
)
from validation.code_to_code_local import (
    run_local_transport as run_local_transport,
)
from validation.code_to_code_reports import (
    MARKDOWN_REPORT_PATH as MARKDOWN_REPORT_PATH,
)
from validation.code_to_code_reports import (
    REPORT_PATH as REPORT_PATH,
)
from validation.code_to_code_reports import (
    REPORT_SCHEMA_VERSION as REPORT_SCHEMA_VERSION,
)
from validation.code_to_code_reports import (
    _build_external_reference_report as _build_external_reference_report,
)
from validation.code_to_code_reports import (
    _external_reference_status as _external_reference_status,
)
from validation.code_to_code_reports import (
    _write_markdown_report as _write_markdown_report,
)
from validation.code_to_code_reports import (
    build_comparison_report as build_comparison_report,
)
from validation.code_to_code_scenario import (
    ITER_SCENARIO as ITER_SCENARIO,
)
from validation.code_to_code_scenario import (
    _torax_config_dict as _torax_config_dict,
)
from validation.code_to_code_torax import (
    TORAX_TMP_CONFIG as TORAX_TMP_CONFIG,
)
from validation.code_to_code_torax import (
    _extract_torax_result as _extract_torax_result,
)
from validation.code_to_code_torax import (
    _run_torax as _run_torax,
)
from validation.code_to_code_torax import (
    _write_torax_config as _write_torax_config,
)
from validation.code_to_code_torax import (
    write_torax_config as write_torax_config,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse provider/strict-status flags and caller-relative output paths.

    argparse accepts optional argv (None uses sys.argv) and exits zero on help
    or two on malformed flags. No scenario is supplied by the CLI; the fixed
    ITER_SCENARIO is used. Paths are guarded by main before any computation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-torax",
        action="store_true",
        help="Run TORAX for diagnostics; configured physics differences remain explicit.",
    )
    parser.add_argument(
        "--require-external",
        action="store_true",
        help="Exit one while physical external-reference admission remains blocked.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPORT_PATH,
        help="Path for the schema-versioned JSON evidence report.",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=MARKDOWN_REPORT_PATH,
        help="Path for the Markdown evidence summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute the fixed scenario and write a declared diagnostic JSON/Markdown pair.

    Parameters
    ----------
    argv : list[str] or None
        Flags parsed by argparse: optional TORAX, required external admission
        and JSON/Markdown paths. None reads the process command line.

    Returns
    -------
    int
        Zero after output for ordinary diagnostics; one after output when
        --require-external is requested and physical admission is blocked.

    Raises
    ------
    SystemExit
        argparse help (zero) or malformed arguments (two).
    ValueError
        Output paths alias each other or a protected input, or custody/domain
        validation refuses the request before computation.
    OSError, Exception
        Output or actual solver/provider errors propagate. No substitute
        simulation, partial success or invented metrics are supplied.

    Notes
    -----
    Output parents are created and JSON/Markdown are written sequentially as
    UTF-8, with overwrites and no pairwise transaction. Provider absence yields
    an explicit blocked report. JSON uses a self-digest for consistency only.
    The core numerical implementation has no counterpart changed by this
    adapter; wall time is an observation, not a controlled comparison benchmark.
    """
    from scpn_control.benchmark_records import require_recorded_campaign
    from validation.report_output_paths import checked_report_destination

    args = _parse_args(argv)
    sources = tuple(Path(__file__).resolve().parent.glob("code_to_code_*.py"))
    checked_report_destination(args.json_out, inputs=(args.markdown_out, TORAX_TMP_CONFIG, *sources))
    checked_report_destination(args.markdown_out, inputs=(args.json_out, TORAX_TMP_CONFIG, *sources))
    require_recorded_campaign(args.json_out, args.markdown_out, repository_root=Path(__file__).resolve().parents[1])
    with_torax = bool(args.with_torax)

    print(f"Running code-to-code benchmark: {ITER_SCENARIO['name']}")
    print("=" * 60)

    print("\n[1/3] Running scpn-control...")
    scpn_result = _run_scpn_control(ITER_SCENARIO)
    print(f"  Te_avg = {scpn_result['Te_avg']:.3f} keV")
    print(f"  Ti_avg = {scpn_result['Ti_avg']:.3f} keV")
    print(f"  Energy balance error = {scpn_result['energy_balance_error']:.4e}")
    print(f"  Particle balance error = {scpn_result['particle_balance_error']:.4e}")
    print(f"  Wall time = {scpn_result['wall_time_s']:.2f} s")

    torax_result = None
    if with_torax:
        print("\n[2/3] Running TORAX...")
        torax_result = _run_torax(ITER_SCENARIO)
        if torax_result:
            print(f"  Status: {torax_result.get('status', 'done')}")
    else:
        print("\n[2/3] TORAX skipped (use --with-torax to enable)")

    print("\n[3/3] Comparing results...")
    comparison = _compare_results(scpn_result, torax_result)
    report = _build_external_reference_report(
        comparison,
        ITER_SCENARIO,
        requested_torax=with_torax,
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    _write_markdown_report(report, args.markdown_out)

    print(f"\nResults saved to {args.json_out}")
    print(f"Summary saved to {args.markdown_out}")
    print(f"External reference status: {report['external_reference']['status']}")

    if comparison["comparison"]:
        print("\nComparison metrics:")
        for k, v in comparison["comparison"].items():
            print(f"  {k}: {v}")

    if args.require_external and not report["external_reference"]["admitted"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
