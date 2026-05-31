# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Digital twin online update benchmark
"""Publish deterministic bounded online-update evidence for the digital twin."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from scpn_control.control.digital_twin_online_update import synthetic_online_update_benchmark

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "digital_twin_online_update.json"
MD_REPORT = REPORT_DIR / "digital_twin_online_update.md"


def main() -> None:
    """Run the synthetic online-update benchmark and write reports."""
    result = synthetic_online_update_benchmark(seed=20240531)
    payload = asdict(result)
    payload["claim_boundary"] = (
        "Bounded synthetic online-update benchmark only. TRANSP/TSC coupling "
        "requires validated external simulator artifacts before measured replay claims."
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MD_REPORT.write_text(
        "\n".join(
            [
                "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
                "<!-- Commercial license available -->",
                "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
                "<!-- ORCID: 0009-0009-3560-0851 -->",
                "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
                "<!-- SCPN Control — Digital twin online update report -->",
                "",
                "# Digital twin online update benchmark",
                "",
                "- Evidence kind: bounded synthetic online update",
                f"- Source: {result.source}",
                f"- Evaluated points: {result.evaluated_points}",
                f"- Baseline loss: {result.baseline_loss:.8e}",
                f"- Best loss: {result.best_loss:.8e}",
                "",
                "Best parameters:",
                "",
                *[f"- `{name}`: {value:.8e}" for name, value in sorted(result.best_parameters.items())],
                "",
                "Claim boundary: this benchmark exercises deterministic Bayesian",
                "model updating against a synthetic digital-twin reference. External",
                "TRANSP/TSC coupling is fail-closed behind validated simulator",
                "artifact metadata and the digital-twin reference gate.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
