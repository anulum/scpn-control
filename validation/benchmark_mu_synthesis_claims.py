#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Deprecated benchmark entrypoint compatibility wrapper.
"""Forward the historical benchmark entrypoint to its truthful replacement."""

from __future__ import annotations

import sys

from scpn_control.benchmark_records import require_recorded_campaign
from validation.benchmark_static_mu_analysis_claims import (
    JSON_REPORT,
    MARKDOWN_REPORT,
    REPORT_DIR,
    main as _canonical_main,
)


def main() -> None:
    """Run the static mu-analysis benchmark with a visible deprecation notice."""
    require_recorded_campaign(JSON_REPORT, MARKDOWN_REPORT, repository_root=REPORT_DIR.parents[1])
    print(
        "DEPRECATED: benchmark_mu_synthesis_claims.py will be removed in 0.25.0; "
        "use benchmark_static_mu_analysis_claims.py.",
        file=sys.stderr,
    )
    _canonical_main()


if __name__ == "__main__":
    main()
