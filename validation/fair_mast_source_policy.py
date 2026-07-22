# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FAIR-MAST Source Policy
"""Authoritative licence and citation policy for FAIR-MAST artefacts.

The MAST Data Catalog states that, except where noted, archive data, metadata,
and site content use Creative Commons Attribution-ShareAlike 4.0 International.
It also asks scientific users to cite the 2025 IEEE TPS service paper and the
2024 SoftwareX FAIR-MAST paper. Keeping those facts here prevents acquisition,
campaign, and dataset emitters from drifting independently.
"""

from __future__ import annotations

FAIR_MAST_CATALOG_URL = "https://mastapp.site/"
FAIR_MAST_LICENCE = "CC-BY-SA-4.0"
FAIR_MAST_LICENCE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
FAIR_MAST_CITATIONS: tuple[str, ...] = (
    "Jackson et al., An Open Data Service for Supporting Research in Machine Learning "
    "on Tokamak Data, IEEE Transactions on Plasma Science (2025), "
    "DOI 10.1109/TPS.2025.3583419",
    "Jackson et al., FAIR-MAST: A fusion device data management system, "
    "SoftwareX 27 (2024) 101869, DOI 10.1016/j.softx.2024.101869",
)
FAIR_MAST_CITATION = "; ".join(FAIR_MAST_CITATIONS)


def fair_mast_provenance() -> dict[str, str | list[str]]:
    """Return a fresh JSON-ready FAIR-MAST provenance block."""
    return {
        "licence": FAIR_MAST_LICENCE,
        "licence_url": FAIR_MAST_LICENCE_URL,
        "citation": FAIR_MAST_CITATION,
        "citations": list(FAIR_MAST_CITATIONS),
        "source_policy_url": FAIR_MAST_CATALOG_URL,
    }
