# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Shared Click report destination validation

"""Click report destination conversion: keep path usage checks and refuse NUL before filesystem calls."""

from __future__ import annotations

from os import PathLike, fspath
from typing import Any

import click


class _ReportOutputPath(click.Path):
    """Preserve Click path usage checks while refusing NUL before filesystem conversion."""

    def convert(self, value: str | PathLike[str], param: click.Parameter | None, ctx: click.Context | None) -> Any:
        """Return the original Click conversion; malformed NUL paths receive fixed usage text."""
        if "\0" in fspath(value):
            self.fail("report output path must not contain NUL", param, ctx)
        return super().convert(value, param, ctx)
