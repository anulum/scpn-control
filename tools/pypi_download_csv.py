# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — PyPI CSV history and atomic publication.
"""Read, merge and atomically replace sparse PyPI download histories."""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from tools.pypi_download_protocol import CATEGORIES, CSV_HEADER, DownloadRows, _valid_count, _valid_date


class DownloadRowsError(ValueError):
    """Refuse a corrupt history or an invalid in-memory count mapping.

    Parameters
    ----------
    message : str
        Authored validation refusal. This remains ValueError-compatible.
    """

    __module__ = "tools.pypi_downloads"


def _validate_rows(rows: Mapping[str, Mapping[str, int]], label: str) -> None:
    """Validate an exact in-memory CSV row mapping before persistence."""
    for row_date, values in rows.items():
        if _valid_date(row_date) is None:
            raise DownloadRowsError(f"invalid date in {label}: {row_date!r}")
        if not values or not set(values).issubset(CATEGORIES):
            raise DownloadRowsError(f"invalid categories in {label}: {row_date}")
        if "with_mirrors" not in values:
            raise DownloadRowsError(f"missing with_mirrors in {label}: {row_date}")
        for category in values:
            if _valid_count(values.get(category)) is None:
                raise DownloadRowsError(f"invalid {category} count in {label}: {row_date}")
        without_mirrors = values.get("without_mirrors")
        if without_mirrors is not None and without_mirrors > values["with_mirrors"]:
            raise DownloadRowsError(f"invalid mirror-count ordering in {label}: {row_date}")


def read_csv(path: Path) -> DownloadRows:
    """Read a schema-fixed download history without modifying it.

    Parameters
    ----------
    path : pathlib.Path
        UTF-8 CSV with date, without_mirrors and with_mirrors columns.

    Returns
    -------
    DownloadRows
        Validated daily rows, or an empty mapping for an absent history. A blank
        without_mirrors field stays absent; with_mirrors is required.

    Raises
    ------
    DownloadRowsError
        Headers, fields, dates, duplicate days, integer text or ordering are bad.
    OSError, UnicodeError, csv.Error
        Native file access, decoding or CSV parsing fails.

    Notes
    -----
    Counts use canonical non-negative integer text, without leading signs or
    zeros. Existing history must validate before main can merge and publish.
    """
    if not path.exists():
        return {}
    rows: DownloadRows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_HEADER:
            raise DownloadRowsError("unexpected CSV header")
        for line_number, record in enumerate(reader, start=2):
            if set(record) != set(CSV_HEADER):
                raise DownloadRowsError(f"unexpected CSV fields at line {line_number}")
            row_date = _valid_date(record.get("date"))
            if row_date is None:
                raise DownloadRowsError(f"invalid date at line {line_number}")
            if row_date in rows:
                raise DownloadRowsError(f"duplicate date {row_date} at line {line_number}")
            values: dict[str, int] = {}
            for category in CATEGORIES:
                raw_count = record.get(category)
                if category == "without_mirrors" and raw_count == "":
                    continue
                try:
                    parsed_count: object = int(raw_count) if raw_count is not None else None
                except ValueError:
                    parsed_count = None
                count = _valid_count(parsed_count)
                if count is None or str(count) != raw_count:
                    raise DownloadRowsError(f"invalid {category} count at line {line_number}")
                values[category] = count
            rows[row_date] = values
    _validate_rows(rows, "CSV history")
    return rows


def merge_rows(existing: DownloadRows, fresh: DownloadRows) -> DownloadRows:
    """Replace overlapping days while retaining every other historical day.

    Parameters
    ----------
    existing : DownloadRows
        Historical daily category counts.
    fresh : DownloadRows
        New daily counts; each overlapping day replaces the entire old row.

    Returns
    -------
    DownloadRows
        Independent row dictionaries; neither input mapping is mutated.

    Raises
    ------
    DownloadRowsError
        Either mapping violates the date, count, category or mirror contract.
    """
    _validate_rows(existing, "existing rows")
    _validate_rows(fresh, "fresh rows")
    merged = {row_date: dict(values) for row_date, values in existing.items()}
    for row_date, values in fresh.items():
        merged[row_date] = dict(values)
    return merged


def write_csv(path: Path, rows: DownloadRows) -> None:
    """Publish date-sorted CSV bytes using one sibling atomic replacement.

    Parameters
    ----------
    path : pathlib.Path
        Destination CSV; missing parent directories are created.
    rows : DownloadRows
        Daily rows satisfying the sparse-category and mirror-count contract.

    Raises
    ------
    DownloadRowsError
        Rows are invalid; validation precedes file creation.
    OSError
        Staging, write, fsync, replacement or temporary cleanup fails.

    Notes
    -----
    The staged file is flushed and fsynced before replacement. Handled replace
    failure preserves existing destination bytes and removes its temporary file
    when possible. This is one-file publication, not a crash recovery protocol
    or protection against concurrent writers. A missing category serializes blank.
    """
    _validate_rows(rows, "output rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary_path = Path(handle.name)
            writer = csv.writer(handle)
            writer.writerow(CSV_HEADER)
            for row_date in sorted(rows):
                writer.writerow(
                    [
                        row_date,
                        *(rows[row_date].get(category, "") for category in CATEGORIES),
                    ]
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def summary(package: str, rows: DownloadRows) -> str:
    """Describe the latest declared day without changing either input.

    Parameters
    ----------
    package : str
        Display identity supplied by the caller.
    rows : DownloadRows
        Validated daily count rows.

    Returns
    -------
    str
        One-line day/count summary, using n/a for absent without_mirrors.

    Raises
    ------
    DownloadRowsError
        Input rows violate the daily count contract.

    Notes
    -----
    The text reports supplied declarations, not independently verified downloads.
    """
    _validate_rows(rows, "summary rows")
    if not rows:
        return f"{package}: no download data available yet"
    latest = max(rows)
    without_mirrors = rows[latest].get("without_mirrors")
    without_summary = "n/a" if without_mirrors is None else str(without_mirrors)
    return (
        f"{package}: {len(rows)} days recorded; latest {latest} "
        f"without_mirrors={without_summary} "
        f"with_mirrors={rows[latest]['with_mirrors']}"
    )
