# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — atomic PyPI download-series snapshot.
"""Record a validated sparse PyPI download series with bounded retry attempts."""

from __future__ import annotations

import argparse
import csv
import http as http
import sys
import time
import tomllib
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.pypi_download_csv import (
    DownloadRowsError as DownloadRowsError,
)
from tools.pypi_download_csv import (
    merge_rows as merge_rows,
)
from tools.pypi_download_csv import (
    read_csv as read_csv,
)
from tools.pypi_download_csv import (
    summary as summary,
)
from tools.pypi_download_csv import (
    write_csv as write_csv,
)
from tools.pypi_download_protocol import (
    CATEGORIES as CATEGORIES,
)
from tools.pypi_download_protocol import (
    CSV_HEADER as CSV_HEADER,
)
from tools.pypi_download_protocol import (
    MAX_RESPONSE_BYTES as MAX_RESPONSE_BYTES,
)
from tools.pypi_download_protocol import (
    MAX_TRANSIENT_WINDOW_SECONDS as MAX_TRANSIENT_WINDOW_SECONDS,
)
from tools.pypi_download_protocol import (
    PAYLOAD_KEYS as PAYLOAD_KEYS,
)
from tools.pypi_download_protocol import (
    PROJECT_PACKAGE as PROJECT_PACKAGE,
)
from tools.pypi_download_protocol import (
    PYPISTATS_HOST as PYPISTATS_HOST,
)
from tools.pypi_download_protocol import (
    PYPISTATS_PATH as PYPISTATS_PATH,
)
from tools.pypi_download_protocol import (
    PYPISTATS_TYPE as PYPISTATS_TYPE,
)
from tools.pypi_download_protocol import (
    REQUEST_TIMEOUT_SECONDS as REQUEST_TIMEOUT_SECONDS,
)
from tools.pypi_download_protocol import (
    RETRY_DELAYS as RETRY_DELAYS,
)
from tools.pypi_download_protocol import (
    RETRYABLE_STATUSES as RETRYABLE_STATUSES,
)
from tools.pypi_download_protocol import (
    ROW_KEYS as ROW_KEYS,
)
from tools.pypi_download_protocol import (
    DownloadConfigurationError as DownloadConfigurationError,
)
from tools.pypi_download_protocol import (
    DownloadRows as DownloadRows,
)
from tools.pypi_download_protocol import (
    DownloadSnapshotError as DownloadSnapshotError,
)
from tools.pypi_download_protocol import (
    DuplicateJSONObjectKeyError as DuplicateJSONObjectKeyError,
)
from tools.pypi_download_protocol import (
    Fetch as Fetch,
)
from tools.pypi_download_protocol import (
    RetryableSnapshotError as RetryableSnapshotError,
)
from tools.pypi_download_protocol import (
    Sleep as Sleep,
)
from tools.pypi_download_protocol import (
    _http_get as _http_get,
)
from tools.pypi_download_protocol import (
    _valid_count as _valid_count,
)
from tools.pypi_download_protocol import (
    _valid_date as _valid_date,
)
from tools.pypi_download_protocol import (
    fetch_overall as fetch_overall,
)
from tools.pypi_download_protocol import (
    fetch_overall_with_retry as fetch_overall_with_retry,
)
from tools.pypi_download_protocol import (
    package_endpoint_path as package_endpoint_path,
)
from tools.pypi_download_protocol import (
    validate_overall as validate_overall,
)

PROJECT_CSV = Path("downloads/scpn-control.csv")


def detect_package(pyproject_path: Path) -> str:
    """Read the stripped distribution name from PEP621 project metadata.

    Parameters
    ----------
    pyproject_path : pathlib.Path
        UTF-8 TOML file containing a project table and nonblank name.

    Returns
    -------
    str
        Declared distribution name with outer whitespace removed.

    Raises
    ------
    DownloadConfigurationError
        The project table or distribution name is absent or malformed.
    OSError, UnicodeError, tomllib.TOMLDecodeError
        Native input access, decoding or TOML parsing fails.
    """
    document = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = document.get("project")
    if not isinstance(project, dict):
        raise DownloadConfigurationError("no [project] table")
    raw_name = project.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise DownloadConfigurationError("no [project] name")
    return raw_name.strip()


def require_project_csv_target(package: str, csv_path: Path) -> None:
    """Require the exact distribution and lexical relative metrics path.

    Parameters
    ----------
    package : str
        Must equal scpn-control.
    csv_path : pathlib.Path
        Must be the relative downloads/scpn-control.csv path.

    Raises
    ------
    DownloadConfigurationError
        Package or path differs. No file or network access occurs here.

    Notes
    -----
    This is the maintained project-only writer contract, not filesystem sandboxing
    or authentication of metrics. Cooperating writers own directory custody.
    """
    if package != PROJECT_PACKAGE:
        raise DownloadConfigurationError(f"project metrics package must be {PROJECT_PACKAGE!r}")
    if csv_path.is_absolute() or csv_path.as_posix() != PROJECT_CSV.as_posix():
        raise DownloadConfigurationError(f"project metrics CSV must be {PROJECT_CSV.as_posix()!r}")


def main(
    argv: list[str] | None = None,
    fetch: Fetch = _http_get,
    sleep: Sleep = time.sleep,
) -> int:
    """Resolve, fetch, validate and atomically publish one CSV snapshot.

    Parameters
    ----------
    argv : list of str or None, optional
        CLI options, or process arguments when omitted. --print-package reads
        metadata without fetching or writing. Snapshot mode requires --csv.
    fetch : callable, optional
        Package-to-bytes transport; defaults to the fixed HTTPS host.
    sleep : callable, optional
        Retry delay callback receiving seconds.

    Returns
    -------
    int
        Zero after successful publication, metadata inspection or exhausted
        transient soft skip; one after authored refusal or caught native failure.

    Raises
    ------
    SystemExit
        Help or invalid CLI options request an argument-parser exit.

    Notes
    -----
    Authored domain errors are displayed; native I/O, TOML, UTF8, JSON and CSV
    errors receive a fixed sentence. A skipped fetch does not touch history.
    The scheduled writer's ten-minute job limit is external to this function.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", default="pyproject.toml")
    parser.add_argument("--package")
    parser.add_argument("--csv")
    parser.add_argument("--project-csv-only", action="store_true")
    parser.add_argument("--print-package", action="store_true")
    arguments = parser.parse_args(argv)

    try:
        package = arguments.package or detect_package(Path(arguments.pyproject))
        if arguments.print_package:
            print(package)
            return 0
        if not arguments.csv:
            parser.error("--csv is required unless --print-package is used")
        csv_path = Path(arguments.csv)
        if arguments.project_csv_only:
            require_project_csv_target(package, csv_path)
        fresh = fetch_overall_with_retry(package, fetch, sleep)
        if fresh is None:
            print(
                "snapshot skipped: pypistats remained unavailable after bounded retries; "
                "the next successful rolling-window fetch will backfill it",
                file=sys.stderr,
            )
            return 0
        rows = merge_rows(read_csv(csv_path), fresh)
        write_csv(csv_path, rows)
    except (DownloadSnapshotError, DownloadRowsError, DownloadConfigurationError) as exc:
        print(f"snapshot failed: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, csv.Error):
        print("snapshot failed: inputs, response or output could not be processed", file=sys.stderr)
        return 1
    print(summary(package, rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
