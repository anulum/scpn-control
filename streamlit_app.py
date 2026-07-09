# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Streamlit Cloud entry point.
"""Root Streamlit Cloud entry point for the live phase dashboard.

Streamlit Cloud expects a repository-root script. The production dashboard logic
already lives in ``examples/streamlit_ws_client.py`` so local users and the cloud
deployment exercise the same WebSocket client code path. With no explicit
arguments this adapter starts that client in embedded-server mode.
"""

from __future__ import annotations

import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent / "examples" / "streamlit_ws_client.py"
DEFAULT_ARGS = ("--embedded",)


def streamlit_cloud_args(argv: Sequence[str]) -> list[str]:
    """Return arguments for the delegated Streamlit dashboard.

    Parameters
    ----------
    argv
        Arguments passed after ``streamlit run streamlit_app.py --``.

    Returns
    -------
    list[str]
        Explicit arguments when supplied, otherwise the embedded-server default
        required by Streamlit Cloud.
    """

    return list(argv) if argv else list(DEFAULT_ARGS)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the existing Streamlit WebSocket client through the root entry point.

    Parameters
    ----------
    argv
        Optional dashboard arguments. When omitted, command-line arguments after
        the root script are forwarded.
    """

    app_args = streamlit_cloud_args(sys.argv[1:] if argv is None else argv)
    previous_argv = sys.argv[:]
    try:
        sys.argv = [str(APP_PATH), *app_args]
        runpy.run_path(str(APP_PATH), run_name="__main__")
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
