# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native dependency metadata HTTP fixture.
"""Serve manufactured release declarations through a real loopback HTTP socket.

The payloads are protocol fixtures, not measured or authenticated PyPI evidence.
No production request or response decoder is replaced.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def release_metadata(
    name: str = "example-distribution", version: str = "1.0", constraint: str | None = ">=3.11"
) -> bytes:
    """Encode the exact release fields consumed by the production metadata API."""
    return json.dumps({"info": {"name": name, "version": version, "requires_python": constraint}}).encode()


@contextmanager
def metadata_service(
    payload: bytes | Mapping[str, bytes],
    *,
    status: int = 200,
    media_type: str = "application/json; charset=utf-8",
    delay_s: float = 0.0,
) -> Iterator[tuple[str, list[str]]]:
    """Own a finite HTTP server and return its origin plus observed request paths.

    Parameters
    ----------
    payload : bytes or mapping of str to bytes
        One body for every request, or exact request-path response bodies.
        Unmapped paths receive HTTP 404.
    status : int
        Response status for selected payloads.
    media_type : str
        Actual HTTP Content-Type header used by the decoder.
    delay_s : float
        Optional finite delay before headers, used to exercise socket timeout.

    Yields
    ------
    tuple of str and list of str
        Loopback origin/prefix and physical request paths. Shutdown joins the
        server and request threads; a timed-out reader may close its socket.
    """
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        """Deliver actual HTTP responses for the metadata transport contract."""

        def do_GET(self) -> None:
            """Record one real GET, then send the selected body or a missing path."""
            requests.append(self.path)
            selected = payload if isinstance(payload, bytes) else payload.get(self.path)
            if delay_s:
                time.sleep(delay_s)
            self.send_response(status if selected is not None else 404)
            self.send_header("Content-Type", media_type)
            body = selected or b""
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, format: str, *args: object) -> None:
            """Keep native server diagnostics out of captured checker verdicts."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/pypi", requests
    finally:
        server.shutdown()
        worker.join(timeout=2)
        server.server_close()
        assert not worker.is_alive()
