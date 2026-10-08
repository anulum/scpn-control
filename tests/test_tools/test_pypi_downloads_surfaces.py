# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — PyPI snapshot CLI and native socket probes.
"""Exercise snapshot failures through the public CLI and real local HTTP I/O."""

from __future__ import annotations

import http.client
import json
import subprocess
import sys
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import TypedDict

import pytest

from tools import pypi_downloads as downloads

ROOT = Path(__file__).resolve().parents[2]
SAFE_FAILURE = "snapshot failed: inputs, response or output could not be processed\n"
PAYLOAD = json.dumps(
    {
        "data": [{"category": "with_mirrors", "date": "2026-07-17", "downloads": 19}],
        "package": "scpn-control",
        "type": "overall_downloads",
    }
).encode()
HISTORY = b"date,without_mirrors,with_mirrors\r\n2026-07-16,7,9\r\n"


class LocalHTTPState(TypedDict):
    """Hold the exact native HTTP fixture response and observed request fields."""

    invalid_status: bool
    status: int
    content_type: str
    body: bytes
    requests: list[tuple[str, str | None, str | None]]


@pytest.fixture
def local_http(monkeypatch: pytest.MonkeyPatch) -> Iterator[LocalHTTPState]:
    """Route the fixed-host constructor to native local HTTP, without claiming TLS."""
    state: LocalHTTPState = {
        "invalid_status": False,
        "status": 200,
        "content_type": "application/json; charset=utf-8",
        "body": PAYLOAD,
        "requests": [],
    }

    class Handler(BaseHTTPRequestHandler):
        """Serve explicitly synthetic fixture declarations over an actual socket."""

        def do_GET(self) -> None:
            """Record request paths and return the selected response bytes."""
            requests = state["requests"]
            requests.append((self.path, self.headers.get("Accept"), self.headers.get("User-Agent")))
            status, body, content_type = state["status"], state["body"], state["content_type"]
            if state["invalid_status"]:
                self.wfile.write(b"PRIVATE_INVALID_STATUS_CANARY\r\n")
                return
            self.send_response(status, "PRIVATE_REMOTE_REASON_CANARY")
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format: str, *args: object) -> None:
            """Keep fixture access logs out of caller-output assertions."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()

    def connect(host: str, *, timeout: int) -> http.client.HTTPConnection:
        assert host == "pypistats.org" and timeout == 30
        return http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=timeout)

    monkeypatch.setattr(http.client, "HTTPSConnection", connect)
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
        assert not worker.is_alive()


@pytest.mark.parametrize("case", ["missing", "toml", "utf8", "directory"])
def test_executable_metadata_native_errors_are_fixed(tmp_path: Path, case: str) -> None:
    """Real executable metadata failures must hide native text and preserve inputs."""
    source = tmp_path / "PRIVATE_METADATA_PATH_CANARY.toml"
    if case == "toml":
        source.write_bytes(b'[project\nname = "scpn-control"\n')
    elif case == "utf8":
        source.write_bytes(b"\xff")
    elif case == "directory":
        source.mkdir()
    before = source.read_bytes() if source.is_file() else None
    result = subprocess.run(
        [sys.executable, "-S", str(ROOT / "tools/pypi_downloads.py"), "--pyproject", str(source), "--print-package"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 1 and result.stdout == "" and result.stderr == SAFE_FAILURE
    assert (source.read_bytes() if source.is_file() else None) == before


def test_native_socket_snapshot_preserves_sparse_history(
    local_http: LocalHTTPState, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A native response reaches the public publisher without inventing absent zeros."""
    target = tmp_path / "history.csv"
    target.write_bytes(HISTORY)
    assert downloads.main(["--package", "scpn-control", "--csv", str(target)]) == 0
    assert target.read_bytes() == HISTORY + b"2026-07-17,,19\r\n"
    assert capsys.readouterr().err == ""
    assert local_http["requests"] == [
        ("/api/packages/scpn-control/overall", "application/json", "scpn-control-metrics/1")
    ]


@pytest.mark.parametrize(
    ("status", "content_type", "body", "expected"),
    [
        (404, "application/json", PAYLOAD, "pypistats returned HTTP 404"),
        (200, "application/json-private-canary", PAYLOAD, "pypistats returned unexpected Content-Type"),
        (200, "text/PRIVATE_TYPE_CANARY", PAYLOAD, "pypistats returned unexpected Content-Type"),
        (200, "application/json", b"PRIVATE_INVALID_JSON_CANARY", "pypistats returned invalid JSON"),
        (200, "application/json", b"\xff", "pypistats returned invalid JSON"),
        (200, "application/json", b"x" * 5_000_001, "pypistats response exceeded the size limit"),
        (
            200,
            "application/json",
            b'{"PRIVATE_DUPLICATE_CANARY":1,"PRIVATE_DUPLICATE_CANARY":2}',
            "pypistats returned invalid JSON: duplicate JSON object name",
        ),
    ],
    ids=["status", "media-prefix", "media-other", "json", "utf8", "size-bound", "duplicate-member"],
)
def test_native_socket_refusals_preserve_csv_and_hide_external_text(
    local_http: LocalHTTPState,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    status: int,
    content_type: str,
    body: bytes,
    expected: str,
) -> None:
    """Real response failures cannot rewrite history or launder remote text."""
    local_http["status"] = status
    local_http["content_type"] = content_type
    local_http["body"] = body
    target = tmp_path / "history.csv"
    target.write_bytes(HISTORY)
    assert downloads.main(["--package", "scpn-control", "--csv", str(target)]) == 1
    output = capsys.readouterr()
    assert output.out == "" and output.err == f"snapshot failed: {expected}\n"
    assert "PRIVATE_" not in output.err and target.read_bytes() == HISTORY
    assert len(list(target.parent.iterdir())) == 1


def test_native_socket_transient_soft_skip_preserves_history(
    local_http: LocalHTTPState, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Four real transient responses soft-skip without modifying history."""
    local_http["status"] = 503
    target = tmp_path / "history.csv"
    target.write_bytes(HISTORY)
    delays: list[float] = []
    assert downloads.main(["--package", "scpn-control", "--csv", str(target)], sleep=delays.append) == 0
    assert delays == [15.0, 30.0, 60.0] and len(local_http["requests"]) == 4
    assert target.read_bytes() == HISTORY and "snapshot skipped" in capsys.readouterr().err


@pytest.mark.parametrize("case", ["utf8", "field-size", "directory", "blocked-parent"])
def test_native_csv_failures_are_fixed_and_keep_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """Public snapshot processing must contain actual filesystem and CSV parser faults."""
    target = tmp_path / "PRIVATE_CSV_PATH_CANARY"
    if case == "utf8":
        target.write_bytes(b"\xff")
    elif case == "field-size":
        target.write_bytes(b"date,without_mirrors,with_mirrors\n2026-07-16," + b"1" * 131_073 + b",9\n")
    elif case == "directory":
        target.mkdir()
    else:
        target.write_bytes(b"PRIVATE_PARENT_CANARY")
        target = target / "history.csv"
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert downloads.main(["--package", "scpn-control", "--csv", str(target)], fetch=lambda package: PAYLOAD) == 1
    output = capsys.readouterr()
    assert output.out == "" and output.err == SAFE_FAILURE
    assert {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("depth", [1100, 100_000])
def test_deep_json_public_main_refuses_without_changing_history(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], depth: int
) -> None:
    """Native JSON depth limits and invalid schemas reach authored CLI refusals."""
    raw = b"[" * depth + b"0" + b"]" * depth
    target = tmp_path / "history.csv"
    target.write_bytes(HISTORY)
    assert downloads.main(["--package", "scpn-control", "--csv", str(target)], fetch=lambda package: raw) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err in {
        "snapshot failed: pypistats returned invalid JSON\n",
        "snapshot failed: pypistats response must be a JSON object\n",
    }
    assert target.read_bytes() == HISTORY and list(tmp_path.iterdir()) == [target]


def test_transport_recursion_is_not_misclassified_as_invalid_json() -> None:
    """Programming faults in an injected transport remain outside decoder refusal."""

    def fetch(package: str) -> bytes:
        raise RecursionError("transport programming fault")

    with pytest.raises(RecursionError, match="transport programming fault"):
        downloads.fetch_overall("scpn-control", fetch=fetch)


def test_summary_refuses_inconsistent_rows_without_native_key_error() -> None:
    """The public summary applies the same row contract before indexing categories."""
    with pytest.raises(downloads.DownloadRowsError, match="missing with_mirrors"):
        downloads.summary("scpn-control", {"2026-07-17": {"without_mirrors": 1}})


def test_native_http_protocol_fault_has_authored_retry_text(local_http: LocalHTTPState) -> None:
    """A real malformed status line must not survive the transport refusal boundary."""
    local_http["invalid_status"] = True
    with pytest.raises(downloads.RetryableSnapshotError) as error:
        downloads.fetch_overall("scpn-control")
    assert str(error.value) == "pypistats request failed"
    assert "PRIVATE_" not in str(error.value)


@pytest.mark.parametrize("refused_event", ["tempfile.mkstemp", "os.rename"])
def test_public_main_native_publication_fault_preserves_predecessor(tmp_path: Path, refused_event: str) -> None:
    """Real creation/replacement audit faults preserve existing bytes and remove staging."""
    target = tmp_path / "history.csv"
    target.write_bytes(HISTORY)
    script = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
from tools.pypi_downloads import main

def refuse(event, arguments):
    if event == {refused_event!r}:
        raise PermissionError("PRIVATE_PUBLICATION_FAULT_CANARY")

sys.addaudithook(refuse)
raise SystemExit(main(["--package", "scpn-control", "--csv", {str(target)!r}], fetch=lambda package: {PAYLOAD!r}))
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=15, check=False
    )
    assert result.returncode == 1 and result.stdout == "" and result.stderr == SAFE_FAILURE
    assert target.read_bytes() == HISTORY and list(tmp_path.iterdir()) == [target]


def test_cold_import_preserves_public_error_pickle_addresses(tmp_path: Path) -> None:
    """Helper-first imports still round-trip authored errors at the original façade address."""
    script = f"""
import sys, pickle
sys.path.insert(0, {str(ROOT)!r})
from tools import pypi_download_protocol as protocol
from tools import pypi_download_csv as history
from tools import pypi_downloads as public
for error_type in (protocol.DownloadSnapshotError, protocol.RetryableSnapshotError,
                   protocol.DuplicateJSONObjectKeyError, protocol.DownloadConfigurationError,
                   history.DownloadRowsError):
    original = error_type("authored failure")
    restored = pickle.loads(pickle.dumps(original))
    assert type(restored) is error_type and str(restored) == "authored failure"
    assert getattr(public, error_type.__name__) is error_type
    assert error_type.__module__ == "tools.pypi_downloads"
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=15, check=False
    )
    assert result.returncode == 0 and result.stdout == "" and result.stderr == ""
