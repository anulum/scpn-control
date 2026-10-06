# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JarvisLabs real refusal tests
"""Exercise actual native OpenSSH refusals and preserve caller file bytes."""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tools.jarvislabs_train import run_ssh_command, scp_download, scp_upload


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="on the macOS runner a bound, non-listening loopback port is not refused; the connection times out",
)
def test_actual_native_transport_failure_preserves_files(tmp_path: Path) -> None:
    """A bound non-listening loopback socket forces real SSH and SCP failures."""
    source = tmp_path / "source.dat"
    source.write_bytes(b"caller-owned-source")
    target = tmp_path / "candidate.dat"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as closed:
        closed.bind(("127.0.0.1", 0))
        endpoint = f"ssh -p {closed.getsockname()[1]} root@127.0.0.1"
        with pytest.raises(subprocess.CalledProcessError) as ssh_failure:
            run_ssh_command(endpoint, "true", timeout=10)
        assert ssh_failure.value.returncode != 0
        assert "refused" in ssh_failure.value.stderr.lower()
        with pytest.raises(subprocess.CalledProcessError) as upload_failure:
            scp_upload(endpoint, str(source), "candidate.dat", timeout=10)
        assert upload_failure.value.returncode != 0
        with pytest.raises(subprocess.CalledProcessError) as download_failure:
            scp_download(endpoint, "source.dat", str(target), timeout=10)
        assert download_failure.value.returncode != 0
    assert source.read_bytes() == b"caller-owned-source"
    assert not target.exists()


def test_actual_download_custody_and_upload_source_refusals(tmp_path: Path) -> None:
    """Existing destinations, dangling links and absent inputs refuse locally."""
    endpoint = "ssh root@127.0.0.1"
    target = tmp_path / "retained.dat"
    target.write_bytes(b"retained-artifact")
    with pytest.raises(FileExistsError):
        scp_download(endpoint, "remote.dat", str(target))
    assert target.read_bytes() == b"retained-artifact"
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(FileExistsError):
        scp_download(endpoint, "remote.dat", str(directory))
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        scp_download(endpoint, "remote.dat", str(link))
    assert link.is_symlink()
    with pytest.raises(NotADirectoryError):
        scp_download(endpoint, "remote.dat", str(tmp_path / "absent-parent" / "new.dat"))
    with pytest.raises(FileNotFoundError):
        scp_upload(endpoint, str(tmp_path / "absent"), "remote.dat")


def test_actual_endpoint_and_timeout_refusals(tmp_path: Path) -> None:
    """Malformed endpoint and deadline inputs refuse through public transports."""
    source = tmp_path / "source.dat"
    source.write_bytes(b"source")
    invalid = (
        "scp root@127.0.0.1",
        "ssh -p 0 root@127.0.0.1",
        "ssh -p 65536 root@127.0.0.1",
        "ssh -p nope root@127.0.0.1",
        "ssh root@-127.0.0.1",
        "ssh root@127.0.0.1 extra",
        "ssh -o BatchMode=no root@127.0.0.1",
        "ssh 'root@127.0.0.1",
        "ssh root@127.0.0.1\n",
        "ssh root@[invalid]",
        "ssh root@[::1] extra",
        "ssh @127.0.0.1",
    )
    for endpoint in invalid:
        with pytest.raises(ValueError):
            run_ssh_command(endpoint, "true")
        with pytest.raises(ValueError):
            scp_upload(endpoint, str(source), "remote.dat")
        with pytest.raises(ValueError):
            scp_download(endpoint, "remote.dat", str(tmp_path / "new.dat"))
    for timeout in (0.0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            run_ssh_command("ssh root@127.0.0.1", "true", timeout)
        with pytest.raises(ValueError):
            scp_upload("ssh root@127.0.0.1", str(source), "remote.dat", timeout)
        with pytest.raises(ValueError):
            scp_download("ssh root@127.0.0.1", "remote.dat", str(tmp_path / "new.dat"), timeout)
    for command in ("", " ", "true\x00"):
        with pytest.raises(ValueError):
            run_ssh_command("ssh root@127.0.0.1", command)
    for remote in ("", "-option", "path; command", "path\nsecond"):
        with pytest.raises(ValueError):
            scp_upload("ssh root@127.0.0.1", str(source), remote)
        with pytest.raises(ValueError):
            scp_download("ssh root@127.0.0.1", remote, str(tmp_path / "new.dat"))
    assert source.read_bytes() == b"source"
