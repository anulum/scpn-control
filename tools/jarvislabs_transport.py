# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JarvisLabs native transport
"""Run validated native SSH/SCP commands and propagate transport failures.

This module never installs tools, configures keys or accepts unknown host keys.
Successful operations require native OpenSSH executables, caller-managed keys
and known-host entries. Failed downloads can leave partial files; no transfer
here authenticates content, guarantees atomicity or creates benchmark custody.
"""

from __future__ import annotations

import ipaddress
import math
import re
import shlex
import subprocess
from pathlib import Path

_OPTIONS = ("-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=30")


def _endpoint(ssh_str: str) -> tuple[str, str]:
    """Parse only ssh user@host or ssh -p PORT user@host without extra flags."""
    if any(ord(character) < 32 for character in ssh_str):
        raise ValueError("SSH endpoint contains control characters")
    try:
        parts = shlex.split(ssh_str)
    except ValueError:
        raise ValueError("SSH endpoint syntax is invalid") from None
    port = "22"
    if len(parts) == 2 and parts[0] == "ssh":
        host = parts[1]
    elif len(parts) == 4 and parts[:2] == ["ssh", "-p"]:
        port, host = parts[2:]
    else:
        raise ValueError("SSH endpoint must contain only user, host and optional port")
    if re.fullmatch(r"[0-9]{1,5}", port) is None or not 1 <= int(port) <= 65535:
        raise ValueError("SSH port is outside its permitted range")
    user, separator, hostname = host.partition("@")
    if separator != "@" or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", user) is None:
        raise ValueError("SSH endpoint requires a valid user and host")
    if hostname.startswith("[") and hostname.endswith("]"):
        try:
            ipaddress.IPv6Address(hostname[1:-1])
        except ValueError:
            raise ValueError("SSH endpoint host is invalid") from None
    elif re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", hostname) is None:
        raise ValueError("SSH endpoint host is invalid")
    return port, host


def _timeout(timeout: float) -> None:
    """Refuse boolean, nonfinite or nonpositive native process deadlines."""
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Transport timeout must be finite and positive")


def _remote_path(remote_path: str) -> None:
    """Require a plain nonempty POSIX transfer path without shell punctuation."""
    if re.fullmatch(r"[A-Za-z0-9_./-]+", remote_path) is None or remote_path.startswith("-"):
        raise ValueError("Remote transfer path is invalid")


def run_ssh_command(ssh_str: str, command: str, timeout: float = 1800) -> subprocess.CompletedProcess[str]:
    """Execute a remote shell command through native SSH and require exit zero.

    Parameters
    ----------
    ssh_str : str
        Exactly ``ssh user@host`` or ``ssh -p PORT user@host``.
    command : str
        Nonempty remote shell program. Shell interpretation occurs remotely.
    timeout : float
        Finite positive wall-clock process deadline in seconds.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Successful native result with captured text stdout and stderr.

    Raises
    ------
    ValueError
        If endpoint, command or timeout syntax is invalid.
    subprocess.CalledProcessError
        If SSH or the remote command exits unsuccessfully.
    subprocess.TimeoutExpired
        If the native process deadline expires.
    OSError
        If native SSH cannot be launched.

    Notes
    -----
    Host-key checking and noninteractive authentication are required. Output,
    arguments and exceptions are not printed; callers control their disclosure.
    """
    port, host = _endpoint(ssh_str)
    _timeout(timeout)
    if not command.strip() or "\x00" in command:
        raise ValueError("A nonempty remote command without NUL is required")
    return subprocess.run(
        ["ssh", "-p", port, *_OPTIONS, "--", host, command],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )


def scp_upload(
    ssh_str: str, local_path: str, remote_path: str, timeout: float = 300
) -> subprocess.CompletedProcess[str]:
    """Upload an existing local file or directory using checked native SCP.

    Use the same endpoint/authentication contract as run_ssh_command. Resolve
    the source to an absolute path; missing or non-file/non-directory sources
    raise FileNotFoundError before transport. Remote paths accept plain POSIX
    characters only. Recursive uploads retain SCP semantics and may replace
    remote bytes. Return captured successful output; native errors propagate.
    """
    port, host = _endpoint(ssh_str)
    _timeout(timeout)
    _remote_path(remote_path)
    source = Path(local_path).resolve()
    if not source.is_file() and not source.is_dir():
        raise FileNotFoundError("Local upload source is unavailable")
    return subprocess.run(
        ["scp", "-P", port, *_OPTIONS, "-r", "--", str(source), f"{host}:{remote_path}"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )


def scp_download(
    ssh_str: str, remote_path: str, local_path: str, timeout: float = 300
) -> subprocess.CompletedProcess[str]:
    """Download with checked native SCP to a new path with an existing parent.

    Existing files, directories and symlinks are refused with FileExistsError;
    missing parents raise NotADirectoryError. No output is called fresh merely
    because old bytes exist. Native failures propagate and can leave a partial
    candidate. The path checks do not lock the destination against concurrent
    writers, validate remote content or make the transfer atomic.
    """
    port, host = _endpoint(ssh_str)
    _timeout(timeout)
    _remote_path(remote_path)
    target = Path(local_path).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError("Download destination already exists")
    if not target.parent.is_dir():
        raise NotADirectoryError("Download parent directory is unavailable")
    return subprocess.run(
        ["scp", "-P", port, *_OPTIONS, "-r", "--", f"{host}:{remote_path}", str(target)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
