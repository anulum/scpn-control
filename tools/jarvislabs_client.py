# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JarvisLabs SDK lifecycle
"""Configure optional JLClient and request its instance lifecycle operations.

Calls use the SDK's global token and can contact JarvisLabs. Importing this
module performs neither configuration nor a provider request. SDK network
requests have their own unbounded timing; polling timeout bounds only the
local poll loop. No response here proves billing cessation or training quality.
"""

from __future__ import annotations

import importlib
import math
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from types import ModuleType


@contextmanager
def _provider_output() -> Iterator[None]:
    """Discard SDK standard streams during a request without storing their bytes.

    Redirection and SDK token configuration are process-global. This adapter
    supports one credential/request stream per process, not concurrent callers.
    Exceptions still propagate; callers decide whether to expose them.
    """
    with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink), redirect_stderr(sink):
        yield


def setup_jarvislabs(token: str) -> ModuleType:
    """Import JLClient lazily and configure its process-global token.

    Parameters
    ----------
    token : str
        Nonempty credential supplied by the caller; never printed here.

    Returns
    -------
    ModuleType
        Actual ``jlclient.jarvisclient`` module, with its global token set.

    Raises
    ------
    ValueError
        If the supplied token is empty or contains only whitespace.
    ImportError
        If the optional JLClient package cannot be imported.

    Notes
    -----
    Configuration does not authenticate the token. Concurrent callers share
    SDK global state; separate credentials require separate processes.
    """
    if not token.strip():
        raise ValueError("A nonempty JarvisLabs token is required")
    client = importlib.import_module("jlclient.jarvisclient")
    client.__dict__["token"] = token
    return client


def get_balance(jarvisclient: ModuleType) -> object:
    """Request and return the SDK's balance response without interpreting it.

    The configured module's ``User.get_balance`` contacts the provider. Its
    opaque response is not a validated balance schema and is not printed.
    SDK exceptions propagate to the caller.
    """
    with _provider_output():
        result: object = jarvisclient.User.get_balance()
    return result


def list_instances(jarvisclient: ModuleType) -> list[object]:
    """Request the provider's instances and require an actual list response.

    Return the opaque SDK instances. A non-list response raises RuntimeError;
    authentication and transport exceptions propagate. No instance is changed.
    """
    with _provider_output():
        result: object = jarvisclient.User.get_instances()
    if not isinstance(result, list):
        raise RuntimeError("JarvisLabs did not return an instance list")
    return result


def _machine_id(instance: object) -> str | int:
    """Require an SDK handle carrying a nonempty string or integer identity."""
    identity: object = getattr(instance, "machine_id", None)
    if isinstance(identity, bool) or not isinstance(identity, (str, int)):
        raise RuntimeError("JarvisLabs did not return an instance identity")
    if isinstance(identity, str) and not identity.strip():
        raise RuntimeError("JarvisLabs did not return an instance identity")
    return identity


def create_instance(jarvisclient: ModuleType, name: str = "scpn-ppo-train") -> object:
    """Request one A5000 GPU, PyTorch template and 20 GB of storage.

    Return the actual SDK handle after checking its identity and status type.
    Availability, prices and quota remain provider decisions. SDK failure or
    malformed responses raise; a partially created provider resource without
    a returned handle cannot be cleaned up by this function.
    """
    if not name.strip():
        raise ValueError("An instance name is required")
    with _provider_output():
        instance: object = jarvisclient.Instance.create(
            "GPU", gpu_type="A5000", num_gpus=1, storage=20, name=name, template="pytorch"
        )
    _machine_id(instance)
    if not isinstance(getattr(instance, "status", None), str):
        raise RuntimeError("JarvisLabs did not return an instance status")
    return instance


def wait_for_ready(instance: object, timeout: float = 300) -> object:
    """Poll ``User.get_instance`` for Running status and a nonempty SSH string.

    Use the configured global SDK and the original handle's machine identity.
    Timeout must be finite and positive; a monotonic deadline bounds the loop,
    excluding any SDK request that blocks internally. Failed or Destroyed
    status raises immediately; transient lookup errors retry after ten seconds
    or the remaining deadline. No remote command is executed here.
    """
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Readiness timeout must be finite and positive")
    identity = _machine_id(instance)
    client = importlib.import_module("jlclient.jarvisclient")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with _provider_output():
                refreshed: object = client.User.get_instance(instance_id=identity)
        except Exception:
            refreshed = None
        status: object = getattr(refreshed, "status", None)
        ssh: object = getattr(refreshed, "ssh_str", None)
        if status == "Running" and isinstance(ssh, str) and ssh.strip():
            return refreshed
        if status in ("Failed", "Destroyed"):
            raise RuntimeError("JarvisLabs instance cannot become ready")
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(10, remaining))
    raise TimeoutError("JarvisLabs instance readiness deadline expired")


def destroy_instance(instance: object) -> bool:
    """Request destruction and return whether SDK explicitly reports success.

    Only a mapping with ``success is True`` is acknowledged. None, malformed
    replies and exceptions return False with a fixed cleanup notice. True is
    provider request acknowledgement, not independently verified destruction
    or stopped billing. Caller must retain unresolved handles for cleanup.
    """
    try:
        _machine_id(instance)
        destroy = getattr(instance, "destroy", None)
        if not callable(destroy):
            raise RuntimeError("JarvisLabs handle cannot request destruction")
        with _provider_output():
            response: object = destroy()
        accepted = isinstance(response, dict) and response.get("success") is True
    except Exception:
        accepted = False
    if accepted:
        print("JarvisLabs acknowledged the destruction request.")
    else:
        print("JarvisLabs destruction was not confirmed; check the instance in the provider dashboard.")
    return accepted
