# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JarvisLabs SDK lifecycle and training workflow.
"""JarvisLabs lifecycle calls and the training workflow against a stand-in SDK module.

The provider is a paid cloud service, so no test contacts it. A stand-in
``jlclient.jarvisclient`` module records every request and replays scripted
responses; the workflow's transport and SDK calls are replaced the same way.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from tools import jarvislabs_client as client
from tools import jarvislabs_train as workflow


class _StandInSDK(ModuleType):
    """Stand-in ``jlclient.jarvisclient`` module with the SDK names the client uses."""

    User: SimpleNamespace
    Instance: SimpleNamespace
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]]
    token: str


def _sdk(monkeypatch: pytest.MonkeyPatch, **responses: Any) -> _StandInSDK:
    """Install a stand-in ``jlclient.jarvisclient`` whose calls return or raise the scripted responses."""
    module = _StandInSDK("jlclient.jarvisclient")
    module.calls = []

    def respond(request: str, /, *args: Any, **kwargs: Any) -> Any:
        """Record a request, print provider noise and return or raise the next scripted response."""
        module.calls.append((request, args, kwargs))
        print("provider noise", request)
        script = responses[request]
        value = script.pop(0) if isinstance(script, list) else script
        if isinstance(value, Exception):
            raise value
        return value

    module.User = SimpleNamespace(
        get_balance=lambda: respond("balance"),
        get_instances=lambda: respond("instances"),
        get_instance=lambda **kwargs: respond("instance", **kwargs),
    )
    module.Instance = SimpleNamespace(create=lambda *a, **k: respond("create", *a, **k))
    package = ModuleType("jlclient")
    monkeypatch.setattr(package, "jarvisclient", module, raising=False)
    monkeypatch.setitem(sys.modules, "jlclient", package)
    monkeypatch.setitem(sys.modules, "jlclient.jarvisclient", module)
    return module


def test_setup_sets_the_global_token_and_refuses_a_blank_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setup imports the SDK lazily, stores the token on it, and refuses whitespace."""
    sdk = _sdk(monkeypatch)
    with pytest.raises(ValueError, match="nonempty JarvisLabs token"):
        client.setup_jarvislabs("   ")
    assert client.setup_jarvislabs("secret") is sdk
    assert sdk.token == "secret"


def test_balance_and_instance_list_are_returned_without_provider_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Responses pass through unchanged; SDK output is discarded; a non-list instance reply is refused."""
    sdk = _sdk(monkeypatch, balance={"credit": 3}, instances=[[1, 2], "not a list"])
    assert client.get_balance(sdk) == {"credit": 3}
    assert client.list_instances(sdk) == [1, 2]
    with pytest.raises(RuntimeError, match="instance list"):
        client.list_instances(sdk)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("identity", [None, True, "  ", 2.5])
def test_handles_without_a_usable_identity_are_refused(identity: object) -> None:
    """Only a nonempty string or a non-boolean integer identifies an instance."""
    with pytest.raises(RuntimeError, match="instance identity"):
        client._machine_id(SimpleNamespace(machine_id=identity))


def test_create_requests_the_documented_machine_and_checks_the_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    """One A5000 GPU, the PyTorch template and 20 GB are requested; the handle needs an identity and status."""
    good = SimpleNamespace(machine_id=7, status="Creating")
    sdk = _sdk(monkeypatch, create=[good, SimpleNamespace(machine_id="m1", status=None)])
    with pytest.raises(ValueError, match="instance name"):
        client.create_instance(sdk, name=" ")
    assert client.create_instance(sdk, name="run") is good
    assert sdk.calls[0] == (
        "create",
        ("GPU",),
        {"gpu_type": "A5000", "num_gpus": 1, "storage": 20, "name": "run", "template": "pytorch"},
    )
    with pytest.raises(RuntimeError, match="instance status"):
        client.create_instance(sdk)


class _Clock:
    """Monotonic clock stand-in that advances only when the poll loop sleeps."""

    def __init__(self) -> None:
        """Start at zero with no recorded sleeps."""
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        """Return the current time."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Record the wait and advance the clock by it."""
        self.sleeps.append(seconds)
        self.now += seconds


def test_readiness_retries_transient_errors_until_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lookup error and a pending status are retried every ten seconds until Running with SSH."""
    clock = _Clock()
    monkeypatch.setattr(client, "time", SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep))
    ready = SimpleNamespace(status="Running", ssh_str="ssh -p 22 root@host")
    sdk = _sdk(
        monkeypatch,
        instance=[OSError("lookup"), SimpleNamespace(status="Running", ssh_str=" "), ready],
    )
    assert client.wait_for_ready(SimpleNamespace(machine_id=9), timeout=60) is ready
    assert clock.sleeps == [10, 10]
    assert all(call[2] == {"instance_id": 9} for call in sdk.calls)


def test_readiness_refuses_bad_timeouts_failed_instances_and_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Timeouts must be finite and positive; Failed raises at once; the last sleep is the remaining time."""
    handle = SimpleNamespace(machine_id="m")
    for timeout in (True, math.nan, 0.0):
        with pytest.raises(ValueError, match="finite and positive"):
            client.wait_for_ready(handle, timeout=timeout)

    clock = _Clock()
    monkeypatch.setattr(client, "time", SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep))
    _sdk(monkeypatch, instance=SimpleNamespace(status="Failed", ssh_str=""))
    with pytest.raises(RuntimeError, match="cannot become ready"):
        client.wait_for_ready(handle, timeout=30)

    _sdk(monkeypatch, instance=SimpleNamespace(status="Pending", ssh_str=""))
    with pytest.raises(TimeoutError, match="deadline expired"):
        client.wait_for_ready(handle, timeout=25)
    assert clock.sleeps == [10, 10, 5]


def test_readiness_does_not_sleep_past_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a lookup itself consumes the remaining time, the loop ends without another sleep."""
    clock = _Clock()

    def slow_lookup(**_kwargs: Any) -> SimpleNamespace:
        """Return a pending instance after using up the whole deadline."""
        clock.now += 100.0
        return SimpleNamespace(status="Pending", ssh_str="")

    sdk = _sdk(monkeypatch)
    sdk.User.get_instance = slow_lookup
    monkeypatch.setattr(client, "time", SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep))
    with pytest.raises(TimeoutError):
        client.wait_for_ready(SimpleNamespace(machine_id=1), timeout=5)
    assert clock.sleeps == []


@pytest.mark.parametrize(
    ("handle", "accepted"),
    [
        (SimpleNamespace(machine_id=1, destroy=lambda: {"success": True}), True),
        (SimpleNamespace(machine_id=1, destroy=lambda: {"success": "yes"}), False),
        (SimpleNamespace(machine_id=1, destroy=lambda: None), False),
        (SimpleNamespace(machine_id=1), False),
        (SimpleNamespace(machine_id=None, destroy=lambda: {"success": True}), False),
    ],
)
def test_destruction_is_acknowledged_only_for_explicit_success(
    handle: object, accepted: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only a mapping with success True counts; every other reply prints the dashboard notice."""
    assert client.destroy_instance(handle) is accepted
    expected = "acknowledged the destruction" if accepted else "check the instance in the provider dashboard"
    assert expected in capsys.readouterr().out


def test_destruction_exceptions_are_contained(capsys: pytest.CaptureFixture[str]) -> None:
    """A raising destroy request is reported as unconfirmed, not propagated."""

    def explode() -> None:
        """Raise like a failing provider request."""
        raise ConnectionError("down")

    assert client.destroy_instance(SimpleNamespace(machine_id="m", destroy=explode)) is False
    assert "not confirmed" in capsys.readouterr().out


def _workflow_stubs(
    monkeypatch: pytest.MonkeyPatch, *, ssh: object = "ssh root@host", destroyed: bool = True
) -> list[Any]:
    """Replace every provider, transport and custody call of the workflow; return the recorded calls."""
    calls: list[Any] = []
    handle = SimpleNamespace(machine_id=5, status="Creating")
    monkeypatch.setenv("JARVISLABS_TOKEN", "token")
    monkeypatch.setattr(workflow, "_check_artifact_paths", lambda output_dir: "campaign-1")

    def setup(token: str) -> str:
        """Record the setup and return a stand-in SDK name."""
        calls.append(("setup", token))
        return "sdk"

    def listing(sdk: str) -> list[object]:
        """Record the listing and return no instances."""
        calls.append(("list", sdk))
        return []

    def create(sdk: str) -> SimpleNamespace:
        """Record the creation and return the handle."""
        calls.append(("create", sdk))
        return handle

    def destroy(instance: object) -> bool:
        """Record the destruction request and return the scripted acknowledgement."""
        calls.append(("destroy", instance))
        return destroyed

    monkeypatch.setattr(workflow, "setup_jarvislabs", setup)
    monkeypatch.setattr(workflow, "get_balance", lambda sdk: calls.append(("balance", sdk)))
    monkeypatch.setattr(workflow, "list_instances", listing)
    monkeypatch.setattr(workflow, "create_instance", create)
    monkeypatch.setattr(workflow, "wait_for_ready", lambda instance: SimpleNamespace(ssh_str=ssh))
    monkeypatch.setattr(workflow, "run_ssh_command", lambda s, c, timeout: calls.append(("ssh", c, timeout)))
    monkeypatch.setattr(workflow, "scp_upload", lambda s, local, remote: calls.append(("up", remote)))
    monkeypatch.setattr(workflow, "scp_download", lambda s, remote, local: calls.append(("down", remote, local)))
    monkeypatch.setattr(workflow, "_report_downloads", lambda output_dir: calls.append(("report", output_dir)))
    monkeypatch.setattr(workflow, "destroy_instance", destroy)
    return calls


def test_workflow_runs_the_recipe_and_always_offers_cleanup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The full recipe uploads seven sources, trains on CPU in the campaign directory and destroys the handle."""
    calls = _workflow_stubs(monkeypatch)
    assert workflow.main(["--output-dir", str(tmp_path)]) == 0
    kinds = [call[0] for call in calls]
    assert kinds[:4] == ["setup", "balance", "list", "create"]
    assert kinds.count("up") == len(workflow.UPLOAD_FILES) and kinds.count("down") == len(workflow.DOWNLOAD_FILES)
    train = [call[1] for call in calls if call[0] == "ssh"][-1]
    assert "CUDA_VISIBLE_DEVICES=''" in train and "--output-dir artifacts/rl/campaign-1" in train
    assert ("report", tmp_path) in calls and kinds[-1] == "destroy"
    downloads = [call for call in calls if call[0] == "down"]
    assert downloads[0][1] == "scpn-control/artifacts/rl/campaign-1/ppo_tokamak.zip"
    assert downloads[0][2] == str(tmp_path / "ppo_tokamak.zip")


def test_workflow_failures_and_unconfirmed_cleanup_return_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing SSH endpoint fails the run but still destroys; unconfirmed cleanup fails a delivered run."""
    calls = _workflow_stubs(monkeypatch, ssh=None)
    assert workflow.main([]) == 1
    assert "results are not accepted" in capsys.readouterr().out
    assert calls[-1][0] == "destroy"

    _workflow_stubs(monkeypatch, destroyed=False)
    assert workflow.main([]) == 1

    calls = _workflow_stubs(monkeypatch)
    monkeypatch.setattr(workflow, "create_instance", lambda sdk: (_ for _ in ()).throw(RuntimeError("quota")))
    assert workflow.main([]) == 1
    assert "destroy" not in [call[0] for call in calls]


def test_artifact_check_requires_sources_and_recorded_custody(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A missing upload source or an unrecorded benchmark campaign stops before any provider call."""
    monkeypatch.setattr(workflow, "REPO_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="upload source"):
        workflow._check_artifact_paths(tmp_path)
    for name in workflow.UPLOAD_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("source\n", encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()
    monkeypatch.setattr(workflow, "require_recorded_campaign", lambda path, repository_root: None)
    with pytest.raises(RuntimeError, match="recorded benchmark custody"):
        workflow._check_artifact_paths(output)


def test_download_report_requires_nine_files_and_prints_the_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every download must be a nonempty regular file; the benchmark metrics are printed per controller."""
    paths = workflow._download_paths(tmp_path)
    for path in paths[:-1]:
        path.write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="training download is unavailable"):
        workflow._report_downloads(tmp_path)
    paths[-1].write_text("x", encoding="utf-8")
    seen: list[Path] = []

    def fake_benchmark(path: Path) -> dict[str, dict[str, float]]:
        """Record the benchmark path and return one controller's metrics."""
        seen.append(path)
        return {"PPO": {"mean_reward": 12.34, "std_reward": 1.5, "disruption_rate": 0.25}}

    monkeypatch.setattr(workflow, "load_benchmark", fake_benchmark)
    workflow._report_downloads(tmp_path)
    assert seen == [tmp_path / "rl_vs_classical.json"]
    assert capsys.readouterr().out == "PPO: reward=12.3 +/- 1.5, disruption=25%\n"
