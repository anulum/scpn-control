# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Exact-current LIF runtime tests.

"""End-to-end tests against the immutable SC-NeuroCore LIF artefacts."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from importlib import resources
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from scpn_control.scpn import (
    ExactCurrentLIFBindingError,
    ExactCurrentLIFExecution,
    ExactCurrentLIFExecutionError,
    ExactCurrentLIFInputError,
    ExactCurrentLIFProfileBinding,
    ExactCurrentLIFRuntime,
    ExactCurrentLIFStateError,
    ExactCurrentLIFTransitionPacket,
    ExactCurrentLIFTransitionTick,
    FusionCompiler,
    StochasticPetriNet,
)
from scpn_control.scpn.exact_current_lif_runtime import (
    PACKET_SCHEMA,
    PROFILE_NAME,
    PROFILE_SCHEMA,
    RUNTIME_CHECKPOINT_SCHEMA,
    RUNTIME_EXECUTION_SCHEMA,
    SC_CONTRACT_COMMIT,
    SC_IMPLEMENTATION_COMMIT,
    SC_MODEL_SOURCE_SHA256,
    SC_PROFILE_DIGEST,
    SC_PROFILE_SHA256,
    SC_REFERENCE_PACKET_SHA256,
    STATE_SCHEMA,
)


def _reference_bytes(name: str) -> bytes:
    return resources.files("sc_neurocore.neurons").joinpath(f"reference_trace_data/{name}").read_bytes()


@pytest.fixture
def binding() -> ExactCurrentLIFProfileBinding:
    """Return the installed, digest-verified SC contract binding."""
    return ExactCurrentLIFProfileBinding.from_installed_reference()


def _single_tick(duration_ms: float, *currents: float) -> ExactCurrentLIFTransitionTick:
    return ExactCurrentLIFTransitionTick(duration_ms, (tuple(currents),))


def _packet_payload(execution: ExactCurrentLIFExecution, index: int = 0) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(execution.packets[index].packet_json))


def test_installed_binding_is_complete_and_exact(
    binding: ExactCurrentLIFProfileBinding,
) -> None:
    """The public binding exposes every required compatibility dimension."""
    assert binding.profile_artifact_sha256 == SC_PROFILE_SHA256
    assert binding.profile_digest == SC_PROFILE_DIGEST
    assert binding.profile_schema == PROFILE_SCHEMA
    assert binding.profile_name == PROFILE_NAME
    assert binding.state_schema == STATE_SCHEMA
    assert binding.packet_schema == PACKET_SCHEMA
    assert binding.model_source_sha256 == SC_MODEL_SOURCE_SHA256
    assert binding.implementation_commit == SC_IMPLEMENTATION_COMMIT
    assert binding.contract_commit == SC_CONTRACT_COMMIT
    assert binding.to_payload()["profile"] == {
        "name": PROFILE_NAME,
        "schema": PROFILE_SCHEMA,
        "artifact_sha256": SC_PROFILE_SHA256,
        "canonical_sha256": SC_PROFILE_DIGEST,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "unknown.profile.v2", "unsupported profile schema or name"),
        ("profile", "unknown_profile", "unsupported profile schema or name"),
    ],
)
def test_binding_rejects_profile_identity_drift(field: str, value: str, message: str) -> None:
    """Unknown profile names and schemas fail closed."""
    payload = cast(dict[str, Any], json.loads(_reference_bytes("exact_current_lif_profile_v1.json")))
    payload[field] = value
    with pytest.raises(ExactCurrentLIFBindingError, match=message):
        ExactCurrentLIFProfileBinding.from_json(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            implementation_commit=SC_IMPLEMENTATION_COMMIT,
            contract_commit=SC_CONTRACT_COMMIT,
        )


def test_binding_rejects_units_source_unknown_missing_duplicate_and_artifact_drift() -> None:
    """Semantic, structural, and byte-level profile drift is rejected."""
    source = _reference_bytes("exact_current_lif_profile_v1.json")
    base = cast(dict[str, Any], json.loads(source))
    cases: list[tuple[dict[str, Any], str]] = []

    units = json.loads(json.dumps(base))
    units["units"]["time"] = "s"
    cases.append((units, "profile unit contract mismatch"))
    model = json.loads(json.dumps(base))
    model["model"]["source_sha256"] = "0" * 64
    cases.append((model, "model source digest mismatch"))
    state_schema = json.loads(json.dumps(base))
    state_schema["state"]["serialization_schema"] = "unknown"
    cases.append((state_schema, "unsupported state schema"))
    unknown = json.loads(json.dumps(base))
    unknown["unexpected"] = True
    cases.append((unknown, "fields mismatch"))
    missing = json.loads(json.dumps(base))
    del missing["events"]
    cases.append((missing, "fields mismatch"))
    semantic_copy = json.loads(json.dumps(base))
    cases.append((semantic_copy, "profile artifact digest mismatch"))

    for payload, message in cases:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with pytest.raises(ExactCurrentLIFBindingError, match=message):
            ExactCurrentLIFProfileBinding.from_json(
                serialized,
                implementation_commit=SC_IMPLEMENTATION_COMMIT,
                contract_commit=SC_CONTRACT_COMMIT,
            )

    duplicate = source.decode().replace('"schema":', '"schema":"duplicate", "schema":', 1)
    with pytest.raises(ExactCurrentLIFBindingError, match="duplicate JSON field"):
        ExactCurrentLIFProfileBinding.from_json(
            duplicate,
            implementation_commit=SC_IMPLEMENTATION_COMMIT,
            contract_commit=SC_CONTRACT_COMMIT,
        )

    for malformed, message in [
        (b"not-json", "must be valid JSON"),
        (b"[]", "profile must be an object"),
        (b"{" + b'"padding":"' + b"x" * (64 * 1024) + b'"}', "exceeds"),
    ]:
        with pytest.raises(ExactCurrentLIFBindingError, match=message):
            ExactCurrentLIFProfileBinding.from_json(
                malformed,
                implementation_commit=SC_IMPLEMENTATION_COMMIT,
                contract_commit=SC_CONTRACT_COMMIT,
            )


@pytest.mark.parametrize(
    ("implementation", "contract", "message"),
    [
        ("0" * 40, SC_CONTRACT_COMMIT, "implementation commit mismatch"),
        (SC_IMPLEMENTATION_COMMIT, "0" * 40, "contract commit mismatch"),
        ("not-a-sha", SC_CONTRACT_COMMIT, "implementation commit mismatch"),
    ],
)
def test_binding_rejects_stale_or_invalid_commits(implementation: str, contract: str, message: str) -> None:
    """Only the implementation and delivery commits named by the profile are admitted."""
    with pytest.raises(ExactCurrentLIFBindingError, match=message):
        ExactCurrentLIFProfileBinding.from_json(
            _reference_bytes("exact_current_lif_profile_v1.json"),
            implementation_commit=implementation,
            contract_commit=contract,
        )


def test_runtime_matches_immutable_sc_multitick_reference(
    binding: ExactCurrentLIFProfileBinding,
) -> None:
    """CONTROL reproduces the complete immutable multi-tick SC packet."""
    reference_bytes = _reference_bytes("exact_current_lif_multitick_v1.json")
    assert __import__("hashlib").sha256(reference_bytes).hexdigest() == SC_REFERENCE_PACKET_SHA256
    reference = cast(dict[str, Any], json.loads(reference_bytes))
    ticks = tuple(_single_tick(tick["duration_ms"], *tick["currents"]) for tick in reference["ticks"])
    runtime = ExactCurrentLIFRuntime(("plasma-transition",), binding, shot_id="mif-reference-v1")

    execution = runtime.execute(ticks)

    assert _packet_payload(execution) == reference
    assert execution.to_payload()["schema"] == RUNTIME_EXECUTION_SCHEMA
    assert len(execution.sha256) == 64
    assert execution.packets[0].to_payload()["packet"] == reference


def test_state_persists_across_calls_and_free_decay_is_exact(
    binding: ExactCurrentLIFProfileBinding,
) -> None:
    """Membrane state crosses call boundaries and zero current decays freely."""
    runtime = ExactCurrentLIFRuntime(("accumulator",), binding, shot_id="shot-persist")
    first = runtime.execute((_single_tick(15.0, 20.0),))
    first_packet = _packet_payload(first)
    assert first_packet["events"] == []

    second = runtime.execute((_single_tick(15.0, 20.0),))
    second_packet = _packet_payload(second)
    assert second_packet["initial_state"] == first_packet["final_state"]
    assert len(second_packet["events"]) == 1

    decay_runtime = ExactCurrentLIFRuntime(("decay",), binding, shot_id="shot-decay")
    charged = _packet_payload(decay_runtime.execute((_single_tick(5.0, 10.0),)))
    decayed = _packet_payload(decay_runtime.execute((_single_tick(5.0, 0.0),)))
    assert decayed["initial_state"] == charged["final_state"]
    assert decayed["final_state"]["voltage"] < charged["final_state"]["voltage"]
    assert decayed["final_state"]["voltage"] > -65.0


def test_threshold_equality_hard_reset_zero_refractory_and_simultaneous_sum(
    binding: ExactCurrentLIFProfileBinding,
) -> None:
    """Inclusive threshold, reset, zero-refractory, and input summation match SC."""
    crossing_duration = 20.0 * math.log(2.0)
    split = ExactCurrentLIFRuntime(("split",), binding, shot_id="shot-equality")
    joined = ExactCurrentLIFRuntime(("joined",), binding, shot_id="shot-equality")

    split_packet = _packet_payload(split.execute((_single_tick(crossing_duration, 15.0, 15.0),)))
    joined_packet = _packet_payload(joined.execute((_single_tick(crossing_duration, 30.0),)))

    assert len(split_packet["events"]) == 1
    assert split_packet["events"] == joined_packet["events"]
    assert split_packet["state_trace"] == joined_packet["state_trace"]
    phases = [sample["phase"] for sample in split_packet["state_trace"]]
    assert phases == ["initial", "threshold", "reset", "tick_end"]
    assert split_packet["state_trace"][2]["voltage"] == -65.0
    assert split_packet["final_state"]["time_ms"] == crossing_duration


def test_checkpoint_restore_replay_and_explicit_shot_reset_are_exact(
    binding: ExactCurrentLIFProfileBinding,
) -> None:
    """A checkpoint restores a bit-identical continuation and reset is explicit."""
    runtime = ExactCurrentLIFRuntime(("first", "second"), binding, shot_id="shot-a")
    runtime.execute(
        (
            ExactCurrentLIFTransitionTick(5.0, ((10.0,), (15.0, 15.0))),
            ExactCurrentLIFTransitionTick(7.0, ((0.0,), (-5.0, 5.0))),
        )
    )
    checkpoint = runtime.serialize_checkpoint()
    checkpoint_payload = cast(dict[str, Any], json.loads(checkpoint))
    assert checkpoint_payload["schema"] == RUNTIME_CHECKPOINT_SCHEMA
    assert checkpoint_payload["transition_names"] == ["first", "second"]

    expected = runtime.execute((ExactCurrentLIFTransitionTick(9.0, ((20.0,), (30.0,))),)).to_json()
    runtime.restore_checkpoint(checkpoint)
    replay = runtime.execute((ExactCurrentLIFTransitionTick(9.0, ((20.0,), (30.0,))),)).to_json()
    assert replay == expected

    runtime.reset_shot("shot-b")
    reset_states = [json.loads(state)["state"] for state in runtime.serialized_states]
    assert reset_states == [
        {"reset_epoch": 1, "shot_id": "shot-b", "time_ms": 0.0, "voltage": -65.0},
        {"reset_epoch": 1, "shot_id": "shot-b", "time_ms": 0.0, "voltage": -65.0},
    ]


def test_checkpoint_rejections_are_failure_atomic(
    binding: ExactCurrentLIFProfileBinding,
) -> None:
    """Every malformed checkpoint leaves all transition sessions unchanged."""
    runtime = ExactCurrentLIFRuntime(("first", "second"), binding, shot_id="shot-a")
    runtime.execute((ExactCurrentLIFTransitionTick(5.0, ((10.0,), (20.0,))),))
    original = runtime.serialized_states
    payload = cast(dict[str, Any], json.loads(runtime.serialize_checkpoint()))

    mutations: list[tuple[dict[str, Any], str]] = []
    wrong_schema = json.loads(json.dumps(payload))
    wrong_schema["schema"] = "unknown"
    mutations.append((wrong_schema, "unsupported checkpoint schema"))
    wrong_binding = json.loads(json.dumps(payload))
    wrong_binding["binding"]["packet_schema"] = "unknown"
    mutations.append((wrong_binding, "compatibility binding mismatch"))
    wrong_order = json.loads(json.dumps(payload))
    wrong_order["transition_names"].reverse()
    mutations.append((wrong_order, "transition ordering mismatch"))
    wrong_count = json.loads(json.dumps(payload))
    wrong_count["states"].pop()
    mutations.append((wrong_count, "state count mismatch"))
    wrong_state_schema = json.loads(json.dumps(payload))
    wrong_state_schema["states"][1]["schema"] = "unknown"
    mutations.append((wrong_state_schema, "unsupported state schema"))
    wrong_state_digest = json.loads(json.dumps(payload))
    wrong_state_digest["states"][1]["profile_sha256"] = "0" * 64
    mutations.append((wrong_state_digest, "state profile digest mismatch"))
    wrong_state_field = json.loads(json.dumps(payload))
    wrong_state_field["states"][1]["state"]["unknown"] = True
    mutations.append((wrong_state_field, "fields mismatch"))
    wrong_shot = json.loads(json.dumps(payload))
    wrong_shot["states"][1]["state"]["shot_id"] = 7
    mutations.append((wrong_shot, "checkpoint shot_id must be a string"))
    split_shot = json.loads(json.dumps(payload))
    split_shot["states"][1]["state"]["shot_id"] = "shot-b"
    mutations.append((split_shot, "states must share"))
    split_time = json.loads(json.dumps(payload))
    split_time["states"][1]["state"]["time_ms"] = 7.0
    mutations.append((split_time, "states must share"))
    split_epoch = json.loads(json.dumps(payload))
    split_epoch["states"][1]["state"]["reset_epoch"] = 3
    mutations.append((split_epoch, "states must share"))
    wrong_state_shape = json.loads(json.dumps(payload))
    wrong_state_shape["states"][1] = []
    mutations.append((wrong_state_shape, "checkpoint state must be an object"))
    unknown_field = json.loads(json.dumps(payload))
    unknown_field["unknown"] = True
    mutations.append((unknown_field, "checkpoint fields mismatch"))

    for mutation, message in mutations:
        with pytest.raises(ExactCurrentLIFStateError, match=message):
            runtime.restore_checkpoint(json.dumps(mutation, sort_keys=True, separators=(",", ":")))
        assert runtime.serialized_states == original

    duplicate = runtime.serialize_checkpoint().replace('"schema":', '"schema":"duplicate","schema":', 1)
    with pytest.raises(ExactCurrentLIFStateError, match="duplicate JSON field"):
        runtime.restore_checkpoint(duplicate)
    assert runtime.serialized_states == original

    for malformed, message in [
        ("not-json", "must be valid JSON"),
        ("[]", "checkpoint must be an object"),
    ]:
        with pytest.raises(ExactCurrentLIFStateError, match=message):
            runtime.restore_checkpoint(malformed)
        assert runtime.serialized_states == original


def test_execution_overflow_is_failure_atomic(
    binding: ExactCurrentLIFProfileBinding,
) -> None:
    """A numerical failure cannot partially commit transition state."""
    runtime = ExactCurrentLIFRuntime(("transition",), binding, shot_id="shot-overflow")
    payload = cast(dict[str, Any], json.loads(runtime.serialize_checkpoint()))
    payload["states"][0]["state"]["time_ms"] = 1e308
    runtime.restore_checkpoint(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    original = runtime.serialized_states

    with pytest.raises(ExactCurrentLIFExecutionError, match="overflowed binary64"):
        runtime.execute((_single_tick(1e308, 0.0),))
    assert runtime.serialized_states == original


def test_late_second_transition_failure_rolls_back_completed_first_transition(
    binding: ExactCurrentLIFProfileBinding,
) -> None:
    """A later transition failure cannot commit an earlier completed session."""
    runtime = ExactCurrentLIFRuntime(("completes", "fails"), binding)
    original = runtime.serialized_states

    with pytest.raises(
        ExactCurrentLIFExecutionError,
        match="non-positive threshold-crossing interval",
    ):
        runtime.execute(
            (
                ExactCurrentLIFTransitionTick(
                    1.0,
                    (
                        (0.0,),
                        (1e308,),
                    ),
                ),
            )
        )

    assert runtime.serialized_states == original


@pytest.mark.parametrize(
    "tick",
    [
        ExactCurrentLIFTransitionTick(1.0, ((0.0,),)),
    ],
)
def test_runtime_rejects_wrong_tick_shape_without_state_change(
    binding: ExactCurrentLIFProfileBinding,
    tick: ExactCurrentLIFTransitionTick,
) -> None:
    """Runtime shape and type errors are rejected before state changes."""
    runtime = ExactCurrentLIFRuntime(("first", "second"), binding)
    original = runtime.serialized_states
    with pytest.raises(ExactCurrentLIFInputError, match="every compiled transition"):
        runtime.execute((tick,))
    with pytest.raises(ExactCurrentLIFInputError, match="only ExactCurrentLIFTransitionTick"):
        runtime.execute((cast(ExactCurrentLIFTransitionTick, object()),))
    assert runtime.serialized_states == original


@pytest.mark.parametrize(
    ("duration", "currents", "message"),
    [
        (True, ((0.0,),), "duration_ms"),
        (0.0, ((0.0,),), "duration_ms"),
        (math.inf, ((0.0,),), "duration_ms"),
        (1.0, ((True,),), "current contributions"),
        (1.0, ((math.nan,),), "current contributions"),
        (1.0, ((1e308, 1e308),), "summed current"),
    ],
)
def test_tick_rejects_invalid_numeric_inputs(
    duration: float, currents: tuple[tuple[float, ...], ...], message: str
) -> None:
    """Tick construction rejects invalid binary64 domains and overflow."""
    with pytest.raises(ExactCurrentLIFInputError, match=message):
        ExactCurrentLIFTransitionTick(duration, currents)


def test_runtime_and_result_constructors_reject_forged_contracts(
    binding: ExactCurrentLIFProfileBinding,
) -> None:
    """Public constructors cannot bypass binding and packet invariants."""
    with pytest.raises(ExactCurrentLIFBindingError, match="non-empty strings"):
        ExactCurrentLIFRuntime((), binding)
    with pytest.raises(ExactCurrentLIFBindingError, match="unique"):
        ExactCurrentLIFRuntime(("same", "same"), binding)
    with pytest.raises(ExactCurrentLIFBindingError, match="binding fields differ"):
        ExactCurrentLIFRuntime(("transition",), replace(binding, packet_schema="unknown"))
    with pytest.raises(ExactCurrentLIFBindingError, match="must be an"):
        ExactCurrentLIFRuntime(("transition",), cast(ExactCurrentLIFProfileBinding, object()))
    with pytest.raises(ExactCurrentLIFBindingError, match="non-empty string"):
        ExactCurrentLIFRuntime(("transition",), binding, shot_id="")
    with pytest.raises(ExactCurrentLIFExecutionError, match="transition_name"):
        ExactCurrentLIFTransitionPacket("", "{}")
    with pytest.raises(ExactCurrentLIFExecutionError, match="unsupported execution packet"):
        ExactCurrentLIFTransitionPacket("transition", "{}")
    with pytest.raises(ExactCurrentLIFExecutionError, match="packets must contain"):
        ExactCurrentLIFExecution(())

    valid_packet = ExactCurrentLIFRuntime(("one",), binding).execute((_single_tick(1.0, 0.0),)).packets[0]
    with pytest.raises(ExactCurrentLIFExecutionError, match="canonical JSON"):
        ExactCurrentLIFTransitionPacket("one", valid_packet.packet_json + "\n")
    with pytest.raises(ExactCurrentLIFExecutionError, match="unique"):
        ExactCurrentLIFExecution((valid_packet, valid_packet))


def test_tick_requires_structural_tuple_contract() -> None:
    """Mutable or non-nested current collections are rejected at the boundary."""
    with pytest.raises(ExactCurrentLIFInputError, match="tuple of current tuples"):
        ExactCurrentLIFTransitionTick(
            1.0,
            cast(tuple[tuple[float, ...], ...], [[1.0]]),
        )


def test_compiler_binds_distinct_runtime_without_changing_lif_fire(
    binding: ExactCurrentLIFProfileBinding,
) -> None:
    """Compiler opt-in is stateful while the established lif_fire gate is unchanged."""
    net = StochasticPetriNet()
    net.add_place("input", initial_tokens=1.0)
    net.add_transition("transition", threshold=0.5)
    net.add_arc("input", "transition", weight=1.0)
    compiler = FusionCompiler(bitstream_length=64, seed=7)

    baseline = compiler.compile(net)
    compiled = compiler.compile(
        net,
        exact_current_lif_binding=binding,
        exact_current_lif_shot_id="shot-compile",
    )

    assert baseline.exact_current_lif_runtime is None
    assert compiled.exact_current_lif_runtime is not None
    np.testing.assert_array_equal(
        compiled.lif_fire(np.array([0.5], dtype=np.float64)),
        baseline.lif_fire(np.array([0.5], dtype=np.float64)),
    )
    result = compiled.exact_current_lif_runtime.execute((_single_tick(5.0, 10.0),))
    assert _packet_payload(result)["final_state"]["shot_id"] == "shot-compile"


def test_reset_rejection_preserves_every_transition(
    binding: ExactCurrentLIFProfileBinding,
) -> None:
    """An invalid explicit shot boundary is failure-atomic."""
    runtime = ExactCurrentLIFRuntime(("first", "second"), binding)
    runtime.execute((ExactCurrentLIFTransitionTick(1.0, ((1.0,), (2.0,))),))
    original = runtime.serialized_states
    with pytest.raises(ExactCurrentLIFStateError, match="non-empty string"):
        runtime.reset_shot("")
    assert runtime.serialized_states == original


_PUBLIC_BINDING_PROBE = """
import importlib.util
import json
import sys
from pathlib import Path

source, fixture, profile_path, action = sys.argv[1:]
spec = importlib.util.spec_from_file_location("exact_current_lif_public_probe", source)
if spec is None or spec.loader is None:
    raise SystemExit("could not load public runtime module")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
sys.path[:] = [fixture] + [
    path for path in sys.path
    if "site-packages" not in path and "dist-packages" not in path
]
for name in tuple(sys.modules):
    if name == "sc_neurocore" or name.startswith("sc_neurocore."):
        del sys.modules[name]
try:
    if action == "installed-reference":
        module.ExactCurrentLIFProfileBinding.from_installed_reference()
    else:
        module.ExactCurrentLIFProfileBinding.from_json(
            Path(profile_path).read_bytes(),
            implementation_commit=module.SC_IMPLEMENTATION_COMMIT,
            contract_commit=module.SC_CONTRACT_COMMIT,
        )
except module.ExactCurrentLIFError as exc:
    print(json.dumps({"type": type(exc).__name__, "code": exc.code}))
else:
    raise SystemExit("incompatible installation was unexpectedly accepted")
"""


def _write_distribution_metadata(root: Path, version: str) -> None:
    dist_info = root / f"sc_neurocore-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: sc-neurocore\nVersion: {version}\n",
        encoding="utf-8",
    )


def _run_incompatible_installation_probe(
    fixture: Path,
    *,
    action: str = "from-json",
) -> dict[str, str]:
    source = Path(__file__).parents[1] / "src/scpn_control/scpn/exact_current_lif_runtime.py"
    profile = resources.files("sc_neurocore.neurons").joinpath("reference_trace_data/exact_current_lif_profile_v1.json")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _PUBLIC_BINDING_PROBE,
            str(source),
            str(fixture),
            str(profile),
            action,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return cast(dict[str, str], json.loads(completed.stdout))


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (
            "missing-distribution",
            {"type": "ExactCurrentLIFUnavailableError", "code": "exact_current_lif_unavailable"},
        ),
        (
            "wrong-version",
            {"type": "ExactCurrentLIFBindingError", "code": "exact_current_lif_binding_mismatch"},
        ),
        (
            "missing-source-path",
            {"type": "ExactCurrentLIFBindingError", "code": "exact_current_lif_binding_mismatch"},
        ),
        (
            "unreadable-source",
            {"type": "ExactCurrentLIFBindingError", "code": "exact_current_lif_binding_mismatch"},
        ),
        (
            "mismatched-source",
            {"type": "ExactCurrentLIFBindingError", "code": "exact_current_lif_binding_mismatch"},
        ),
        (
            "absent-public-api",
            {"type": "ExactCurrentLIFUnavailableError", "code": "exact_current_lif_unavailable"},
        ),
    ],
)
def test_public_binding_rejects_incompatible_installed_contracts(
    tmp_path: Path,
    case: str,
    expected: dict[str, str],
) -> None:
    """Isolated real package layouts fail through the typed public boundary."""
    package = tmp_path / "sc_neurocore"
    solvers = package / "solvers"
    solvers.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (solvers / "__init__.py").write_text("", encoding="utf-8")
    contract_source = solvers / "exact_lif_profile.py"

    if case != "missing-distribution":
        _write_distribution_metadata(tmp_path, "9.9.0" if case == "wrong-version" else "3.16.0")
    if case == "missing-source-path":
        contract_source.write_text("del __file__\n", encoding="utf-8")
    elif case == "unreadable-source":
        contract_source.write_text(
            f"__file__ = {str(solvers)!r}\n",
            encoding="utf-8",
        )
    elif case == "absent-public-api":
        installed_solvers = resources.files("sc_neurocore.solvers")
        contract_source.write_bytes(installed_solvers.joinpath("exact_lif_profile.py").read_bytes())
        (solvers / "exact_lif.py").write_bytes(installed_solvers.joinpath("exact_lif.py").read_bytes())
    else:
        contract_source.write_text("CONTRACT_PRESENT = True\n", encoding="utf-8")

    assert _run_incompatible_installation_probe(tmp_path) == expected


def test_public_binding_rejects_missing_installed_profile_artifact(
    tmp_path: Path,
) -> None:
    """A distribution without the immutable profile yields a typed failure."""
    package = tmp_path / "sc_neurocore"
    neurons = package / "neurons"
    neurons.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (neurons / "__init__.py").write_text("", encoding="utf-8")
    _write_distribution_metadata(tmp_path, "3.16.0")

    assert _run_incompatible_installation_probe(
        tmp_path,
        action="installed-reference",
    ) == {
        "type": "ExactCurrentLIFUnavailableError",
        "code": "exact_current_lif_unavailable",
    }
