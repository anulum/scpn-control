# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Distributed CI modularity tests.

"""Exercise CI ownership, parity inventory, and fail-closed aggregation."""

from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path

import pytest

from tools import check_ci_workflow_modularity as modularity
from tools import ci_workflow_inventory as inventory
from tools.ci_workflow_inventory import WorkflowCategory, WorkflowPolicy

_ACTION_SHA = "0123456789abcdef0123456789abcdef01234567"


def _policy() -> WorkflowPolicy:
    """Return a complete two-job policy for isolated mutation tests."""
    return {
        "schema_version": 1,
        "coordinator": ".github/workflows/ci.yml",
        "required_gate": "ci-gate",
        "stable_required_jobs": ["unit"],
        "event_policy": {
            "push_branches": ["main"],
            "pull_request_branches": ["main"],
            "permissions": {},
            "concurrency_group": "ci-${{ github.ref }}",
            "cancel_in_progress": True,
        },
        "limits": {
            "coordinator_max_lines": 100,
            "coordinator_max_bytes": 16_384,
            "reusable_max_lines": 100,
            "reusable_max_bytes": 16_384,
            "max_reusable_workflows": 4,
        },
        "categories": [
            {
                "id": "unit-quality",
                "workflow": ".github/workflows/ci-unit-quality.yml",
                "caller_needs": [],
                "secrets": ["TOKEN"],
                "jobs": ["unit", "consume"],
            }
        ],
        "job_order": ["unit", "consume"],
        "dependency_graph": {"consume": ["unit"]},
        "artifacts": [
            {
                "name": "unit-evidence",
                "producer": "unit",
                "consumers": ["consume"],
            }
        ],
        "conditional_steps": {"unit": ["Conditional probe"]},
        "step_output_consumers": {"unit": ["probe.value"]},
    }


def _write_fixture(root: Path, policy: WorkflowPolicy | None = None) -> WorkflowPolicy:
    """Write one valid distributed CI fixture below root."""
    resolved = _policy() if policy is None else policy
    workflow_root = root / ".github" / "workflows"
    workflow_root.mkdir(parents=True, exist_ok=True)
    (root / "tools").mkdir(exist_ok=True)
    (root / "tests").mkdir(exist_ok=True)
    (root / "scripts").mkdir(exist_ok=True)
    (root / "tools" / "ci_workflow_policy.json").write_text(
        json.dumps(resolved),
        encoding="utf-8",
    )
    (workflow_root / "ci.yml").write_text(
        """name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
permissions: {}
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true
jobs:
  unit-quality:
    uses: ./.github/workflows/ci-unit-quality.yml
    secrets:
      TOKEN: ${{ secrets.TOKEN }}
  ci-gate:
    needs: [unit-quality]
    runs-on: ubuntu-latest
    if: always()
    steps:
      - env:
          CATEGORY_RESULTS: ${{ toJSON(needs) }}
        run: |
          failures = {name: value["result"] for name, value in results.items() if value["result"] != "success"}
""",
        encoding="utf-8",
    )
    (workflow_root / "ci-unit-quality.yml").write_text(
        f"""name: CI / Unit Quality
on:
  workflow_call:
    secrets:
      TOKEN:
        required: false
permissions: {{}}
jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@{_ACTION_SHA}
      - name: Conditional probe
        id: probe
        if: always()
        run: echo "value=ready"
      - run: echo "${{{{ steps.probe.outputs.value }}}}"
      - uses: actions/upload-artifact@{_ACTION_SHA}
        with:
          name: unit-evidence
          path: evidence.json
  consume:
    needs: unit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@{_ACTION_SHA}
        with:
          name: unit-evidence
""",
        encoding="utf-8",
    )
    return resolved


def _redirect(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point inventory and audit modules at an isolated repository."""
    monkeypatch.setattr(inventory, "REPOSITORY_ROOT", root)
    monkeypatch.setattr(inventory, "CI_WORKFLOW_POLICY", root / "tools/ci_workflow_policy.json")
    monkeypatch.setattr(modularity, "REPOSITORY_ROOT", root)


def test_live_inventory_is_complete_unique_bounded_and_fail_closed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Require the real ten-category, 27-job distributed graph to pass."""
    policy = inventory.load_ci_workflow_policy()
    jobs = [job for category in policy["categories"] for job in category["jobs"]]

    assert len(policy["categories"]) == 10
    assert len(jobs) == len(set(jobs)) == len(policy["job_order"]) == 27
    assert set(jobs) == set(policy["job_order"])
    assert modularity.audit_ci_workflow_modularity(policy) == []
    assert modularity.main() == 0
    assert "27 jobs in 10 categories" in capsys.readouterr().out


def test_inventory_reconstructs_jobs_and_resolves_exclusive_owners() -> None:
    """Expose real job bodies without restoring a duplicate monolith."""
    source = inventory.read_ci_workflow_source()
    policy = inventory.load_ci_workflow_policy()

    assert source.count("  python-tests:\n") == 1
    assert source.count("  ci-gate:\n") == 1
    assert source.index("  python-tests:\n") < source.index("  studio-web:\n")
    assert inventory.workflow_path_for_job("python-lint").name == "ci-static-governance.yml"
    assert inventory.workflow_path_for_job("ci-gate").name == "ci.yml"
    assert inventory.category_for_job("studio-web") == "product-deployment"
    assert inventory.ci_workflow_paths(policy)[0].name == "ci.yml"
    with pytest.raises(KeyError):
        inventory.workflow_path_for_job("unknown-job")
    with pytest.raises(KeyError):
        inventory.category_for_job("unknown-job")


def test_inventory_rejects_non_object_duplicate_and_missing_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed on malformed policy, duplicate jobs, and an absent gate."""
    policy = _write_fixture(tmp_path)
    _redirect(tmp_path, monkeypatch)
    policy_path = tmp_path / "tools/ci_workflow_policy.json"

    policy_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        inventory.load_ci_workflow_policy()
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    category = tmp_path / ".github/workflows/ci-unit-quality.yml"
    category.write_text(
        category.read_text(encoding="utf-8") + "  unit:\n    runs-on: ubuntu-latest\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="multiple times in one workflow"):
        inventory.read_ci_workflow_source()

    _write_fixture(tmp_path, policy)
    duplicate = copy.deepcopy(policy["categories"][0])
    duplicate["id"] = "duplicate-quality"
    duplicate["workflow"] = ".github/workflows/ci-duplicate-quality.yml"
    policy["categories"].append(duplicate)
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    duplicate_path = tmp_path / duplicate["workflow"]
    duplicate_path.write_text(
        (tmp_path / ".github/workflows/ci-unit-quality.yml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="multiple reusable workflows"):
        inventory.read_ci_workflow_source()

    policy = _policy()
    _write_fixture(tmp_path, policy)
    coordinator = tmp_path / ".github/workflows/ci.yml"
    coordinator.write_text("name: CI\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing jobs"):
        inventory.read_ci_workflow_source()

    _write_fixture(tmp_path, policy)
    coordinator.write_text("name: CI\non:\n  push:\njobs:\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required gate"):
        inventory.read_ci_workflow_source()

    _write_fixture(tmp_path, policy)
    policy["job_order"] = ["unit", "consume", "absent"]
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="missing jobs"):
        inventory.read_ci_workflow_source()


def test_audit_reports_policy_limits_and_physical_inventory_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject unsupported schemas, count growth, size growth, and stray categories."""
    policy = _write_fixture(tmp_path)
    _redirect(tmp_path, monkeypatch)
    policy["schema_version"] = 2
    policy["limits"].update(
        {
            "max_reusable_workflows": 0,
            "coordinator_max_lines": 1,
            "coordinator_max_bytes": 1,
            "reusable_max_lines": 1,
            "reusable_max_bytes": 1,
        }
    )
    (tmp_path / ".github/workflows/ci-stray.yml").write_text(
        "name: stray\non:\n  workflow_call:\njobs: {}\n",
        encoding="utf-8",
    )

    errors = modularity.audit_ci_workflow_modularity(policy)

    assert any("schema_version" in error for error in errors)
    assert any("reusable workflow count" in error for error in errors)
    assert any("physical CI category" in error for error in errors)
    assert sum("lines exceed" in error for error in errors) == 2
    assert sum("bytes exceed" in error for error in errors) == 2


def test_audit_reports_coordinator_policy_and_call_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject trigger, permission, concurrency, call, and secret-surface changes."""
    policy = _write_fixture(tmp_path)
    _redirect(tmp_path, monkeypatch)
    coordinator = tmp_path / ".github/workflows/ci.yml"
    text = coordinator.read_text(encoding="utf-8")
    text = text.replace("branches: [main]", "branches: [other]", 1)
    text = text.replace("  pull_request:\n    branches: [main]\n", "  workflow_dispatch:\n")
    text = text.replace("permissions: {}", "permissions:\n  contents: write")
    text = text.replace("group: ci-${{ github.ref }}", "group: drift")
    text = text.replace("cancel-in-progress: true", "cancel-in-progress: false")
    text = text.replace("uses: ./.github/workflows/ci-unit-quality.yml", "uses: ./wrong.yml")
    text = text.replace("TOKEN: ${{ secrets.TOKEN }}", "TOKEN: wrong")
    text = text.replace(
        "  unit-quality:\n",
        "  unit-quality:\n    needs: unexpected\n    runs-on: ubuntu-latest\n",
    )
    coordinator.write_text(text, encoding="utf-8")

    errors = modularity.audit_ci_workflow_modularity(policy)

    assert any("push trigger" in error for error in errors)
    assert any("pull-request trigger" in error for error in errors)
    assert any("undeclared triggers" in error for error in errors)
    assert any("permissions differ" in error for error in errors)
    assert any("concurrency differs" in error for error in errors)
    assert any("targets the wrong workflow" in error for error in errors)
    assert any("incorrect dependencies" in error for error in errors)
    assert any("incorrect secret surface" in error for error in errors)
    assert any("executable job fields" in error for error in errors)


def test_audit_reports_category_ownership_secret_and_action_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject direct triggers, privilege growth, mutable actions, and owner drift."""
    policy = _write_fixture(tmp_path)
    _redirect(tmp_path, monkeypatch)
    category = tmp_path / ".github/workflows/ci-unit-quality.yml"
    text = category.read_text(encoding="utf-8")
    text = text.replace("workflow_call:", "push:")
    text = text.replace("required: false", "required: true")
    text = text.replace("permissions: {}", "permissions:\n  contents: write")
    text = text.replace(f"actions/checkout@{_ACTION_SHA}", "actions/checkout@main")
    text = text.replace("  consume:\n", "  renamed:\n")
    category.write_text(text, encoding="utf-8")

    errors = modularity.audit_ci_workflow_modularity(policy)

    assert any("expose only workflow_call" in error for error in errors)
    assert any("permissions differ" in error for error in errors)
    assert any("job order or ownership" in error for error in errors)
    assert any("unpinned action" in error for error in errors)
    assert any("incomplete or contains undeclared" in error for error in errors)


def test_audit_reports_reusable_secret_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require the declared reusable secret surface to remain optional and exact."""
    policy = _write_fixture(tmp_path)
    _redirect(tmp_path, monkeypatch)
    category = tmp_path / ".github/workflows/ci-unit-quality.yml"
    category.write_text(
        category.read_text(encoding="utf-8")
        .replace("required: false", "required: true")
        .replace("      TOKEN:\n", "      OTHER_TOKEN:\n"),
        encoding="utf-8",
    )

    errors = modularity.audit_ci_workflow_modularity(policy)

    assert any("secrets must be optional" in error for error in errors)
    assert any("secrets differ from policy" in error for error in errors)

    _write_fixture(tmp_path, policy)
    category.write_text(
        category.read_text(encoding="utf-8").replace(
            "    secrets:\n      TOKEN:\n        required: false\n",
            "    secrets: []\n",
        ),
        encoding="utf-8",
    )
    assert any("secrets differ from policy" in error for error in modularity.audit_ci_workflow_modularity(policy))


def test_audit_reports_dependency_artifact_condition_and_output_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind executable ordering and evidence plumbing to the versioned policy."""
    policy = _write_fixture(tmp_path)
    _redirect(tmp_path, monkeypatch)
    category = tmp_path / ".github/workflows/ci-unit-quality.yml"
    text = category.read_text(encoding="utf-8")
    text = text.replace("    needs: unit\n", "")
    text = text.replace("name: unit-evidence", "name: changed-evidence", 1)
    text = text.replace("name: Conditional probe", "name: Changed probe")
    text = text.replace("steps.probe.outputs.value", "steps.other.outputs.value")
    category.write_text(text, encoding="utf-8")

    errors = modularity.audit_ci_workflow_modularity(policy)

    assert any("dependencies differ" in error for error in errors)
    assert any("artifact producers or consumers" in error for error in errors)
    assert any("conditional-step inventory" in error for error in errors)
    assert any("step-output dependency" in error for error in errors)


def test_audit_reports_gate_and_legacy_reader_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject incomplete aggregation and renewed monolith coupling."""
    policy = _write_fixture(tmp_path)
    _redirect(tmp_path, monkeypatch)
    coordinator = tmp_path / ".github/workflows/ci.yml"
    text = coordinator.read_text(encoding="utf-8")
    text = text.replace("needs: [unit-quality]", "needs: []")
    text = text.replace("if: always()", "if: success()")
    text = text.replace("toJSON(needs)", "needs")
    text = text.replace('value["result"] != "success"', "False")
    coordinator.write_text(text, encoding="utf-8")
    (tmp_path / "tests/bad_reader.py").write_text(
        'Path(".github/workflows/ci.yml").read_text()\n',
        encoding="utf-8",
    )

    errors = modularity.audit_ci_workflow_modularity(policy)

    assert any("does not aggregate" in error for error in errors)
    assert any("if: always" in error for error in errors)
    assert any("fail closed" in error for error in errors)
    assert any("distributed CI inventory" in error for error in errors)


def test_audit_reports_missing_call_gate_required_job_and_lifted_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an incomplete coordinator and an unowned cross-category edge."""
    policy = _write_fixture(tmp_path)
    _redirect(tmp_path, monkeypatch)
    policy["stable_required_jobs"] = ["absent"]
    policy["categories"][0]["caller_needs"] = ["absent-category"]
    coordinator = tmp_path / ".github/workflows/ci.yml"
    coordinator.write_text(
        coordinator.read_text(encoding="utf-8")
        .replace("  unit-quality:\n", "  removed-call:\n")
        .replace("  ci-gate:\n", "  removed-gate:\n"),
        encoding="utf-8",
    )

    errors = modularity.audit_ci_workflow_modularity(policy)

    assert any("jobs do not match" in error for error in errors)
    assert any("missing reusable call" in error for error in errors)
    assert any("stable branch-protection" in error for error in errors)
    assert any("lifted cross-category dependencies" in error for error in errors)
    assert any("missing required gate" in error for error in errors)


def test_audit_handles_non_mapping_triggers_jobs_and_job_bodies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return deterministic findings for structurally invalid workflow mappings."""
    policy = _write_fixture(tmp_path)
    _redirect(tmp_path, monkeypatch)
    coordinator = tmp_path / ".github/workflows/ci.yml"
    coordinator.write_text(
        "name: CI\non: []\npermissions: {}\nconcurrency: []\njobs: []\n",
        encoding="utf-8",
    )

    errors = modularity.audit_ci_workflow_modularity(policy)

    assert any("trigger must be a mapping" in error for error in errors)
    assert any("concurrency differs" in error for error in errors)
    assert any("jobs must be a mapping" in error for error in errors)

    _write_fixture(tmp_path, policy)
    category = tmp_path / ".github/workflows/ci-unit-quality.yml"
    category.write_text(
        "name: CI / Unit Quality\non:\n  workflow_call:\npermissions: {}\njobs: []\n",
        encoding="utf-8",
    )
    errors = modularity.audit_ci_workflow_modularity(policy)
    assert any("jobs must be a mapping" in error for error in errors)

    _write_fixture(tmp_path, policy)
    category.write_text(
        category.read_text(encoding="utf-8").replace(
            "  unit:\n    runs-on: ubuntu-latest\n    steps:",
            "  unit: []\n  discarded:\n    runs-on: ubuntu-latest\n    steps:",
        ),
        encoding="utf-8",
    )
    errors = modularity.audit_ci_workflow_modularity(policy)
    assert any("job order or ownership" in error for error in errors)
    assert any("job unit must be a mapping" in error for error in errors)


def test_audit_reports_duplicate_job_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject one executable job claimed by two reusable categories."""
    policy = _write_fixture(tmp_path)
    duplicate: WorkflowCategory = {
        "id": "duplicate-quality",
        "workflow": ".github/workflows/ci-duplicate-quality.yml",
        "caller_needs": [],
        "secrets": [],
        "jobs": ["unit"],
    }
    policy["categories"].append(duplicate)
    duplicate_path = tmp_path / duplicate["workflow"]
    duplicate_path.write_text(
        f"""name: CI / Duplicate
on:
  workflow_call:
permissions: {{}}
jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@{_ACTION_SHA}
""",
        encoding="utf-8",
    )
    coordinator = tmp_path / ".github/workflows/ci.yml"
    coordinator.write_text(
        coordinator.read_text(encoding="utf-8")
        .replace(
            "  ci-gate:\n",
            "  duplicate-quality:\n    uses: ./.github/workflows/ci-duplicate-quality.yml\n  ci-gate:\n",
        )
        .replace("needs: [unit-quality]", "needs: [unit-quality, duplicate-quality]"),
        encoding="utf-8",
    )
    _redirect(tmp_path, monkeypatch)

    errors = modularity.audit_ci_workflow_modularity(policy)

    assert any("duplicated" in error for error in errors)


def test_low_level_parsers_reject_malformed_needs_jobs_and_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep malformed helper inputs fail-closed without masking later findings."""
    assert modularity._needs({"needs": "first"}) == ["first"]
    with pytest.raises(ValueError, match="needs must"):
        modularity._needs({"needs": [1]})
    missing_job = tmp_path / "missing.yml"
    missing_job.write_text("jobs: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="workflow job not found"):
        modularity._job_text(missing_job, "missing")

    errors: list[str] = []
    modularity._check_action_pins({"jobs": []}, tmp_path / "workflow.yml", errors)
    assert errors == [f"{tmp_path / 'workflow.yml'}: jobs must be a mapping"]

    errors = []
    modularity._check_action_pins(
        {
            "jobs": {
                "not-a-job": [],
                "odd-steps": {"steps": "not-a-list"},
                "local-call": {"uses": "./.github/workflows/local.yml"},
            }
        },
        tmp_path / "workflow.yml",
        errors,
    )
    assert errors == []

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    monkeypatch.setattr(modularity, "REPOSITORY_ROOT", empty_root)
    modularity._check_no_direct_coordinator_readers(errors)


def test_low_level_evidence_inventory_ignores_non_artifact_steps() -> None:
    """Classify only named upload and download steps as artifact contracts."""
    workflows = [
        {
            "jobs": {
                "probe": {
                    "steps": [
                        None,
                        {"uses": 1},
                        {"uses": "action@example", "with": []},
                        {"uses": "action@example", "with": {"name": 1}},
                        {"uses": "action@example", "with": {"name": "ignored"}},
                        {"name": "conditional without if"},
                        {"if": "always()"},
                    ]
                }
            }
        }
    ]

    assert modularity._artifact_inventory(workflows) == {"ignored": {"producer": None, "consumers": []}}
    assert modularity._conditional_steps(workflows) == {}
    assert modularity._step_output_consumers(workflows) == {}


def test_direct_script_entrypoint_and_failure_output_are_exercised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Cover both script execution and the diagnostic nonzero main path."""
    script = Path(modularity.__file__)
    with pytest.raises(SystemExit) as script_exit:
        runpy.run_path(str(script), run_name="__main__")
    assert script_exit.value.code == 0
    capsys.readouterr()

    policy = _write_fixture(tmp_path)
    _redirect(tmp_path, monkeypatch)
    policy["schema_version"] = 2
    monkeypatch.setattr(modularity, "load_ci_workflow_policy", lambda: policy)

    assert modularity.main() == 1
    assert "schema_version" in capsys.readouterr().out


def test_audit_rejects_non_mapping_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject syntactically valid YAML that is not a workflow mapping."""
    policy = _write_fixture(tmp_path)
    _redirect(tmp_path, monkeypatch)
    (tmp_path / ".github/workflows/ci-unit-quality.yml").write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="workflow must be a mapping"):
        modularity.audit_ci_workflow_modularity(policy)


def test_live_lint_job_installs_checker_runtime_and_typing_dependencies() -> None:
    """Keep the YAML-backed checker reproducible in lint and strict-typing jobs."""
    root = inventory.REPOSITORY_ROOT
    workflow = (root / ".github/workflows/ci-static-governance.yml").read_text(encoding="utf-8")
    lint_input = (root / "requirements/ci-lint.in").read_text(encoding="utf-8")
    lint_lock = (root / "requirements/ci-lint.txt").read_text(encoding="utf-8")
    test_input = (root / "requirements/ci-test.in").read_text(encoding="utf-8")

    assert "pip install --require-hashes -r requirements/ci-lint.txt" in workflow
    assert "python tools/check_ci_workflow_modularity.py" in workflow
    assert "PyYAML==6.0.3" in lint_input
    assert "pyyaml==6.0.3" in lint_lock
    assert "types-PyYAML==6.0.12.20260815" in test_input
