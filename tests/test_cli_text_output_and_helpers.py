# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI text-output branches and option helpers
"""Text-mode output branches and option helpers of the SCPN-CONTROL CLI.

Exercises the reactor-reference validation command family in text mode (the
fail summary, the per-error console loop, and the non-zero exit), the
`_coerce_float` option parser, the `_emit_campaign_result` console renderer, and
the empty-sample percentile branch of the timing benchmark.
"""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from scpn_control.cli import _coerce_float, _emit_campaign_result, main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ── reactor-reference validation commands (text mode) ─────────────────


_REFERENCE_COMMANDS = [
    ("validate-current-drive-reference", "Current-drive reference: fail"),
    ("validate-mu-synthesis-reference", "Mu-synthesis reference: fail"),
    ("validate-volt-second-reference", "Volt-second reference: fail"),
    ("validate-burn-reference", "Burn reference: fail"),
    ("validate-free-boundary-reference", "Free-boundary reference: fail"),
]


@pytest.mark.parametrize(("command", "summary"), _REFERENCE_COMMANDS)
def test_reference_command_text_failure_path(runner, tmp_path, command, summary):
    result = runner.invoke(main, [command, "--artifact-root", str(tmp_path), "--require-reference-artifacts"])
    assert result.exit_code == 1
    assert summary in result.output
    assert "reference_artifacts=0" in result.output


@pytest.mark.parametrize(("command", "_summary"), _REFERENCE_COMMANDS)
def test_reference_command_json_out_and_report_file(runner, tmp_path, command, _summary):
    out = tmp_path / "nested" / "report.json"
    result = runner.invoke(
        main,
        [
            command,
            "--artifact-root",
            str(tmp_path),
            "--require-reference-artifacts",
            "--json-out",
            "--output-json",
            str(out),
        ],
    )
    assert result.exit_code == 1
    assert out.is_file()
    assert '"status"' in out.read_text(encoding="utf-8")
    assert '"status"' in result.output  # --json-out echoes the report to stdout


# ── _coerce_float ─────────────────────────────────────────────────────


class TestCoerceFloat:
    def test_returns_none_for_none(self):
        assert _coerce_float("itpa_cgb", None) is None

    def test_parses_valid_float(self):
        assert _coerce_float("itpa_cgb", "1.5") == 1.5

    def test_rejects_non_numeric(self):
        with pytest.raises(click.ClickException, match="invalid float for itpa_cgb"):
            _coerce_float("itpa_cgb", "not-a-number")


# ── _emit_campaign_result ─────────────────────────────────────────────


class TestEmitCampaignResult:
    def test_json_mode_emits_json(self, capsys):
        _emit_campaign_result({"status": "normal", "steps": 3}, json_out=True)
        out = capsys.readouterr().out
        assert '"status": "normal"' in out
        assert '"steps": 3' in out

    def test_text_mode_renders_execution_native_and_admission(self, capsys):
        summary = {
            "status": "normal",
            "execution": {"mode": "rust"},
            "target_r": 6.2,
            "target_z": 0.0,
            "steps": 10,
            "elapsed_s": 1.25,
            "native": {"ticks": 10, "halts": 0},
            "runtime_admission": {"status": "qualified", "production_claim_allowed": True},
        }
        _emit_campaign_result(summary, json_out=False)
        out = capsys.readouterr().out
        assert "Hardware campaign completed: normal" in out
        assert "mode=rust" in out
        assert "Native summary keys:" in out
        assert "Runtime admission: qualified" in out

    def test_text_mode_reports_anomaly_status_on_stderr(self, capsys):
        _emit_campaign_result({"status": "runtime_aborted"}, json_out=False)
        captured = capsys.readouterr()
        assert "Hardware campaign completed: runtime_aborted" in captured.out
        assert "Anomaly details: status=runtime_aborted" in captured.err


# ── run-hardware-campaign (mocked native execution plane) ─────────────


class _FakeEngine:
    """Records configuration calls and replays a scripted hardware loop."""

    loop_behaviour: object = "normal"  # "normal" | RuntimeError | KeyboardInterrupt
    native_backend_available = True

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _noop(self, *args, **kwargs):
        return None

    configure_acados_targets = _noop
    configure_itpa_gyro_bohm = _noop
    configure_kuramoto_weights = _noop
    configure_transport = _noop
    configure_execution_affinity = _noop
    configure_native_formal_verification = _noop
    set_max_publish_failures = _noop

    def execute_hardware_loop(self, **kwargs):
        behaviour = type(self).loop_behaviour
        if behaviour == "runtime":
            raise RuntimeError("runtime safety abort")
        if behaviour == "keyboard":
            raise KeyboardInterrupt
        return {"status": "normal", "execution": {"mode": "rust"}, "steps": kwargs.get("steps")}

    emergency_result: object = {"shutdown": "ok"}

    def execute_emergency_shutdown(self):
        return type(self).emergency_result


@pytest.fixture
def fake_engine(monkeypatch):
    import scpn_control.core.rust_engine as rust_engine

    _FakeEngine.loop_behaviour = "normal"
    _FakeEngine.native_backend_available = True
    _FakeEngine.emergency_result = {"shutdown": "ok"}
    monkeypatch.setattr(rust_engine, "NeuroCyberneticEngine", _FakeEngine)
    return _FakeEngine


def _campaign_args(*extra: str) -> list[str]:
    return ["run-hardware-campaign", "--steps", "4", *extra]


def test_campaign_emits_json_summary(runner, fake_engine):
    result = runner.invoke(main, _campaign_args("--json-out", "--itpa-cgb", "1.5", "--max-publish-failures", "2"))
    assert result.exit_code == 0
    assert '"status": "normal"' in result.output


def test_campaign_text_summary_with_file_options(runner, fake_engine, tmp_path):
    weights = tmp_path / "weights.json"
    weights.write_text("[1.0, 2.0]", encoding="utf-8")
    itpa = tmp_path / "itpa.json"
    itpa.write_text("{}", encoding="utf-8")
    result = runner.invoke(main, _campaign_args("--kuramoto-weights", str(weights), "--itpa-path", str(itpa)))
    assert result.exit_code == 0
    assert "Hardware campaign completed: normal" in result.output


def test_campaign_requires_native_backend_when_flagged(runner, fake_engine):
    fake_engine.native_backend_available = False
    result = runner.invoke(main, _campaign_args("--require-native"))
    assert result.exit_code != 0
    assert "Native bridge not available" in result.output


def test_campaign_rejects_non_numeric_itpa_cgb(runner, fake_engine):
    result = runner.invoke(main, _campaign_args("--itpa-cgb", "not-a-number"))
    assert result.exit_code != 0
    assert "invalid float for itpa_cgb" in result.output


@pytest.mark.parametrize("json_out", [True, False])
def test_campaign_runtime_abort_triggers_emergency_shutdown(runner, fake_engine, json_out):
    fake_engine.loop_behaviour = "runtime"
    args = _campaign_args("--json-out") if json_out else _campaign_args()
    result = runner.invoke(main, args)
    assert result.exit_code == 0  # the abort is handled, not surfaced as a CLI error
    combined = result.output
    if json_out:
        assert "runtime_aborted" in combined
    else:
        assert "Campaign aborted" in combined


@pytest.mark.parametrize("json_out", [True, False])
def test_campaign_keyboard_interrupt_triggers_emergency_shutdown(runner, fake_engine, json_out):
    fake_engine.loop_behaviour = "keyboard"
    args = _campaign_args("--json-out") if json_out else _campaign_args()
    result = runner.invoke(main, args)
    assert result.exit_code == 0
    if json_out:
        assert "keyboard_interrupt" in result.output
    else:
        assert "interrupted by operator" in result.output


# ── validate command: text-mode per-section error console loops ───────


def test_validate_text_mode_renders_section_error_loops(runner, tmp_path):
    registry = tmp_path / "physics_traceability.json"
    registry.write_text('{"schema_version":"1.0","entries":[]}', encoding="utf-8")
    bad_report = tmp_path / "bad.json"
    bad_report.write_text("{}", encoding="utf-8")
    result = runner.invoke(
        main,
        [
            "validate",
            "--data-manifest-root",
            str(tmp_path),
            "--jax-gk-parity-root",
            str(tmp_path),
            "--physics-traceability-registry",
            str(registry),
            "--multi-shot-campaign-python-report",
            str(bad_report),
            "--runtime-admission-report",
            str(bad_report),
            "--native-formal-certificate-report",
            str(bad_report),
        ],
    )
    assert result.exit_code == 1
    assert "Status: fail" in result.output
    # every gated section prints its fail summary line in text mode
    assert "JAX GK parity: fail" in result.output
    assert "Physics traceability: fail" in result.output


# ── validate-release-evidence (text mode) ─────────────────────────────


def test_validate_release_evidence_text_failure(runner, tmp_path):
    report = tmp_path / "release.json"
    report.write_text("{}", encoding="utf-8")
    result = runner.invoke(main, ["validate-release-evidence", str(report)])
    assert result.exit_code == 1
    assert "Release evidence: fail" in result.output


def test_validate_release_evidence_text_failure_without_sha_when_report_unparsable(runner, tmp_path):
    """An unparsable report fails before a digest is computed, so the SHA line is skipped (arc 302->304)."""
    report = tmp_path / "release.json"
    report.write_text("{ this is not valid json", encoding="utf-8")
    result = runner.invoke(main, ["validate-release-evidence", str(report)])
    assert result.exit_code == 1
    assert "Release evidence: fail" in result.output
    assert "Report SHA-256:" not in result.output


# ── phase-sync-server TLS argument validation ─────────────────────────


def test_live_server_rejects_partial_tls(runner, tmp_path):
    cert = tmp_path / "cert.pem"
    cert.write_text("dummy", encoding="utf-8")
    result = runner.invoke(main, ["live", "--tls-cert", str(cert)])
    assert result.exit_code != 0
    assert "--tls-cert and --tls-key must be provided together" in result.output


def test_live_server_loads_tls_material(runner, tmp_path):
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("not-a-real-cert", encoding="utf-8")
    key.write_text("not-a-real-key", encoding="utf-8")
    # Both provided: the command reaches ssl.SSLContext.load_cert_chain, which
    # rejects the bogus material before the blocking server loop starts.
    result = runner.invoke(main, ["live", "--tls-cert", str(cert), "--tls-key", str(key)])
    assert result.exit_code != 0
    assert result.exception is not None


# ── remaining campaign / validate defensive branches ──────────────────


def test_campaign_emergency_shutdown_non_dict_is_normalised(runner, fake_engine):
    fake_engine.loop_behaviour = "runtime"
    fake_engine.emergency_result = None  # non-dict telemetry must be coerced to {}
    result = runner.invoke(main, _campaign_args("--json-out"))
    assert result.exit_code == 0
    assert '"emergency": {}' in result.output


def test_campaign_itpa_cgb_coerced_to_none_is_rejected(runner, fake_engine, monkeypatch):
    import scpn_control.cli as cli_module

    monkeypatch.setattr(cli_module, "_coerce_float", lambda name, value: None)
    result = runner.invoke(main, _campaign_args("--itpa-cgb", "1.0"))
    assert result.exit_code != 0
    assert "--itpa-cgb must be numeric" in result.output


@pytest.mark.parametrize(
    ("module_name", "summary"),
    [
        ("validation.validate_multi_shot_campaign_evidence", "Multi-shot campaign evidence: fail"),
        ("validation.validate_runtime_admission_evidence", "Runtime admission evidence: fail"),
    ],
)
def test_validate_section_failure_text_loop_via_mocked_validator(runner, monkeypatch, module_name, summary):
    import importlib

    module = importlib.import_module(module_name)
    func_name = next(n for n in dir(module) if n.startswith("validate_") and n.endswith("evidence"))

    class _FailResult:
        def as_dict(self):
            return {"status": "fail", "errors": ["mocked section failure"]}

    monkeypatch.setattr(module, func_name, lambda *args, **kwargs: _FailResult())
    result = runner.invoke(main, ["validate", "--no-data-manifests", "--no-jax-gk-parity", "--no-physics-traceability"])
    assert result.exit_code == 1
    assert summary in result.output
    assert "mocked section failure" in result.output
