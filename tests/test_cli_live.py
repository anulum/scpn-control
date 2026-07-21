# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI live command tests
"""Tests for the ``live`` CLI command (phase-sync monitor + WebSocket server wiring)."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from scpn_control.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_live_command_wires_monitor_and_server(runner, monkeypatch):
    import scpn_control.phase.realtime_monitor as realtime_monitor
    import scpn_control.phase.ws_phase_stream as ws_phase_stream

    calls = {}

    class FakeMonitor:
        @classmethod
        def from_paper27(cls, *, L, N_per, zeta_uniform, psi_driver):
            calls["monitor"] = {
                "L": L,
                "N_per": N_per,
                "zeta_uniform": zeta_uniform,
                "psi_driver": psi_driver,
            }
            return "monitor"

    class FakeServer:
        def __init__(
            self,
            *,
            monitor,
            tick_interval_s,
            api_key,
            command_rate_limit,
            command_rate_window_s,
            max_payload_bytes,
            max_client_write_buffer_bytes,
            require_client_auth,
            allow_query_token_auth,
            require_tls,
            allow_insecure_remote,
            allowed_origins,
            allowed_actions,
        ):
            calls["server_init"] = {
                "monitor": monitor,
                "tick_interval_s": tick_interval_s,
                "api_key": api_key,
                "command_rate_limit": command_rate_limit,
                "command_rate_window_s": command_rate_window_s,
                "max_payload_bytes": max_payload_bytes,
                "max_client_write_buffer_bytes": max_client_write_buffer_bytes,
                "require_client_auth": require_client_auth,
                "allow_query_token_auth": allow_query_token_auth,
                "require_tls": require_tls,
                "allow_insecure_remote": allow_insecure_remote,
                "allowed_origins": allowed_origins,
                "allowed_actions": allowed_actions,
            }

        def serve_sync(self, *, host, port, ssl_context):
            calls["serve"] = {"host": host, "port": port, "ssl_context": ssl_context}

    monkeypatch.setattr(realtime_monitor, "RealtimeMonitor", FakeMonitor)
    monkeypatch.setattr(ws_phase_stream, "PhaseStreamServer", FakeServer)

    result = runner.invoke(
        main,
        [
            "live",
            "--host",
            "127.0.0.1",
            "--port",
            "9001",
            "--layers",
            "3",
            "--n-per",
            "4",
            "--zeta",
            "0.7",
            "--psi",
            "0.2",
            "--tick-interval",
            "0.005",
            "--api-key",
            "secret-token-123456",
            "--max-payload-bytes",
            "1234",
            "--max-client-write-buffer-bytes",
            "8192",
            "--allow-query-token-auth",
            "--require-tls",
            "--allow-insecure-remote",
            "--allowed-origin",
            "https://ops.example",
            "--allowed-action",
            "set_psi",
            "--allowed-action",
            "stop",
        ],
    )

    assert result.exit_code == 0
    assert calls["monitor"] == {"L": 3, "N_per": 4, "zeta_uniform": 0.7, "psi_driver": 0.2}
    assert calls["server_init"] == {
        "monitor": "monitor",
        "tick_interval_s": 0.005,
        "api_key": "secret-token-123456",
        "command_rate_limit": 20,
        "command_rate_window_s": 1.0,
        "max_payload_bytes": 1234,
        "max_client_write_buffer_bytes": 8192,
        "require_client_auth": True,
        "allow_query_token_auth": True,
        "require_tls": True,
        "allow_insecure_remote": True,
        "allowed_origins": ("https://ops.example",),
        "allowed_actions": ("set_psi", "stop"),
    }
    assert calls["serve"] == {"host": "127.0.0.1", "port": 9001, "ssl_context": None}
    assert "Starting phase sync server on ws://127.0.0.1:9001" in result.output


def test_live_help(runner):
    result = runner.invoke(main, ["live", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.output
    assert "--zeta" in result.output
