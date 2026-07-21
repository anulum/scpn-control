# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — WebSocket Phase Stream Tests
"""Regression tests for ws_phase_stream: handler, tick loop, serve."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import ssl
import sys
import time
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

import scpn_control.phase.ws_phase_stream as ws_phase_stream
from scpn_control.phase.realtime_monitor import RealtimeMonitor
from scpn_control.phase.ws_phase_stream import (
    WEBSOCKET_RUNTIME_EVIDENCE_LOCAL_ONLY,
    PhaseStreamServer,
    assert_websocket_runtime_claim_admissible,
    load_websocket_runtime_evidence,
    save_websocket_runtime_evidence,
    websocket_runtime_evidence,
)


def _make_monitor():
    return RealtimeMonitor.from_paper27(L=4, N_per=10, zeta_uniform=0.5, psi_driver=0.0)


_AUTH_HEADERS = {"Authorization": "Bearer secret-token-123456"}


class _FakeWS:
    """Mock WebSocket for testing handler and tick loop."""

    def __init__(self, messages=None, headers=None, path="/"):
        self._messages = list(messages or [])
        self._sent: list[str] = []
        self.request_headers = _AUTH_HEADERS.copy() if headers is None else headers
        self.path = path
        self._idx = 0
        self.closed = False
        self.close_code = None
        self.close_reason = None
        self.remote_address = ("127.0.0.1", 50123)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._messages):
            raise StopAsyncIteration
        msg = self._messages[self._idx]
        self._idx += 1
        return msg

    async def send(self, data):
        self._sent.append(data)

    async def close(self, code=1000, reason=""):
        self.closed = True
        self.close_code = code
        self.close_reason = reason


class _CaseInsensitiveHeaders:
    """Case-insensitive header mapping mirroring the websockets>=15 ``Headers``.

    The real :class:`websockets.datastructures.Headers` resolves ``get`` without
    regard to case; reproducing that here lets the modern-API tests exercise the
    fixed auth path without installing the optional ``ws`` extra (CI omits it).
    """

    def __init__(self, items):
        self._items = {key.lower(): value for key, value in dict(items).items()}

    def get(self, name):
        return self._items.get(name.lower())


class _ModernWS(_FakeWS):
    """Mock WebSocket in the websockets>=15 connection shape.

    websockets >= 15 removed ``request_headers``/``path`` from the connection and
    moved the opening handshake onto ``connection.request`` (headers + path). This
    fake carries only that modern surface — the legacy attributes are deleted — so
    the auth path is exercised exactly the way a real ``ServerConnection`` presents
    it, guarding against silent 100% auth denial on a fresh install.
    """

    def __init__(self, messages=None, headers=None, path="/"):
        super().__init__(messages=messages, headers=headers, path=path)
        request_headers = _AUTH_HEADERS if headers is None else headers
        self.request = SimpleNamespace(headers=_CaseInsensitiveHeaders(request_headers), path=path)
        del self.request_headers
        del self.path


class TestPhaseStreamServer:
    def test_init(self):
        mon = _make_monitor()
        server = PhaseStreamServer(monitor=mon, tick_interval_s=0.01)
        assert server.tick_interval_s == 0.01
        assert server.api_key is None
        assert server.command_rate_limit == 20
        assert server.max_payload_bytes == 65536
        assert server.client_send_timeout_s == 0.25
        assert server.max_clients == 128
        assert server.require_client_auth is True
        assert server.allow_query_token_auth is False
        assert server.require_tls is False
        assert server.allow_insecure_remote is False
        assert server.allowed_origins == ()
        assert server.allowed_actions == ("set_psi", "set_pac_gamma", "reset", "stop")

    @pytest.mark.parametrize("tick_interval_s", [0.0, -1.0, math.nan, math.inf])
    def test_init_rejects_nonpositive_or_nonfinite_tick_interval(self, tick_interval_s):
        mon = _make_monitor()
        with pytest.raises(ValueError, match="tick_interval_s"):
            PhaseStreamServer(monitor=mon, tick_interval_s=tick_interval_s)

    def test_init_rejects_invalid_security_domains(self):
        mon = _make_monitor()
        with pytest.raises(ValueError, match="command_rate_limit"):
            PhaseStreamServer(monitor=mon, command_rate_limit=0)
        with pytest.raises(ValueError, match="command_rate_window_s"):
            PhaseStreamServer(monitor=mon, command_rate_window_s=0.0)
        with pytest.raises(ValueError, match="max_payload_bytes"):
            PhaseStreamServer(monitor=mon, max_payload_bytes=0)
        with pytest.raises(ValueError, match="max_payload_bytes"):
            PhaseStreamServer(monitor=mon, max_payload_bytes=True)
        with pytest.raises(ValueError, match="client_send_timeout_s"):
            PhaseStreamServer(monitor=mon, client_send_timeout_s=0.0)
        with pytest.raises(ValueError, match="client_send_timeout_s"):
            PhaseStreamServer(monitor=mon, client_send_timeout_s=math.inf)
        with pytest.raises(ValueError, match="max_clients"):
            PhaseStreamServer(monitor=mon, max_clients=0)
        with pytest.raises(ValueError, match="max_clients"):
            PhaseStreamServer(monitor=mon, max_clients=True)
        with pytest.raises(ValueError, match="max_client_write_buffer_bytes"):
            PhaseStreamServer(monitor=mon, max_client_write_buffer_bytes=0)
        with pytest.raises(ValueError, match="require_client_auth"):
            PhaseStreamServer(monitor=mon, require_client_auth="yes")
        with pytest.raises(ValueError, match="allow_query_token_auth"):
            PhaseStreamServer(monitor=mon, allow_query_token_auth="yes")
        with pytest.raises(ValueError, match="require_tls"):
            PhaseStreamServer(monitor=mon, require_tls="yes")
        with pytest.raises(ValueError, match="allow_insecure_remote"):
            PhaseStreamServer(monitor=mon, allow_insecure_remote="yes")
        with pytest.raises(ValueError, match="api_key"):
            PhaseStreamServer(monitor=mon, api_key="short")
        with pytest.raises(ValueError, match="allowed_origins"):
            PhaseStreamServer(monitor=mon, allowed_origins=("",))
        with pytest.raises(ValueError, match="allowed_actions"):
            PhaseStreamServer(monitor=mon, allowed_actions=())
        with pytest.raises(ValueError, match="allowed_actions"):
            PhaseStreamServer(monitor=mon, allowed_actions=("shell",))

    def test_init_accepts_numpy_integral_security_limits(self):
        mon = _make_monitor()
        server = PhaseStreamServer(
            monitor=mon,
            command_rate_limit=np.int64(3),
            max_payload_bytes=np.int64(4096),
            max_clients=np.int64(8),
            max_client_write_buffer_bytes=np.int64(8192),
        )

        assert server.command_rate_limit == 3
        assert server.max_payload_bytes == 4096
        assert server.max_clients == 8
        assert server.max_client_write_buffer_bytes == 8192

    def test_handler_rejects_unauthenticated_control_connection_when_api_key_configured(self, caplog):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
            ws = _FakeWS([json.dumps({"action": "set_psi", "value": 0.5})], headers={})

            with caplog.at_level(logging.WARNING, logger=ws_phase_stream.__name__):
                await server._handler(ws)

            assert mon.psi_driver == pytest.approx(0.0)
            assert ws.closed
            assert ws.close_code == 1008
            assert "authentication" in (ws.close_reason or "")
            assert ws not in server._clients
            assert "phase_stream_security_event event=authentication_failed" in caplog.text

        asyncio.run(_run())

    def test_handler_rejects_malformed_bearer_scheme_without_mutation(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
            ws = _FakeWS(
                [json.dumps({"action": "set_psi", "value": 0.5})],
                headers={"Authorization": "Basic secret-token-123456"},
            )

            await server._handler(ws)

            assert mon.psi_driver == pytest.approx(0.0)
            assert ws.closed
            assert ws.close_code == 1008
            assert "authentication" in (ws.close_reason or "")

        asyncio.run(_run())

    def test_origin_check_treats_non_mapping_headers_as_no_origin(self):
        mon = _make_monitor()
        server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
        ws = _FakeWS(headers=object())

        assert server._origin_allowed(ws) is True

    def test_rate_limiter_uses_token_bucket_without_fixed_window_burst(self, monkeypatch):
        mon = _make_monitor()
        server = PhaseStreamServer(
            monitor=mon,
            api_key="secret-token-123456",
            command_rate_limit=20,
            command_rate_window_s=1.0,
        )
        ws = _FakeWS()
        times = iter([0.999] * 20 + [1.001])
        monkeypatch.setattr(time, "monotonic", lambda: next(times))

        for _ in range(20):
            assert server._rate_limited(ws) is False
        assert server._rate_limited(ws) is True

    def test_rate_limiter_applies_peer_bucket_across_connections(self):
        mon = _make_monitor()
        server = PhaseStreamServer(
            monitor=mon,
            api_key="secret-token-123456",
            command_rate_limit=2,
            command_rate_window_s=10.0,
        )
        first = _FakeWS()
        second = _FakeWS()
        first.remote_address = ("10.1.2.3", 10000)
        second.remote_address = ("10.1.2.3", 10001)

        assert server._rate_limited(first) is False
        assert server._rate_limited(first) is False
        assert server._rate_limited(second) is True

    def test_handler_rejects_browser_origin_without_allowlist(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
            ws = _FakeWS(
                [json.dumps({"action": "set_psi", "value": 0.5})],
                headers={"Authorization": "Bearer secret-token-123456", "Origin": "https://evil.invalid"},
            )

            await server._handler(ws)

            assert mon.psi_driver == pytest.approx(0.0)
            assert ws.closed
            assert ws.close_code == 1008
            assert "origin" in (ws.close_reason or "")

        asyncio.run(_run())

    def test_handler_accepts_configured_browser_origin(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(
                monitor=mon,
                api_key="secret-token-123456",
                allowed_origins=("https://ops.example",),
            )
            ws = _FakeWS(
                [json.dumps({"action": "set_psi", "value": 0.5})],
                headers={"Authorization": "Bearer secret-token-123456", "Origin": "https://ops.example"},
            )

            await server._handler(ws)

            assert mon.psi_driver == pytest.approx(0.5)
            assert not ws.closed

        asyncio.run(_run())

    @pytest.mark.parametrize(
        "headers",
        [
            {"Authorization": "Bearer secret-token-123456"},
            {"X-SCPN-API-Key": "secret-token-123456"},
            {"authorization": "Bearer secret-token-123456"},
        ],
    )
    def test_handler_accepts_authenticated_control_connection(self, headers):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
            ws = _FakeWS([json.dumps({"action": "set_psi", "value": 0.5})], headers=headers)

            await server._handler(ws)

            assert mon.psi_driver == pytest.approx(0.5)
            assert not ws.closed

        asyncio.run(_run())

    def test_handler_rejects_clients_beyond_configured_backpressure_capacity(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456", max_clients=1)
            admitted = _FakeWS()
            excess = _FakeWS([json.dumps({"action": "set_psi", "value": 0.5})])
            server._clients.add(admitted)

            await server._handler(excess)

            assert mon.psi_driver == pytest.approx(0.0)
            assert excess.closed
            assert excess.close_code == 1013
            assert "capacity" in (excess.close_reason or "")
            assert excess not in server._clients
            assert admitted in server._clients

        asyncio.run(_run())

    def test_handler_allows_observer_connection_when_auth_explicitly_disabled(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, require_client_auth=False)
            ws = _FakeWS([json.dumps({"action": "set_psi", "value": 0.5})], headers={})

            await server._handler(ws)

            assert mon.psi_driver == pytest.approx(0.5)
            assert not ws.closed

        asyncio.run(_run())

    def test_handler_accepts_query_token_for_authentication(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(
                monitor=mon,
                api_key="secret-token-123456",
                allow_query_token_auth=True,
            )
            ws = _FakeWS(
                [json.dumps({"action": "set_pac_gamma", "value": 0.2})],
                headers={},
                path="/phase?token=secret-token-123456",
            )

            await server._handler(ws)

            assert mon.pac_gamma == pytest.approx(0.2)

        asyncio.run(_run())

    def test_handler_accepts_access_token_query_parameter_when_enabled(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(
                monitor=mon,
                api_key="secret-token-123456",
                allow_query_token_auth=True,
            )
            ws = _FakeWS(
                [json.dumps({"action": "set_psi", "value": 0.4})],
                headers={},
                path="/phase?access_token=secret-token-123456",
            )

            await server._handler(ws)

            assert mon.psi_driver == pytest.approx(0.4)

        asyncio.run(_run())

    def test_handler_rejects_query_token_when_not_explicitly_enabled(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
            ws = _FakeWS(
                [json.dumps({"action": "set_pac_gamma", "value": 0.2})],
                headers={},
                path="/phase?token=secret-token-123456",
            )

            await server._handler(ws)

            assert mon.pac_gamma == pytest.approx(0.0)
            assert ws.closed
            assert ws.close_code == 1008

        asyncio.run(_run())

    def test_handler_rate_limits_control_commands_without_state_mutation(self, monkeypatch):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(
                monitor=mon, api_key="secret-token-123456", command_rate_limit=1, command_rate_window_s=60.0
            )
            ws = _FakeWS(
                [
                    json.dumps({"action": "set_psi", "value": 0.1}),
                    json.dumps({"action": "set_psi", "value": 0.9}),
                ]
            )
            monkeypatch.setattr(time, "monotonic", lambda: 100.0)

            await server._handler(ws)

            assert mon.psi_driver == pytest.approx(0.1)
            assert json.loads(ws._sent[-1])["error"] == "rate_limited"
            assert ws.closed
            assert ws.close_code == 1008

        asyncio.run(_run())

    def test_handler_rejects_disallowed_action_without_state_mutation(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(
                monitor=mon,
                api_key="secret-token-123456",
                allowed_actions=("stop",),
            )
            ws = _FakeWS([json.dumps({"action": "set_psi", "value": 0.9})])

            await server._handler(ws)

            assert mon.psi_driver == pytest.approx(0.0)
            assert json.loads(ws._sent[-1])["error"] == "action_not_allowed"
            assert not ws.closed

        asyncio.run(_run())

    def test_handler_drops_client_when_control_response_backpressures(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(
                monitor=mon,
                api_key="secret-token-123456",
                client_send_timeout_s=0.001,
                allowed_actions=("stop",),
            )

            class _SlowResponseWS(_FakeWS):
                async def send(self, data):
                    await asyncio.sleep(10.0)

            ws = _SlowResponseWS([json.dumps({"action": "set_psi", "value": 0.9})])

            await asyncio.wait_for(server._handler(ws), timeout=0.05)

            assert mon.psi_driver == pytest.approx(0.0)
            assert ws.closed
            assert ws.close_code == 1011
            assert "backpressure" in (ws.close_reason or "")
            assert ws not in server._clients

        asyncio.run(_run())

    def test_send_frame_drops_client_when_transport_buffer_exceeds_limit(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(
                monitor=mon,
                api_key="secret-token-123456",
                max_client_write_buffer_bytes=64,
            )

            class _BufferedTransport:
                def get_write_buffer_size(self):
                    return 128

            class _BufferedWS(_FakeWS):
                transport = _BufferedTransport()

            ws = _BufferedWS()

            dead = await server._send_frame_or_dead(ws, json.dumps({"R_global": 1.0}))

            assert dead is ws
            assert ws.closed
            assert ws.close_code == 1011
            assert "backpressure" in (ws.close_reason or "")
            assert server.runtime_counters()["backpressure_disconnects"] == 1

        asyncio.run(_run())

    def test_send_frame_keeps_client_when_transport_buffer_within_limit(self):
        """A transport buffer at or below the limit keeps the client (arc 498->506)."""

        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(
                monitor=mon,
                api_key="secret-token-123456",
                max_client_write_buffer_bytes=64,
            )

            class _SmallBufferTransport:
                def get_write_buffer_size(self):
                    return 32  # <= max_client_write_buffer_bytes -> no backpressure

            class _SmallBufferWS(_FakeWS):
                transport = _SmallBufferTransport()

            ws = _SmallBufferWS()

            dead = await server._send_frame_or_dead(ws, json.dumps({"R_global": 1.0}))

            assert dead is None
            assert not ws.closed
            assert server.runtime_counters()["backpressure_disconnects"] == 0

        asyncio.run(_run())

    def test_handler_rate_limits_invalid_json_before_parsing(self, monkeypatch):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(
                monitor=mon,
                api_key="secret-token-123456",
                command_rate_limit=1,
                command_rate_window_s=60.0,
            )
            ws = _FakeWS(["not-json", json.dumps({"action": "set_psi", "value": 0.9})])
            monkeypatch.setattr(time, "monotonic", lambda: 100.0)

            await server._handler(ws)

            assert mon.psi_driver == pytest.approx(0.0)
            assert json.loads(ws._sent[-1])["error"] == "rate_limited"
            assert ws.closed

        asyncio.run(_run())

    def test_handler_rejects_oversized_payload_without_state_mutation(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456", max_payload_bytes=32)
            ws = _FakeWS([json.dumps({"action": "set_psi", "value": 0.9, "pad": "x" * 64})])

            await server._handler(ws)

            assert mon.psi_driver == pytest.approx(0.0)
            assert json.loads(ws._sent[-1])["error"] == "payload_too_large"
            assert ws.closed
            assert ws.close_code == 1009

        asyncio.run(_run())

    def test_handler_rejects_oversized_binary_payload_without_json_parsing(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456", max_payload_bytes=4)
            ws = _FakeWS([b'{"action":"set_psi","value":0.9}'])

            await server._handler(ws)

            assert mon.psi_driver == pytest.approx(0.0)
            assert json.loads(ws._sent[-1])["error"] == "payload_too_large"
            assert ws.closed
            assert ws.close_code == 1009

        asyncio.run(_run())

    def test_handler_set_psi(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
            ws = _FakeWS([json.dumps({"action": "set_psi", "value": 0.5})])
            await server._handler(ws)
            assert mon.psi_driver == pytest.approx(0.5)
            assert ws not in server._clients

        asyncio.run(_run())

    def test_handler_ignores_non_object_json_commands(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
            ws = _FakeWS([json.dumps(["set_psi", 0.9]), json.dumps({"action": "stop"})])

            await server._handler(ws)

            assert mon.psi_driver == pytest.approx(0.0)
            assert server._running is False

        asyncio.run(_run())

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "not-a-number"])
    def test_handler_rejects_invalid_psi_commands_without_state_mutation(self, value):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
            ws = _FakeWS(
                [
                    json.dumps({"action": "set_psi", "value": value}),
                    json.dumps({"action": "stop"}),
                ]
            )
            await server._handler(ws)
            assert mon.psi_driver == pytest.approx(0.0)
            assert server._running is False

        asyncio.run(_run())

    def test_handler_set_pac_gamma(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
            ws = _FakeWS([json.dumps({"action": "set_pac_gamma", "value": 0.3})])
            await server._handler(ws)
            assert mon.pac_gamma == pytest.approx(0.3)

        asyncio.run(_run())

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "not-a-number"])
    def test_handler_rejects_invalid_pac_gamma_commands_without_state_mutation(self, value):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
            ws = _FakeWS(
                [
                    json.dumps({"action": "set_pac_gamma", "value": value}),
                    json.dumps({"action": "stop"}),
                ]
            )
            await server._handler(ws)
            assert mon.pac_gamma == pytest.approx(0.0)
            assert server._running is False

        asyncio.run(_run())

    def test_handler_reset(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
            ws = _FakeWS([json.dumps({"action": "reset", "seed": 99})])
            await server._handler(ws)

        asyncio.run(_run())

    def test_handler_stop(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
            ws = _FakeWS([json.dumps({"action": "stop"})])
            await server._handler(ws)
            assert server._running is False

        asyncio.run(_run())

    def test_handler_bad_json_ignored(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
            ws = _FakeWS(["not-json", json.dumps({"action": "stop"})])
            await server._handler(ws)

        asyncio.run(_run())

    def test_handler_reports_malformed_json_frame(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
            ws = _FakeWS(["not-json", json.dumps({"action": "stop"})])
            await server._handler(ws)
            errors = [json.loads(sent).get("error") for sent in ws._sent]
            assert "malformed_frame" in errors
            assert server.runtime_counters()["malformed_frame_rejections"] == 1

        asyncio.run(_run())

    def test_handler_reports_non_object_frame(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
            ws = _FakeWS([json.dumps([1, 2, 3]), json.dumps({"action": "stop"})])
            await server._handler(ws)
            errors = [json.loads(sent).get("error") for sent in ws._sent]
            assert "malformed_frame" in errors
            assert server.runtime_counters()["malformed_frame_rejections"] == 1
            assert mon.psi_driver == pytest.approx(0.0)

        asyncio.run(_run())

    def test_origin_rejected_when_missing_header_and_allowlist_configured(self):
        mon = _make_monitor()
        server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456", allowed_origins=("https://ui.example",))
        assert server._origin_allowed(_FakeWS(headers={})) is False

    def test_origin_allowed_when_present_and_in_allowlist(self):
        mon = _make_monitor()
        server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456", allowed_origins=("https://ui.example",))
        assert server._origin_allowed(_FakeWS(headers={"Origin": "https://ui.example"})) is True

    def test_origin_rejected_when_present_and_not_in_allowlist(self):
        mon = _make_monitor()
        server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456", allowed_origins=("https://ui.example",))
        assert server._origin_allowed(_FakeWS(headers={"Origin": "https://evil.example"})) is False

    def test_stop_signals_tick_loop_to_halt(self):
        mon = _make_monitor()
        server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
        server._running = True
        server.stop()
        assert server._running is False

    def test_serve_cancels_tick_task_on_cancellation(self, monkeypatch):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456", tick_interval_s=0.01)

            class _FakeServeCtx:
                def __init__(self, *args, **kwargs):
                    pass

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    return False

            import types

            fake_ws = types.ModuleType("websockets")
            fake_ws.serve = _FakeServeCtx
            monkeypatch.setitem(sys.modules, "websockets", fake_ws)

            serve_task = asyncio.create_task(server.serve(port=9999))
            await asyncio.sleep(0.03)
            assert server._running is True
            serve_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await serve_task
            assert server._running is False

        asyncio.run(_run())

    def test_rate_limit_window_resets_after_configured_interval(self, monkeypatch):
        mon = _make_monitor()
        server = PhaseStreamServer(
            monitor=mon,
            api_key="secret-token-123456",
            command_rate_limit=1,
            command_rate_window_s=1.0,
        )
        ws = _FakeWS()
        times = iter([10.0, 11.5])
        monkeypatch.setattr(time, "monotonic", lambda: next(times))

        assert server._rate_limited(ws) is False
        assert server._rate_limited(ws) is False

    def test_tick_loop_sends_to_clients(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456", tick_interval_s=0.001)
            ws = _FakeWS()
            server._clients.add(ws)
            server._running = True

            async def _stop_after_ticks():
                await asyncio.sleep(0.05)
                server._running = False

            await asyncio.gather(server._tick_loop(), _stop_after_ticks())
            assert len(ws._sent) > 0
            frame = json.loads(ws._sent[0])
            assert "R_global" in frame

        asyncio.run(_run())

    def test_tick_loop_no_clients_idles(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456", tick_interval_s=0.001)
            server._running = True

            async def _stop_soon():
                await asyncio.sleep(0.06)
                server._running = False

            await asyncio.gather(server._tick_loop(), _stop_soon())

        asyncio.run(_run())

    def test_tick_loop_dead_client_removed(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456", tick_interval_s=0.001)

            class _DeadWS(_FakeWS):
                async def send(self, data):
                    raise ConnectionError("gone")

            dead = _DeadWS()
            server._clients.add(dead)
            server._running = True

            async def _stop():
                await asyncio.sleep(0.03)
                server._running = False

            await asyncio.gather(server._tick_loop(), _stop())
            assert dead not in server._clients

        asyncio.run(_run())

    def test_tick_loop_drops_slow_broadcast_client(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(
                monitor=mon,
                api_key="secret-token-123456",
                tick_interval_s=0.001,
                client_send_timeout_s=0.001,
            )

            class _SlowWS(_FakeWS):
                async def send(self, data):
                    await asyncio.sleep(10.0)

            slow = _SlowWS()
            server._clients.add(slow)
            server._running = True

            async def _stop_when_dropped():
                deadline = time.monotonic() + 0.1
                while slow in server._clients and time.monotonic() < deadline:
                    await asyncio.sleep(0.001)
                server._running = False

            await asyncio.gather(server._tick_loop(), _stop_when_dropped())
            assert slow not in server._clients
            assert slow.closed
            assert slow.close_code == 1011
            assert "backpressure" in (slow.close_reason or "")

        asyncio.run(_run())

    def test_tick_loop_slow_client_does_not_delay_healthy_client(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(
                monitor=mon,
                api_key="secret-token-123456",
                tick_interval_s=0.001,
                client_send_timeout_s=0.05,
            )
            fast_sent = asyncio.Event()
            fast_send_elapsed: list[float] = []

            class _SlowWS(_FakeWS):
                async def send(self, data):
                    await asyncio.sleep(10.0)

            class _FastWS(_FakeWS):
                async def send(self, data):
                    await super().send(data)
                    fast_send_elapsed.append(time.monotonic() - start)
                    fast_sent.set()

            class _OrderedClients:
                def __init__(self, clients):
                    self.clients = list(clients)

                def __bool__(self):
                    return bool(self.clients)

                def __iter__(self):
                    return iter(tuple(self.clients))

                def __isub__(self, dead):
                    self.clients = [client for client in self.clients if client not in dead]
                    return self

                def __contains__(self, client):
                    return client in self.clients

            slow = _SlowWS()
            fast = _FastWS()
            server._clients = _OrderedClients([slow, fast])
            server._running = True

            start = time.monotonic()

            async def _stop_after_fast_send():
                await asyncio.wait_for(fast_sent.wait(), timeout=0.02)
                server._running = False

            await asyncio.gather(server._tick_loop(), _stop_after_fast_send())

            assert fast_send_elapsed[0] < 0.02
            assert len(fast._sent) == 1
            assert slow not in server._clients

        asyncio.run(_run())

    def test_serve_requires_websockets(self, monkeypatch):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
            monkeypatch.setitem(sys.modules, "websockets", None)
            with pytest.raises(ImportError, match="websockets"):
                await server.serve()

        asyncio.run(_run())

    def test_serve_creates_tick_task_and_starts_websocket_server(self, monkeypatch):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456", tick_interval_s=0.01)

            serve_called = {}

            class _FakeServeCtx:
                def __init__(self, handler, host, port, **kwargs):
                    serve_called["handler"] = handler
                    serve_called["host"] = host
                    serve_called["port"] = port
                    serve_called["ssl"] = kwargs.get("ssl")
                    serve_called["max_size"] = kwargs.get("max_size")

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

            import types

            fake_ws = types.ModuleType("websockets")
            fake_ws.serve = _FakeServeCtx
            monkeypatch.setitem(sys.modules, "websockets", fake_ws)

            async def _stop_tick():
                await asyncio.sleep(0.02)
                server._running = False

            stop_task = asyncio.create_task(_stop_tick())
            await server.serve(host="127.0.0.1", port=9999)
            await stop_task
            assert serve_called["host"] == "127.0.0.1"
            assert serve_called["port"] == 9999
            assert serve_called["ssl"] is None
            assert serve_called["max_size"] == 65536

        asyncio.run(_run())

    def test_serve_defaults_to_loopback(self, monkeypatch):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456", tick_interval_s=0.01)
            serve_called = {}

            class _FakeServeCtx:
                def __init__(self, handler, host, port, **kwargs):
                    serve_called["host"] = host
                    serve_called["port"] = port
                    serve_called["ssl"] = kwargs.get("ssl")
                    serve_called["max_size"] = kwargs.get("max_size")

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

            import types

            fake_ws = types.ModuleType("websockets")
            fake_ws.serve = _FakeServeCtx
            monkeypatch.setitem(sys.modules, "websockets", fake_ws)

            async def _stop_tick():
                await asyncio.sleep(0.02)
                server._running = False

            stop_task = asyncio.create_task(_stop_tick())
            await server.serve(port=9999)
            await stop_task

            assert serve_called == {"host": "127.0.0.1", "port": 9999, "ssl": None, "max_size": 65536}

        asyncio.run(_run())

    def test_serve_rejects_non_loopback_bind_without_api_key_or_tls(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon)
            with pytest.raises(ValueError, match="api_key"):
                await server.serve(host="0.0.0.0", port=9999)

        asyncio.run(_run())

    def test_serve_rejects_non_loopback_without_api_key_even_when_auth_disabled(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, require_client_auth=False)
            with pytest.raises(ValueError, match="api_key"):
                await server.serve(host="0.0.0.0", port=9999)

        asyncio.run(_run())

    def test_serve_rejects_missing_api_key_when_auth_required(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon)
            with pytest.raises(ValueError, match="api_key"):
                await server.serve(host="127.0.0.1", port=9999)

        asyncio.run(_run())

    def test_serve_sync_delegates_to_async_server(self, monkeypatch):
        mon = _make_monitor()
        server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
        called = {}
        tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

        async def _fake_serve(host, port, ssl_context):
            called["host"] = host
            called["port"] = port
            called["ssl_context"] = ssl_context

        monkeypatch.setattr(server, "serve", _fake_serve)

        server.serve_sync(host="127.0.0.1", port=8766, ssl_context=tls_context)

        assert called == {"host": "127.0.0.1", "port": 8766, "ssl_context": tls_context}

    def test_main_builds_configured_secure_server_from_cli(self, monkeypatch):
        captured = {}

        class _FakeMonitorFactory:
            @staticmethod
            def from_paper27(*, L, N_per, zeta_uniform, psi_driver):
                captured["monitor"] = {
                    "L": L,
                    "N_per": N_per,
                    "zeta_uniform": zeta_uniform,
                    "psi_driver": psi_driver,
                }
                return object()

        class _FakeServer:
            def __init__(self, **kwargs):
                captured["server"] = kwargs

            def serve_sync(self, *, host, port, ssl_context):
                captured["serve"] = {"host": host, "port": port, "ssl_context": ssl_context}

        monkeypatch.setattr(ws_phase_stream, "RealtimeMonitor", _FakeMonitorFactory)
        monkeypatch.setattr(ws_phase_stream, "PhaseStreamServer", _FakeServer)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "ws_phase_stream",
                "--host",
                "127.0.0.1",
                "--port",
                "9999",
                "--layers",
                "8",
                "--n-per",
                "12",
                "--zeta",
                "0.25",
                "--psi",
                "0.4",
                "--tick-interval",
                "0.002",
                "--api-key",
                "secret-token-123456",
                "--command-rate-limit",
                "7",
                "--command-rate-window-s",
                "2.5",
                "--max-payload-bytes",
                "4096",
                "--client-send-timeout-s",
                "0.125",
                "--max-clients",
                "9",
                "--max-client-write-buffer-bytes",
                "8192",
                "--allow-query-token-auth",
                "--require-tls",
                "--allowed-origin",
                "https://ops.example",
                "--allowed-action",
                "stop",
            ],
        )

        ws_phase_stream.main()

        assert captured["monitor"] == {"L": 8, "N_per": 12, "zeta_uniform": 0.25, "psi_driver": 0.4}
        assert captured["server"]["api_key"] == "secret-token-123456"
        assert captured["server"]["command_rate_limit"] == 7
        assert captured["server"]["command_rate_window_s"] == 2.5
        assert captured["server"]["max_payload_bytes"] == 4096
        assert captured["server"]["client_send_timeout_s"] == 0.125
        assert captured["server"]["max_clients"] == 9
        assert captured["server"]["max_client_write_buffer_bytes"] == 8192
        assert captured["server"]["allow_query_token_auth"] is True
        assert captured["server"]["require_tls"] is True
        assert captured["server"]["allowed_origins"] == ("https://ops.example",)
        assert captured["server"]["allowed_actions"] == ("stop",)
        assert captured["serve"] == {"host": "127.0.0.1", "port": 9999, "ssl_context": None}

    def test_main_rejects_partial_tls_material(self, monkeypatch):
        class _FakeMonitorFactory:
            @staticmethod
            def from_paper27(**_kwargs):
                return object()

        monkeypatch.setattr(ws_phase_stream, "RealtimeMonitor", _FakeMonitorFactory)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "ws_phase_stream",
                "--api-key",
                "secret-token-123456",
                "--tls-cert",
                "cert.pem",
            ],
        )

        with pytest.raises(ValueError, match="--tls-cert and --tls-key"):
            ws_phase_stream.main()

    def test_serve_rejects_plaintext_non_loopback_by_default(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
            with pytest.raises(ValueError, match="ssl_context"):
                await server.serve(host="0.0.0.0", port=9999)

        asyncio.run(_run())

    def test_serve_requires_tls_when_configured(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456", require_tls=True)
            with pytest.raises(ValueError, match="ssl_context"):
                await server.serve(host="127.0.0.1", port=9999)

        asyncio.run(_run())

    def test_serve_passes_tls_context_to_websocket_server(self, monkeypatch):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(
                monitor=mon,
                api_key="secret-token-123456",
                tick_interval_s=0.01,
                max_payload_bytes=1234,
                require_tls=True,
            )
            tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            serve_called = {}

            class _FakeServeCtx:
                def __init__(self, handler, host, port, **kwargs):
                    serve_called["host"] = host
                    serve_called["ssl"] = kwargs.get("ssl")
                    serve_called["max_size"] = kwargs.get("max_size")

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

            import types

            fake_ws = types.ModuleType("websockets")
            fake_ws.serve = _FakeServeCtx
            monkeypatch.setitem(sys.modules, "websockets", fake_ws)

            async def _stop_tick():
                await asyncio.sleep(0.02)
                server._running = False

            stop_task = asyncio.create_task(_stop_tick())
            await server.serve(host="0.0.0.0", port=9999, ssl_context=tls_context)
            await stop_task

            assert serve_called["host"] == "0.0.0.0"
            assert serve_called["ssl"] is tls_context
            assert serve_called["max_size"] == 1234

        asyncio.run(_run())

    def test_runtime_counters_record_command_broadcast_and_backpressure(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(
                monitor=mon,
                api_key="secret-token-123456",
                tick_interval_s=0.001,
                client_send_timeout_s=0.001,
            )
            ws = _FakeWS([json.dumps({"action": "set_psi", "value": 0.2})])

            await server._handler(ws)

            counters = server.runtime_counters()
            assert counters["auth_successes"] == 1
            assert counters["command_frames"] == 1
            assert counters["peak_connected_clients"] == 1

            class _SlowWS(_FakeWS):
                async def send(self, data):
                    await asyncio.sleep(10.0)

            slow = _SlowWS()
            server._clients.add(slow)
            server._running = True

            async def _stop_when_dropped():
                deadline = time.monotonic() + 0.1
                while slow in server._clients and time.monotonic() < deadline:
                    await asyncio.sleep(0.001)
                server._running = False

            await asyncio.gather(server._tick_loop(), _stop_when_dropped())

            counters = server.runtime_counters()
            assert counters["backpressure_disconnects"] == 1
            assert counters["broadcast_frames"] == 0

        asyncio.run(_run())

    def test_websocket_runtime_evidence_round_trips_without_secret_and_supports_claim(self, tmp_path):
        mon = _make_monitor()
        server = PhaseStreamServer(
            monitor=mon,
            api_key="secret-token-123456",
            require_tls=True,
            allowed_origins=("https://ops.example",),
        )
        evidence = websocket_runtime_evidence(
            server,
            deployment_id="phase-stream-prod-a",
            bind_host="ops.phase.internal",
            uses_tls=True,
            counters={
                "auth_successes": 1,
                "command_frames": 1,
                "broadcast_frames": 2,
                "peak_connected_clients": 1,
            },
            generated_utc="2026-05-31T00:00:00Z",
            facility_claim_allowed=True,
        )

        encoded = json.dumps(asdict(evidence), sort_keys=True)
        assert "secret-token-123456" not in encoded
        assert evidence.require_tls is True
        assert evidence.command_frames == 1
        assert_websocket_runtime_claim_admissible(evidence)

        path = tmp_path / "websocket_runtime.json"
        save_websocket_runtime_evidence(evidence, path)
        assert load_websocket_runtime_evidence(path, require_facility_claim=True) == evidence

    def test_websocket_runtime_evidence_rejects_local_only_and_tls_gaps(self):
        mon = _make_monitor()
        server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456", require_tls=True)
        local_only = websocket_runtime_evidence(
            server,
            deployment_id="local-phase-stream",
            counters={
                "auth_successes": 1,
                "command_frames": 1,
                "broadcast_frames": 1,
                "peak_connected_clients": 1,
            },
            generated_utc="2026-05-31T00:00:00Z",
        )
        with pytest.raises(ValueError, match="local-only"):
            assert_websocket_runtime_claim_admissible(local_only)

        lax_server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
        with pytest.raises(ValueError, match="TLS enforcement"):
            websocket_runtime_evidence(
                lax_server,
                deployment_id="phase-stream-prod-a",
                bind_host="ops.phase.internal",
                uses_tls=True,
                counters={
                    "auth_successes": 1,
                    "command_frames": 1,
                    "broadcast_frames": 1,
                    "peak_connected_clients": 1,
                },
                generated_utc="2026-05-31T00:00:00Z",
                facility_claim_allowed=True,
            )

        with pytest.raises(ValueError, match="authenticated commands"):
            websocket_runtime_evidence(
                server,
                deployment_id="phase-stream-prod-a",
                bind_host="ops.phase.internal",
                uses_tls=True,
                counters={
                    "auth_successes": 1,
                    "broadcast_frames": 1,
                    "peak_connected_clients": 1,
                },
                generated_utc="2026-05-31T00:00:00Z",
                facility_claim_allowed=True,
            )

    def test_websocket_runtime_evidence_rejects_tampering_and_duplicate_keys(self, tmp_path):
        mon = _make_monitor()
        server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456", require_tls=True)
        evidence = websocket_runtime_evidence(
            server,
            deployment_id="phase-stream-prod-a",
            bind_host="ops.phase.internal",
            uses_tls=True,
            counters={
                "auth_successes": 1,
                "command_frames": 1,
                "broadcast_frames": 1,
                "peak_connected_clients": 1,
            },
            generated_utc="2026-05-31T00:00:00Z",
            facility_claim_allowed=True,
        )
        path = tmp_path / "websocket_runtime.json"
        save_websocket_runtime_evidence(evidence, path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["command_frames"] = 9
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="payload_sha256"):
            load_websocket_runtime_evidence(path)

        duplicate_path = tmp_path / "duplicate_websocket_runtime.json"
        duplicate_path.write_text(
            (
                '{"schema_version":"'
                + ws_phase_stream.WEBSOCKET_RUNTIME_EVIDENCE_SCHEMA_VERSION
                + '","schema_version":"'
                + ws_phase_stream.WEBSOCKET_RUNTIME_EVIDENCE_SCHEMA_VERSION
                + '"}'
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate JSON key"):
            load_websocket_runtime_evidence(duplicate_path)


# ── Helper-validator, runtime-evidence and handler-edge contracts ─────────────


_CONFIG_KEYS = (
    "bind_host",
    "uses_tls",
    "require_client_auth",
    "api_key_configured",
    "api_key_min_bytes",
    "require_tls",
    "allow_query_token_auth",
    "allow_insecure_remote",
    "command_rate_limit",
    "command_rate_window_s",
    "max_payload_bytes",
    "client_send_timeout_s",
    "max_clients",
    "max_client_write_buffer_bytes",
    "allowed_actions",
    "allowed_origins",
)


def _local_evidence_dict():
    server = PhaseStreamServer(monitor=_make_monitor(), api_key="x" * 16)
    return asdict(websocket_runtime_evidence(server, deployment_id="deploy-local"))


def _qualified_evidence_dict():
    server = PhaseStreamServer(
        monitor=_make_monitor(),
        api_key="x" * 32,
        require_client_auth=True,
        require_tls=True,
    )
    evidence = websocket_runtime_evidence(
        server,
        deployment_id="deploy-qualified",
        uses_tls=True,
        facility_claim_allowed=True,
        counters={
            "auth_successes": 3,
            "command_frames": 5,
            "broadcast_frames": 7,
            "peak_connected_clients": 2,
        },
    )
    return asdict(evidence)


def _reseal_payload(payload):
    payload["payload_sha256"] = ws_phase_stream._payload_sha256(payload)
    return payload


def _reseal_config_and_payload(payload):
    config = {key: payload[key] for key in _CONFIG_KEYS}
    config["allowed_actions"] = tuple(payload["allowed_actions"])
    config["allowed_origins"] = tuple(payload["allowed_origins"])
    payload["config_sha256"] = ws_phase_stream._runtime_config_sha256(config)
    payload["payload_sha256"] = ws_phase_stream._payload_sha256(payload)
    return payload


def test_payload_size_bytes_handles_non_text_message():
    assert ws_phase_stream._payload_size_bytes(12345) == len("12345".encode("utf-8"))


def test_websocket_peer_accepts_string_remote():
    assert ws_phase_stream._websocket_peer(SimpleNamespace(remote_address="10.0.0.5")) == "10.0.0.5"


def test_websocket_peer_falls_back_to_unknown():
    assert ws_phase_stream._websocket_peer(SimpleNamespace(remote_address=None)) == "unknown"


def test_utc_now_is_z_suffixed():
    assert ws_phase_stream._utc_now().endswith("Z")


@pytest.mark.parametrize(
    ("fn_name", "arg", "match"),
    [
        ("_require_nonnegative_int", -1, "non-negative integer"),
        ("_require_positive_int", 0, "positive integer"),
        ("_require_finite_positive", object(), "finite positive"),
        ("_require_finite_positive", "not-a-number", "finite positive"),
        ("_require_finite_positive", -1.0, "finite positive"),
    ],
)
def test_scalar_validators_reject_bad_values(fn_name, arg, match):
    with pytest.raises(ValueError, match=match):
        getattr(ws_phase_stream, fn_name)("field", arg)


def test_tuple_of_nonempty_strings_rejects_non_sequence():
    with pytest.raises(ValueError, match="must be a list or tuple"):
        ws_phase_stream._tuple_of_nonempty_strings("field", 5)


def test_tuple_of_nonempty_strings_rejects_blank_entry():
    with pytest.raises(ValueError, match="non-empty strings"):
        ws_phase_stream._tuple_of_nonempty_strings("field", ["ok", "  "])


def test_server_rejects_empty_api_key():
    with pytest.raises(ValueError, match="api_key must be non-empty"):
        PhaseStreamServer(monitor=_make_monitor(), api_key="")


def test_runtime_evidence_rejects_non_server():
    with pytest.raises(ValueError, match="server must be PhaseStreamServer"):
        websocket_runtime_evidence(object(), deployment_id="d")  # type: ignore[arg-type]


def test_runtime_evidence_rejects_blank_deployment_id():
    server = PhaseStreamServer(monitor=_make_monitor(), api_key="x" * 16)
    with pytest.raises(ValueError, match="deployment_id must be non-empty"):
        websocket_runtime_evidence(server, deployment_id="  ")


def test_runtime_evidence_rejects_blank_bind_host():
    server = PhaseStreamServer(monitor=_make_monitor(), api_key="x" * 16)
    with pytest.raises(ValueError, match="bind_host must be non-empty"):
        websocket_runtime_evidence(server, deployment_id="d", bind_host="  ")


def test_runtime_evidence_rejects_non_bool_uses_tls():
    server = PhaseStreamServer(monitor=_make_monitor(), api_key="x" * 16)
    with pytest.raises(ValueError, match="uses_tls must be boolean"):
        websocket_runtime_evidence(server, deployment_id="d", uses_tls="yes")  # type: ignore[arg-type]


def test_save_evidence_rejects_non_evidence(tmp_path):
    with pytest.raises(ValueError, match="evidence must be WebSocketRuntimeEvidence"):
        save_websocket_runtime_evidence({"not": "evidence"}, tmp_path / "e.json")  # type: ignore[arg-type]


def test_load_evidence_rejects_non_object(tmp_path):
    path = tmp_path / "e.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_websocket_runtime_evidence(path)


def test_validate_runtime_payload_rejects_non_sha_digest():
    payload = _local_evidence_dict()
    payload["payload_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="payload_sha256 must be a SHA-256"):
        ws_phase_stream._validate_websocket_runtime_payload(payload, require_facility_claim=False)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", "bad", "schema_version is unsupported"),
        ("generated_utc", "2026-01-01T00:00:00", "generated_utc must be a UTC timestamp"),
        ("deployment_id", "  ", "deployment_id must be non-empty"),
        ("bind_host", "  ", "bind_host must be non-empty"),
        ("uses_tls", "x", "uses_tls must be boolean"),
        ("allowed_actions", ["bogus_action"], "known WebSocket commands"),
        ("config_sha256", "z", "config_sha256 must be a SHA-256"),
        ("config_sha256", "a" * 64, "config_sha256 does not match"),
    ],
)
def test_validate_runtime_payload_rejects_field(field, value, match):
    payload = _local_evidence_dict()
    payload[field] = value
    _reseal_payload(payload)
    with pytest.raises(ValueError, match=match):
        ws_phase_stream._validate_websocket_runtime_payload(payload, require_facility_claim=False)


def test_validate_runtime_payload_rejects_claim_status_mismatch():
    payload = _qualified_evidence_dict()
    payload["claim_status"] = WEBSOCKET_RUNTIME_EVIDENCE_LOCAL_ONLY
    _reseal_payload(payload)
    with pytest.raises(ValueError, match="claim_status does not match"):
        ws_phase_stream._validate_websocket_runtime_payload(payload, require_facility_claim=False)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"require_client_auth": False}, "requires authenticated clients"),
        ({"api_key_configured": False}, "requires authenticated clients"),
        ({"api_key_min_bytes": 8}, "hardened API-key policy"),
        ({"uses_tls": False}, "requires TLS"),
        ({"require_tls": False}, "requires TLS enforcement"),
        ({"allow_query_token_auth": True}, "forbids query-token"),
        ({"allow_insecure_remote": True}, "forbids insecure remote"),
        ({"max_payload_bytes": 70000}, "payload cap"),
        ({"auth_successes": 0}, "at least one authenticated session"),
        ({"command_frames": 0}, "observed authenticated commands"),
        ({"broadcast_frames": 0}, "observed broadcast frames"),
        ({"peak_connected_clients": 0}, "observed client connectivity"),
        ({"backpressure_disconnects": 1}, "cannot contain backpressure"),
    ],
)
def test_qualified_runtime_evidence_rejects(overrides, match):
    payload = _qualified_evidence_dict()
    payload.update(overrides)
    _reseal_config_and_payload(payload)
    with pytest.raises(ValueError, match=match):
        ws_phase_stream._validate_websocket_runtime_payload(payload, require_facility_claim=True)


def test_close_client_without_close_method():
    async def _run():
        server = PhaseStreamServer(monitor=_make_monitor(), api_key="x" * 16)
        await server._close_client(SimpleNamespace(), code=1000, reason="x")

    asyncio.run(_run())


def test_handler_breaks_when_oversized_payload_response_fails():
    async def _run():
        server = PhaseStreamServer(monitor=_make_monitor(), api_key="secret-token-123456")

        class _DeadSendWS(_FakeWS):
            async def send(self, data):
                raise ConnectionError("dead")

        ws = _DeadSendWS(["x" * (server.max_payload_bytes + 10)], headers=_AUTH_HEADERS)
        await server._handler(ws)
        assert ws not in server._clients

    asyncio.run(_run())


def test_handler_breaks_when_rate_limit_response_fails():
    async def _run():
        server = PhaseStreamServer(monitor=_make_monitor(), api_key="secret-token-123456", command_rate_limit=1)

        class _DeadSendWS(_FakeWS):
            async def send(self, data):
                raise ConnectionError("dead")

        msg = json.dumps({"action": "set_psi", "value": 0.5})
        ws = _DeadSendWS([msg, msg], headers=_AUTH_HEADERS)
        await server._handler(ws)
        assert ws not in server._clients

    asyncio.run(_run())


def test_handler_breaks_when_malformed_frame_response_fails():
    async def _run():
        server = PhaseStreamServer(monitor=_make_monitor(), api_key="secret-token-123456")

        class _DeadSendWS(_FakeWS):
            async def send(self, data):
                raise ConnectionError("dead")

        ws = _DeadSendWS(["not-json"], headers=_AUTH_HEADERS)
        await server._handler(ws)
        assert ws not in server._clients

    asyncio.run(_run())


def test_handler_breaks_when_non_object_frame_response_fails():
    async def _run():
        server = PhaseStreamServer(monitor=_make_monitor(), api_key="secret-token-123456")

        class _DeadSendWS(_FakeWS):
            async def send(self, data):
                raise ConnectionError("dead")

        ws = _DeadSendWS([json.dumps([1, 2, 3])], headers=_AUTH_HEADERS)
        await server._handler(ws)
        assert ws not in server._clients

    asyncio.run(_run())


def test_tick_loop_reraises_cancelled_error():
    async def _run():
        server = PhaseStreamServer(monitor=_make_monitor(), api_key="x" * 16, tick_interval_s=0.001)

        class _CancelWS(_FakeWS):
            async def send(self, data):
                raise asyncio.CancelledError

        server._clients.add(_CancelWS())
        server._running = True
        with pytest.raises(asyncio.CancelledError):
            await server._tick_loop()

    asyncio.run(_run())


def test_tick_loop_logs_and_continues_on_send_exception():
    async def _run():
        server = PhaseStreamServer(monitor=_make_monitor(), api_key="x" * 16, tick_interval_s=0.001)

        class _ExcWS(_FakeWS):
            async def send(self, data):
                raise ValueError("boom")

        server._clients.add(_ExcWS())
        server._running = True

        async def _stop():
            await asyncio.sleep(0.03)
            server._running = False

        await asyncio.gather(server._tick_loop(), _stop())

    asyncio.run(_run())


def test_main_builds_tls_context(monkeypatch, tmp_path):
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("cert", encoding="utf-8")
    key.write_text("key", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["scpn-ws", "--tls-cert", str(cert), "--tls-key", str(key)])
    monkeypatch.setattr(ssl.SSLContext, "load_cert_chain", lambda self, *a, **k: None)
    monkeypatch.setattr(ws_phase_stream.PhaseStreamServer, "serve_sync", lambda self, **k: None)
    ws_phase_stream.main()


class TestModernWebsocketsApi:
    """websockets>=15 removed request_headers/path; auth reads connection.request.

    websockets >= 15 (16.0 is the currently resolved release of the optional ``ws``
    extra) presents the opening handshake on ``connection.request`` and no longer
    exposes ``request_headers``/``path`` on the connection, so the pre-fix helpers
    denied every client. These tests exercise the fixed helpers against the modern
    connection shape and the legacy connection, keeping both branches covered
    without the optional extra installed.
    """

    def test_get_header_reads_modern_request_headers_case_insensitively(self):
        ws = _ModernWS(headers={"Authorization": "Bearer secret-token-123456"})
        assert ws_phase_stream._get_header(ws, "Authorization") == "Bearer secret-token-123456"
        assert ws_phase_stream._get_header(ws, "authorization") == "Bearer secret-token-123456"

    def test_request_path_reads_modern_request_path_with_query(self):
        ws = _ModernWS(path="/phase?token=secret-token-123456")
        assert ws_phase_stream._request_path(ws) == "/phase?token=secret-token-123456"

    def test_handler_authenticates_control_connection_via_modern_headers(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
            ws = _ModernWS([json.dumps({"action": "set_psi", "value": 0.5})])

            await server._handler(ws)

            assert mon.psi_driver == pytest.approx(0.5)
            assert not ws.closed

        asyncio.run(_run())

    def test_handler_authenticates_query_token_via_modern_request_path(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(
                monitor=mon,
                api_key="secret-token-123456",
                allow_query_token_auth=True,
            )
            ws = _ModernWS(
                [json.dumps({"action": "set_pac_gamma", "value": 0.2})],
                headers={},
                path="/phase?token=secret-token-123456",
            )

            await server._handler(ws)

            assert mon.pac_gamma == pytest.approx(0.2)

        asyncio.run(_run())

    def test_helpers_fall_back_to_legacy_connection_attributes(self):
        ws = _FakeWS(headers={"Authorization": "Bearer legacy-token-123456"}, path="/legacy?token=x")
        assert ws_phase_stream._get_header(ws, "Authorization") == "Bearer legacy-token-123456"
        assert ws_phase_stream._request_path(ws) == "/legacy?token=x"

    def test_helpers_return_empty_when_connection_exposes_neither_surface(self):
        ws = SimpleNamespace()
        assert ws_phase_stream._get_header(ws, "Authorization") is None
        assert ws_phase_stream._request_path(ws) == ""

    def test_helpers_resolve_a_real_websockets_request_object(self):
        pytest.importorskip("websockets")
        from websockets.datastructures import Headers
        from websockets.http11 import Request

        request = Request(
            path="/phase?token=secret-token-123456",
            headers=Headers([("Authorization", "Bearer secret-token-123456")]),
        )
        ws = SimpleNamespace(request=request)
        assert ws_phase_stream._get_header(ws, "Authorization") == "Bearer secret-token-123456"
        assert ws_phase_stream._get_header(ws, "authorization") == "Bearer secret-token-123456"
        assert ws_phase_stream._request_path(ws) == "/phase?token=secret-token-123456"

    def test_live_end_to_end_auth_with_a_real_websockets_client(self):
        websockets = pytest.importorskip("websockets")

        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret-token-123456")
            async with websockets.serve(server._handler, "127.0.0.1", 0) as running:
                port = running.sockets[0].getsockname()[1]
                async with websockets.connect(
                    f"ws://127.0.0.1:{port}/phase",
                    additional_headers={"Authorization": "Bearer secret-token-123456"},
                ) as client:
                    await client.send(json.dumps({"action": "set_psi", "value": 0.7}))
                    await asyncio.sleep(0.1)
            assert mon.psi_driver == pytest.approx(0.7)

        asyncio.run(asyncio.wait_for(_run(), timeout=10))
