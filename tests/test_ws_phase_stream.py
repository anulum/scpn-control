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

import pytest

import scpn_control.phase.ws_phase_stream as ws_phase_stream
from scpn_control.phase.realtime_monitor import RealtimeMonitor
from scpn_control.phase.ws_phase_stream import PhaseStreamServer


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
