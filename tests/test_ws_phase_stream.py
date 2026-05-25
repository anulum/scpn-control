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
import math
import ssl
import sys
import time

import pytest

from scpn_control.phase.realtime_monitor import RealtimeMonitor
from scpn_control.phase.ws_phase_stream import PhaseStreamServer


def _make_monitor():
    return RealtimeMonitor.from_paper27(L=4, N_per=10, zeta_uniform=0.5, psi_driver=0.0)


class _FakeWS:
    """Mock WebSocket for testing handler and tick loop."""

    def __init__(self, messages=None, headers=None, path="/"):
        self._messages = list(messages or [])
        self._sent: list[str] = []
        self.request_headers = headers or {}
        self.path = path
        self._idx = 0
        self.closed = False
        self.close_code = None
        self.close_reason = None

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

    def test_handler_rejects_unauthenticated_control_connection_when_api_key_configured(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret")
            ws = _FakeWS([json.dumps({"action": "set_psi", "value": 0.5})])

            await server._handler(ws)

            assert mon.psi_driver == pytest.approx(0.0)
            assert ws.closed
            assert ws.close_code == 1008
            assert "authentication" in (ws.close_reason or "")
            assert ws not in server._clients

        asyncio.run(_run())

    @pytest.mark.parametrize(
        "headers",
        [
            {"Authorization": "Bearer secret"},
            {"X-SCPN-API-Key": "secret"},
        ],
    )
    def test_handler_accepts_authenticated_control_connection(self, headers):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret")
            ws = _FakeWS([json.dumps({"action": "set_psi", "value": 0.5})], headers=headers)

            await server._handler(ws)

            assert mon.psi_driver == pytest.approx(0.5)
            assert not ws.closed

        asyncio.run(_run())

    def test_handler_accepts_query_token_for_authentication(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret")
            ws = _FakeWS([json.dumps({"action": "set_pac_gamma", "value": 0.2})], path="/phase?token=secret")

            await server._handler(ws)

            assert mon.pac_gamma == pytest.approx(0.2)

        asyncio.run(_run())

    def test_handler_rate_limits_control_commands_without_state_mutation(self, monkeypatch):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, command_rate_limit=1, command_rate_window_s=60.0)
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

    def test_handler_set_psi(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon)
            ws = _FakeWS([json.dumps({"action": "set_psi", "value": 0.5})])
            await server._handler(ws)
            assert mon.psi_driver == pytest.approx(0.5)
            assert ws not in server._clients

        asyncio.run(_run())

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "not-a-number"])
    def test_handler_rejects_invalid_psi_commands_without_state_mutation(self, value):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon)
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
            server = PhaseStreamServer(monitor=mon)
            ws = _FakeWS([json.dumps({"action": "set_pac_gamma", "value": 0.3})])
            await server._handler(ws)
            assert mon.pac_gamma == pytest.approx(0.3)

        asyncio.run(_run())

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "not-a-number"])
    def test_handler_rejects_invalid_pac_gamma_commands_without_state_mutation(self, value):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon)
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
            server = PhaseStreamServer(monitor=mon)
            ws = _FakeWS([json.dumps({"action": "reset", "seed": 99})])
            await server._handler(ws)

        asyncio.run(_run())

    def test_handler_stop(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon)
            ws = _FakeWS([json.dumps({"action": "stop"})])
            await server._handler(ws)
            assert server._running is False

        asyncio.run(_run())

    def test_handler_bad_json_ignored(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon)
            ws = _FakeWS(["not-json", json.dumps({"action": "stop"})])
            await server._handler(ws)

        asyncio.run(_run())

    def test_tick_loop_sends_to_clients(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, tick_interval_s=0.001)
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
            server = PhaseStreamServer(monitor=mon, tick_interval_s=0.001)
            server._running = True

            async def _stop_soon():
                await asyncio.sleep(0.06)
                server._running = False

            await asyncio.gather(server._tick_loop(), _stop_soon())

        asyncio.run(_run())

    def test_tick_loop_dead_client_removed(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, tick_interval_s=0.001)

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

    def test_serve_requires_websockets(self, monkeypatch):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon)
            monkeypatch.setitem(sys.modules, "websockets", None)
            with pytest.raises(ImportError, match="websockets"):
                await server.serve()

        asyncio.run(_run())

    def test_serve_creates_tick_task_and_starts_websocket_server(self, monkeypatch):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, tick_interval_s=0.01)

            serve_called = {}

            class _FakeServeCtx:
                def __init__(self, handler, host, port, **kwargs):
                    serve_called["handler"] = handler
                    serve_called["host"] = host
                    serve_called["port"] = port
                    serve_called["ssl"] = kwargs.get("ssl")

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

        asyncio.run(_run())

    def test_serve_defaults_to_loopback(self, monkeypatch):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, tick_interval_s=0.01)
            serve_called = {}

            class _FakeServeCtx:
                def __init__(self, handler, host, port, **kwargs):
                    serve_called["host"] = host
                    serve_called["port"] = port
                    serve_called["ssl"] = kwargs.get("ssl")

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

            assert serve_called == {"host": "127.0.0.1", "port": 9999, "ssl": None}

        asyncio.run(_run())

    def test_serve_rejects_non_loopback_bind_without_api_key_or_tls(self):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon)
            with pytest.raises(ValueError, match="api_key"):
                await server.serve(host="0.0.0.0", port=9999)

        asyncio.run(_run())

    def test_serve_passes_tls_context_to_websocket_server(self, monkeypatch):
        async def _run():
            mon = _make_monitor()
            server = PhaseStreamServer(monitor=mon, api_key="secret", tick_interval_s=0.01)
            tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            serve_called = {}

            class _FakeServeCtx:
                def __init__(self, handler, host, port, **kwargs):
                    serve_called["host"] = host
                    serve_called["ssl"] = kwargs.get("ssl")

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

        asyncio.run(_run())
