# ──────────────────────────────────────────────────────────────────────
# SCPN Control — WebSocket Phase Stream Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for ws_phase_stream: handler, tick loop, serve."""
from __future__ import annotations

import asyncio
import json

import pytest

from scpn_control.phase.realtime_monitor import RealtimeMonitor
from scpn_control.phase.ws_phase_stream import PhaseStreamServer


def _make_monitor():
    return RealtimeMonitor.from_paper27(L=4, N_per=10, zeta_uniform=0.5, psi_driver=0.0)


class _FakeWS:
    """Mock WebSocket for testing handler and tick loop."""

    def __init__(self, messages=None):
        self._messages = list(messages or [])
        self._sent: list[str] = []
        self._idx = 0
        self.closed = False

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


class TestPhaseStreamServer:
    def test_init(self):
        mon = _make_monitor()
        server = PhaseStreamServer(monitor=mon, tick_interval_s=0.01)
        assert server.tick_interval_s == 0.01

    @pytest.mark.asyncio
    async def test_handler_set_psi(self):
        mon = _make_monitor()
        server = PhaseStreamServer(monitor=mon)
        ws = _FakeWS([json.dumps({"action": "set_psi", "value": 0.5})])
        await server._handler(ws)
        assert mon.psi_driver == pytest.approx(0.5)
        assert ws not in server._clients

    @pytest.mark.asyncio
    async def test_handler_set_pac_gamma(self):
        mon = _make_monitor()
        server = PhaseStreamServer(monitor=mon)
        ws = _FakeWS([json.dumps({"action": "set_pac_gamma", "value": 0.3})])
        await server._handler(ws)
        assert mon.pac_gamma == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_handler_reset(self):
        mon = _make_monitor()
        server = PhaseStreamServer(monitor=mon)
        ws = _FakeWS([json.dumps({"action": "reset", "seed": 99})])
        await server._handler(ws)

    @pytest.mark.asyncio
    async def test_handler_stop(self):
        mon = _make_monitor()
        server = PhaseStreamServer(monitor=mon)
        ws = _FakeWS([json.dumps({"action": "stop"})])
        await server._handler(ws)
        assert server._running is False

    @pytest.mark.asyncio
    async def test_handler_bad_json_ignored(self):
        mon = _make_monitor()
        server = PhaseStreamServer(monitor=mon)
        ws = _FakeWS(["not-json", json.dumps({"action": "stop"})])
        await server._handler(ws)

    @pytest.mark.asyncio
    async def test_tick_loop_sends_to_clients(self):
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

    @pytest.mark.asyncio
    async def test_tick_loop_no_clients_idles(self):
        mon = _make_monitor()
        server = PhaseStreamServer(monitor=mon, tick_interval_s=0.001)
        server._running = True

        async def _stop_soon():
            await asyncio.sleep(0.06)
            server._running = False

        await asyncio.gather(server._tick_loop(), _stop_soon())

    @pytest.mark.asyncio
    async def test_tick_loop_dead_client_removed(self):
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

    @pytest.mark.asyncio
    async def test_serve_requires_websockets(self, monkeypatch):
        mon = _make_monitor()
        server = PhaseStreamServer(monitor=mon)
        import scpn_control.phase.ws_phase_stream as ws_mod
        import sys
        monkeypatch.setitem(sys.modules, "websockets", None)
        with pytest.raises(ImportError, match="websockets"):
            await server.serve()
