# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — WebSocket Phase Sync Stream
"""
Async WebSocket server streaming RealtimeMonitor tick snapshots.

Start standalone::

    python -m scpn_control.phase.ws_phase_stream --port 8765

Or embed in an existing asyncio loop::

    server = PhaseStreamServer(monitor, api_key="operator-token")
    await server.serve(host="127.0.0.1", port=8765)

Clients receive JSON frames every tick::

    {"tick": 1, "R_global": 0.42, "V_global": 0.83, "lambda_exp": -0.12, ...}
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import math
import os
import ssl
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlsplit

from scpn_control.phase.realtime_monitor import RealtimeMonitor

logger = logging.getLogger(__name__)


def _finite_command_value(cmd: dict[str, Any]) -> float | None:
    """Return a finite scalar command value, or ``None`` for invalid input."""
    try:
        value = float(cmd["value"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _is_loopback_host(host: str) -> bool:
    normalised = host.strip().lower()
    return normalised in {"localhost", "127.0.0.1", "::1"}


def _get_header(websocket: Any, name: str) -> str | None:
    headers = getattr(websocket, "request_headers", {}) or {}
    if hasattr(headers, "get"):
        return headers.get(name) or headers.get(name.lower())
    return None


def _bearer_token(authorization: str | None) -> str | None:
    if authorization is None:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


@dataclass
class PhaseStreamServer:
    """Async WebSocket server wrapping a RealtimeMonitor."""

    monitor: RealtimeMonitor
    tick_interval_s: float = 0.001
    api_key: str | None = None
    command_rate_limit: int = 20
    command_rate_window_s: float = 1.0
    _clients: set = field(default_factory=set, init=False, repr=False)
    _client_windows: dict[Any, tuple[float, int]] = field(default_factory=dict, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.tick_interval_s) or self.tick_interval_s <= 0.0:
            raise ValueError("tick_interval_s must be finite and positive")
        if self.api_key is not None and not self.api_key:
            raise ValueError("api_key must be non-empty when configured")
        if (
            isinstance(self.command_rate_limit, bool)
            or not isinstance(self.command_rate_limit, int)
            or self.command_rate_limit <= 0
        ):
            raise ValueError("command_rate_limit must be a positive integer")
        if not math.isfinite(self.command_rate_window_s) or self.command_rate_window_s <= 0.0:
            raise ValueError("command_rate_window_s must be finite and positive")

    def _authorised(self, websocket: Any) -> bool:
        if self.api_key is None:
            return True
        candidate = _bearer_token(_get_header(websocket, "Authorization"))
        candidate = candidate or _get_header(websocket, "X-SCPN-API-Key")
        if candidate is None:
            path = getattr(websocket, "path", "") or ""
            query = parse_qs(urlsplit(path).query)
            values = query.get("token") or query.get("access_token") or []
            candidate = values[0] if values else None
        return candidate is not None and hmac.compare_digest(candidate, self.api_key)

    def _rate_limited(self, websocket: Any) -> bool:
        now = time.monotonic()
        window_start, count = self._client_windows.get(websocket, (now, 0))
        if now - window_start >= self.command_rate_window_s:
            window_start, count = now, 0
        count += 1
        self._client_windows[websocket] = (window_start, count)
        return count > self.command_rate_limit

    async def _handler(self, websocket: Any) -> None:
        if not self._authorised(websocket):
            await websocket.close(code=1008, reason="authentication required")
            return
        self._clients.add(websocket)
        logger.info("Client connected (%d total)", len(self._clients))
        try:
            async for msg in websocket:
                try:
                    cmd = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                if not isinstance(cmd, dict):
                    continue
                if self._rate_limited(websocket):
                    await websocket.send(json.dumps({"error": "rate_limited"}))
                    await websocket.close(code=1008, reason="command rate limit exceeded")
                    break
                if cmd.get("action") == "set_psi":
                    value = _finite_command_value(cmd)
                    if value is not None:
                        self.monitor.psi_driver = value
                elif cmd.get("action") == "set_pac_gamma":
                    value = _finite_command_value(cmd)
                    if value is not None:
                        self.monitor.pac_gamma = value
                elif cmd.get("action") == "reset":
                    self.monitor.reset(seed=cmd.get("seed", 42))
                elif cmd.get("action") == "stop":
                    self._running = False
        finally:
            self._clients.discard(websocket)
            self._client_windows.pop(websocket, None)
            logger.info("Client disconnected (%d remain)", len(self._clients))

    async def _tick_loop(self) -> None:
        self._running = True
        while self._running:
            if not self._clients:
                await asyncio.sleep(0.05)
                continue
            snap = self.monitor.tick()
            frame = json.dumps(snap)
            dead = set()
            for ws in self._clients:
                try:
                    await ws.send(frame)
                except (ConnectionError, OSError):
                    dead.add(ws)
            self._clients -= dead
            await asyncio.sleep(self.tick_interval_s)

    async def serve(self, host: str = "127.0.0.1", port: int = 8765, ssl_context: ssl.SSLContext | None = None) -> None:
        """Start WebSocket server and tick loop."""
        if not _is_loopback_host(host) and self.api_key is None:
            raise ValueError("api_key is required when binding the phase stream outside loopback")
        try:
            import websockets  # noqa: F811
        except ImportError as exc:
            raise ImportError("pip install websockets") from exc

        tick_task = asyncio.create_task(self._tick_loop())
        async with websockets.serve(self._handler, host, port, ssl=ssl_context):
            scheme = "wss" if ssl_context is not None else "ws"
            logger.info("Phase stream listening on %s://%s:%d", scheme, host, port)
            await tick_task

    def serve_sync(self, host: str = "127.0.0.1", port: int = 8765, ssl_context: ssl.SSLContext | None = None) -> None:
        """Blocking entry point."""
        asyncio.run(self.serve(host, port, ssl_context))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SCPN Phase Sync WebSocket Stream")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--n-per", type=int, default=50)
    parser.add_argument("--zeta", type=float, default=0.5)
    parser.add_argument("--psi", type=float, default=0.0)
    parser.add_argument("--tick-interval", type=float, default=0.001)
    parser.add_argument("--api-key", default=os.environ.get("SCPN_PHASE_WS_API_KEY"))
    parser.add_argument("--command-rate-limit", type=int, default=20)
    parser.add_argument("--command-rate-window-s", type=float, default=1.0)
    parser.add_argument("--tls-cert", default=None)
    parser.add_argument("--tls-key", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    mon = RealtimeMonitor.from_paper27(
        L=args.layers,
        N_per=args.n_per,
        zeta_uniform=args.zeta,
        psi_driver=args.psi,
    )
    tls_context = None
    if args.tls_cert or args.tls_key:
        if not args.tls_cert or not args.tls_key:
            raise ValueError("--tls-cert and --tls-key must be provided together")
        tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls_context.load_cert_chain(args.tls_cert, args.tls_key)
    server = PhaseStreamServer(
        monitor=mon,
        tick_interval_s=args.tick_interval,
        api_key=args.api_key,
        command_rate_limit=args.command_rate_limit,
        command_rate_window_s=args.command_rate_window_s,
    )
    server.serve_sync(host=args.host, port=args.port, ssl_context=tls_context)


if __name__ == "__main__":
    main()
