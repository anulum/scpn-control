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
import contextlib
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
_MIN_API_KEY_BYTES = 16
_KNOWN_ACTIONS = frozenset({"set_psi", "set_pac_gamma", "reset", "stop"})
_DEFAULT_ALLOWED_ACTIONS = ("set_psi", "set_pac_gamma", "reset", "stop")


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


def _payload_size_bytes(message: object) -> int:
    if isinstance(message, (bytes, bytearray)):
        return len(message)
    if isinstance(message, str):
        return len(message.encode("utf-8"))
    return len(str(message).encode("utf-8"))


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


def _api_key_is_strong(api_key: str) -> bool:
    return len(api_key.encode("utf-8")) >= _MIN_API_KEY_BYTES


@dataclass
class PhaseStreamServer:
    """Async WebSocket server wrapping a RealtimeMonitor."""

    monitor: RealtimeMonitor
    tick_interval_s: float = 0.001
    api_key: str | None = None
    command_rate_limit: int = 20
    command_rate_window_s: float = 1.0
    max_payload_bytes: int = 65536
    client_send_timeout_s: float = 0.25
    max_clients: int = 128
    require_client_auth: bool = True
    allow_query_token_auth: bool = False
    require_tls: bool = False
    allow_insecure_remote: bool = False
    allowed_origins: tuple[str, ...] = ()
    allowed_actions: tuple[str, ...] = _DEFAULT_ALLOWED_ACTIONS
    _clients: set = field(default_factory=set, init=False, repr=False)
    _client_windows: dict[Any, tuple[float, int]] = field(default_factory=dict, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.tick_interval_s) or self.tick_interval_s <= 0.0:
            raise ValueError("tick_interval_s must be finite and positive")
        if self.api_key is not None and not self.api_key:
            raise ValueError("api_key must be non-empty when configured")
        if self.api_key is not None and not _api_key_is_strong(self.api_key):
            raise ValueError(f"api_key must be at least {_MIN_API_KEY_BYTES} bytes")
        if (
            isinstance(self.command_rate_limit, bool)
            or not isinstance(self.command_rate_limit, int)
            or self.command_rate_limit <= 0
        ):
            raise ValueError("command_rate_limit must be a positive integer")
        if not math.isfinite(self.command_rate_window_s) or self.command_rate_window_s <= 0.0:
            raise ValueError("command_rate_window_s must be finite and positive")
        if (
            isinstance(self.max_payload_bytes, bool)
            or not isinstance(self.max_payload_bytes, int)
            or self.max_payload_bytes <= 0
        ):
            raise ValueError("max_payload_bytes must be a positive integer")
        if not math.isfinite(self.client_send_timeout_s) or self.client_send_timeout_s <= 0.0:
            raise ValueError("client_send_timeout_s must be finite and positive")
        if isinstance(self.max_clients, bool) or not isinstance(self.max_clients, int) or self.max_clients <= 0:
            raise ValueError("max_clients must be a positive integer")
        if not isinstance(self.require_client_auth, bool):
            raise ValueError("require_client_auth must be a boolean")
        if not isinstance(self.allow_query_token_auth, bool):
            raise ValueError("allow_query_token_auth must be a boolean")
        if not isinstance(self.require_tls, bool):
            raise ValueError("require_tls must be a boolean")
        if not isinstance(self.allow_insecure_remote, bool):
            raise ValueError("allow_insecure_remote must be a boolean")
        self.allowed_origins = tuple(str(origin).strip() for origin in self.allowed_origins)
        if any(not origin for origin in self.allowed_origins):
            raise ValueError("allowed_origins must contain only non-empty origins")
        self.allowed_actions = tuple(str(action).strip() for action in self.allowed_actions)
        if not self.allowed_actions:
            raise ValueError("allowed_actions must not be empty")
        unknown_actions = sorted(set(self.allowed_actions) - _KNOWN_ACTIONS)
        if unknown_actions:
            raise ValueError(f"allowed_actions contains unknown actions: {unknown_actions}")

    def _origin_allowed(self, websocket: Any) -> bool:
        origin = _get_header(websocket, "Origin")
        if origin is None:
            return True
        return origin in self.allowed_origins

    def _authorised(self, websocket: Any) -> bool:
        if self.api_key is None:
            return not self.require_client_auth
        candidate = _bearer_token(_get_header(websocket, "Authorization"))
        candidate = candidate or _get_header(websocket, "X-SCPN-API-Key")
        if candidate is None and self.allow_query_token_auth:
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
        if not self._origin_allowed(websocket):
            await websocket.close(code=1008, reason="origin not allowed")
            return
        if not self._authorised(websocket):
            await websocket.close(code=1008, reason="authentication required")
            return
        if len(self._clients) >= self.max_clients:
            await websocket.close(code=1013, reason="client capacity exceeded")
            return
        self._clients.add(websocket)
        logger.info("Client connected (%d total)", len(self._clients))
        try:
            async for msg in websocket:
                if _payload_size_bytes(msg) > self.max_payload_bytes:
                    await websocket.send(json.dumps({"error": "payload_too_large"}))
                    await websocket.close(code=1009, reason="payload too large")
                    break
                if self._rate_limited(websocket):
                    await websocket.send(json.dumps({"error": "rate_limited"}))
                    await websocket.close(code=1008, reason="command rate limit exceeded")
                    break
                try:
                    cmd = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                if not isinstance(cmd, dict):
                    continue
                action = cmd.get("action")
                if action not in self.allowed_actions:
                    await websocket.send(json.dumps({"error": "action_not_allowed"}))
                    continue
                if action == "set_psi":
                    value = _finite_command_value(cmd)
                    if value is not None:
                        self.monitor.psi_driver = value
                elif action == "set_pac_gamma":
                    value = _finite_command_value(cmd)
                    if value is not None:
                        self.monitor.pac_gamma = value
                elif action == "reset":
                    self.monitor.reset(seed=cmd.get("seed", 42))
                elif action == "stop":
                    self._running = False
        finally:
            self._clients.discard(websocket)
            self._client_windows.pop(websocket, None)
            logger.info("Client disconnected (%d remain)", len(self._clients))

    async def _close_backpressured_client(self, websocket: Any) -> None:
        close = getattr(websocket, "close", None)
        if close is None:
            return
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                close(code=1011, reason="broadcast backpressure timeout"),
                timeout=self.client_send_timeout_s,
            )

    async def _send_frame_or_dead(self, websocket: Any, frame: str) -> Any | None:
        try:
            await asyncio.wait_for(websocket.send(frame), timeout=self.client_send_timeout_s)
        except TimeoutError:
            await self._close_backpressured_client(websocket)
            return websocket
        except (ConnectionError, OSError):
            return websocket
        return None

    async def _tick_loop(self) -> None:
        self._running = True
        while self._running:
            if not self._clients:
                await asyncio.sleep(0.05)
                continue
            snap = self.monitor.tick()
            frame = json.dumps(snap)
            send_results = await asyncio.gather(
                *(self._send_frame_or_dead(ws, frame) for ws in tuple(self._clients)),
            )
            dead = {ws for ws in send_results if ws is not None}
            self._clients -= dead
            await asyncio.sleep(self.tick_interval_s)

    async def serve(self, host: str = "127.0.0.1", port: int = 8765, ssl_context: ssl.SSLContext | None = None) -> None:
        """Start WebSocket server and tick loop."""
        if self.require_client_auth and self.api_key is None:
            raise ValueError("api_key is required unless require_client_auth is disabled")
        if self.require_tls and ssl_context is None:
            raise ValueError("ssl_context is required when require_tls is enabled")
        if not _is_loopback_host(host) and self.api_key is None:
            raise ValueError("api_key is required when binding the phase stream outside loopback")
        if not _is_loopback_host(host) and ssl_context is None and not self.allow_insecure_remote:
            raise ValueError("ssl_context is required for non-loopback binds unless allow_insecure_remote is enabled")
        try:
            import websockets  # noqa: F811
        except ImportError as exc:
            raise ImportError("pip install websockets") from exc

        tick_task = asyncio.create_task(self._tick_loop())
        async with websockets.serve(
            self._handler,
            host,
            port,
            ssl=ssl_context,
            max_size=self.max_payload_bytes,
        ):
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
    parser.add_argument(
        "--max-payload-bytes",
        type=int,
        default=65536,
    )
    parser.add_argument("--client-send-timeout-s", type=float, default=0.25)
    parser.add_argument("--max-clients", type=int, default=128)
    parser.add_argument("--require-tls", action="store_true")
    parser.add_argument("--allow-unauthenticated-clients", action="store_true")
    parser.add_argument("--allow-query-token-auth", action="store_true")
    parser.add_argument("--allow-insecure-remote", action="store_true")
    parser.add_argument("--allowed-origin", action="append", default=[])
    parser.add_argument("--allowed-action", action="append", choices=sorted(_KNOWN_ACTIONS), default=None)
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
        max_payload_bytes=args.max_payload_bytes,
        client_send_timeout_s=args.client_send_timeout_s,
        max_clients=args.max_clients,
        require_client_auth=not args.allow_unauthenticated_clients,
        allow_query_token_auth=args.allow_query_token_auth,
        require_tls=args.require_tls,
        allow_insecure_remote=args.allow_insecure_remote,
        allowed_origins=tuple(args.allowed_origin),
        allowed_actions=tuple(args.allowed_action) if args.allowed_action else _DEFAULT_ALLOWED_ACTIONS,
    )
    server.serve_sync(host=args.host, port=args.port, ssl_context=tls_context)


if __name__ == "__main__":
    main()
