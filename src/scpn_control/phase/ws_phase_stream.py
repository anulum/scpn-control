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
import hashlib
import hmac
import json
import logging
import math
import numbers
import os
import ssl
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlsplit

from scpn_control.phase.realtime_monitor import RealtimeMonitor

logger = logging.getLogger(__name__)
_MIN_API_KEY_BYTES = 16
_KNOWN_ACTIONS = frozenset({"set_psi", "set_pac_gamma", "reset", "stop"})
_DEFAULT_ALLOWED_ACTIONS = ("set_psi", "set_pac_gamma", "reset", "stop")
_RUNTIME_CONFIG_DIGEST_KEY = b"scpn-control.phase-stream.runtime-config.v1"
WEBSOCKET_RUNTIME_EVIDENCE_SCHEMA_VERSION = "scpn-control.websocket-runtime-evidence.v1"
WEBSOCKET_RUNTIME_EVIDENCE_LOCAL_ONLY = "bounded_websocket_runtime_evidence_only"
WEBSOCKET_RUNTIME_EVIDENCE_QUALIFIED = "qualified_websocket_runtime_evidence"


@dataclass(frozen=True)
class WebSocketRuntimeEvidence:
    """Tamper-evident WebSocket runtime security and backpressure evidence."""

    schema_version: str
    generated_utc: str
    deployment_id: str
    bind_host: str
    uses_tls: bool
    require_client_auth: bool
    api_key_configured: bool
    api_key_min_bytes: int
    require_tls: bool
    allow_query_token_auth: bool
    allow_insecure_remote: bool
    command_rate_limit: int
    command_rate_window_s: float
    max_payload_bytes: int
    client_send_timeout_s: float
    max_clients: int
    max_client_write_buffer_bytes: int
    allowed_actions: tuple[str, ...]
    allowed_origins: tuple[str, ...]
    config_sha256: str
    auth_successes: int
    command_frames: int
    broadcast_frames: int
    peak_connected_clients: int
    authentication_failures: int
    origin_rejections: int
    payload_rejections: int
    rate_limit_rejections: int
    capacity_rejections: int
    action_rejections: int
    backpressure_disconnects: int
    facility_claim_allowed: bool
    claim_status: str
    payload_sha256: str


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


def _websocket_peer(websocket: Any) -> str:
    remote = getattr(websocket, "remote_address", None)
    if isinstance(remote, tuple) and remote:
        return str(remote[0])
    if isinstance(remote, str) and remote.strip():
        return remote.strip()
    return "unknown"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON key in WebSocket runtime evidence: {key}")
        seen.add(key)
        out[key] = value
    return out


def _require_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(name: str, value: object) -> int:
    result = _require_nonnegative_int(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _require_finite_positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _tuple_of_nonempty_strings(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ValueError(f"{name} must contain only non-empty strings")
    return result


def _ws_config_payload(
    server: "PhaseStreamServer",
    *,
    bind_host: str,
    uses_tls: bool,
) -> dict[str, Any]:
    return {
        "bind_host": bind_host,
        "uses_tls": uses_tls,
        "require_client_auth": bool(server.require_client_auth),
        "api_key_configured": server.api_key is not None,
        "api_key_min_bytes": _MIN_API_KEY_BYTES,
        "require_tls": bool(server.require_tls),
        "allow_query_token_auth": bool(server.allow_query_token_auth),
        "allow_insecure_remote": bool(server.allow_insecure_remote),
        "command_rate_limit": int(server.command_rate_limit),
        "command_rate_window_s": float(server.command_rate_window_s),
        "max_payload_bytes": int(server.max_payload_bytes),
        "client_send_timeout_s": float(server.client_send_timeout_s),
        "max_clients": int(server.max_clients),
        "max_client_write_buffer_bytes": int(server.max_client_write_buffer_bytes),
        "allowed_actions": tuple(server.allowed_actions),
        "allowed_origins": tuple(server.allowed_origins),
    }


def _runtime_config_sha256(payload: Mapping[str, Any]) -> str:
    return hmac.new(
        _RUNTIME_CONFIG_DIGEST_KEY,
        _canonical_json(dict(payload)).encode("utf-8"),
        "sha256",
    ).hexdigest()


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
    max_client_write_buffer_bytes: int = 262144
    require_client_auth: bool = True
    allow_query_token_auth: bool = False
    require_tls: bool = False
    allow_insecure_remote: bool = False
    allowed_origins: tuple[str, ...] = ()
    allowed_actions: tuple[str, ...] = _DEFAULT_ALLOWED_ACTIONS
    _clients: set = field(default_factory=set, init=False, repr=False)
    _client_windows: dict[Any, tuple[float, float]] = field(default_factory=dict, init=False, repr=False)
    _peer_windows: dict[str, tuple[float, float]] = field(default_factory=dict, init=False, repr=False)
    _runtime_counters: dict[str, int] = field(default_factory=dict, init=False, repr=False)
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
            or not isinstance(self.command_rate_limit, numbers.Integral)
            or self.command_rate_limit <= 0
        ):
            raise ValueError("command_rate_limit must be a positive integer")
        self.command_rate_limit = int(self.command_rate_limit)
        if not math.isfinite(self.command_rate_window_s) or self.command_rate_window_s <= 0.0:
            raise ValueError("command_rate_window_s must be finite and positive")
        if (
            isinstance(self.max_payload_bytes, bool)
            or not isinstance(self.max_payload_bytes, numbers.Integral)
            or self.max_payload_bytes <= 0
        ):
            raise ValueError("max_payload_bytes must be a positive integer")
        self.max_payload_bytes = int(self.max_payload_bytes)
        if not math.isfinite(self.client_send_timeout_s) or self.client_send_timeout_s <= 0.0:
            raise ValueError("client_send_timeout_s must be finite and positive")
        if (
            isinstance(self.max_clients, bool)
            or not isinstance(self.max_clients, numbers.Integral)
            or self.max_clients <= 0
        ):
            raise ValueError("max_clients must be a positive integer")
        self.max_clients = int(self.max_clients)
        if (
            isinstance(self.max_client_write_buffer_bytes, bool)
            or not isinstance(self.max_client_write_buffer_bytes, numbers.Integral)
            or self.max_client_write_buffer_bytes <= 0
        ):
            raise ValueError("max_client_write_buffer_bytes must be a positive integer")
        self.max_client_write_buffer_bytes = int(self.max_client_write_buffer_bytes)
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
        self._runtime_counters = {
            "auth_successes": 0,
            "command_frames": 0,
            "broadcast_frames": 0,
            "peak_connected_clients": 0,
            "authentication_failures": 0,
            "origin_rejections": 0,
            "payload_rejections": 0,
            "rate_limit_rejections": 0,
            "capacity_rejections": 0,
            "action_rejections": 0,
            "backpressure_disconnects": 0,
        }

    def _bump_runtime_counter(self, name: str, amount: int = 1) -> None:
        self._runtime_counters[name] = int(self._runtime_counters.get(name, 0)) + int(amount)

    def runtime_counters(self) -> dict[str, int]:
        """Return a copy of WebSocket runtime security and backpressure counters."""

        return dict(self._runtime_counters)

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

    def _audit_security_event(self, websocket: Any, event: str, reason: str) -> None:
        counter = {
            "authentication_failed": "authentication_failures",
            "origin_rejected": "origin_rejections",
            "payload_rejected": "payload_rejections",
            "rate_limit_rejected": "rate_limit_rejections",
            "capacity_rejected": "capacity_rejections",
            "action_rejected": "action_rejections",
        }.get(event)
        if counter is not None:
            self._bump_runtime_counter(counter)
        logger.warning(
            "phase_stream_security_event event=%s peer=%s reason=%s",
            event,
            _websocket_peer(websocket),
            reason,
        )

    def _bucket_rate_limited(self, buckets: dict[Any, tuple[float, float]], key: Any, now: float) -> bool:
        capacity = float(self.command_rate_limit)
        refill_rate = capacity / self.command_rate_window_s
        updated_at, tokens = buckets.get(key, (now, capacity))
        elapsed = max(0.0, now - updated_at)
        tokens = min(capacity, tokens + elapsed * refill_rate)
        limited = tokens < 1.0
        if not limited:
            tokens -= 1.0
        buckets[key] = (now, tokens)
        return limited

    def _rate_limited(self, websocket: Any) -> bool:
        now = time.monotonic()
        client_limited = self._bucket_rate_limited(self._client_windows, websocket, now)
        peer_limited = self._bucket_rate_limited(self._peer_windows, _websocket_peer(websocket), now)
        return client_limited or peer_limited

    async def _handler(self, websocket: Any) -> None:
        if not self._origin_allowed(websocket):
            self._audit_security_event(websocket, "origin_rejected", "origin not allowed")
            await self._close_client(websocket, code=1008, reason="origin not allowed")
            return
        if not self._authorised(websocket):
            self._audit_security_event(websocket, "authentication_failed", "authentication required")
            await self._close_client(websocket, code=1008, reason="authentication required")
            return
        if len(self._clients) >= self.max_clients:
            self._audit_security_event(websocket, "capacity_rejected", "client capacity exceeded")
            await self._close_client(websocket, code=1013, reason="client capacity exceeded")
            return
        self._clients.add(websocket)
        self._bump_runtime_counter("auth_successes")
        self._runtime_counters["peak_connected_clients"] = max(
            int(self._runtime_counters.get("peak_connected_clients", 0)),
            len(self._clients),
        )
        logger.info("Client connected (%d total)", len(self._clients))
        try:
            async for msg in websocket:
                if _payload_size_bytes(msg) > self.max_payload_bytes:
                    self._audit_security_event(websocket, "payload_rejected", "payload too large")
                    if not await self._send_response(websocket, {"error": "payload_too_large"}):
                        break
                    await self._close_client(websocket, code=1009, reason="payload too large")
                    break
                if self._rate_limited(websocket):
                    self._audit_security_event(websocket, "rate_limit_rejected", "command rate limit exceeded")
                    if not await self._send_response(websocket, {"error": "rate_limited"}):
                        break
                    await self._close_client(websocket, code=1008, reason="command rate limit exceeded")
                    break
                try:
                    cmd = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                if not isinstance(cmd, dict):
                    continue
                action = cmd.get("action")
                if action not in self.allowed_actions:
                    self._audit_security_event(websocket, "action_rejected", "action not allowed")
                    if not await self._send_response(websocket, {"error": "action_not_allowed"}):
                        break
                    continue
                if action == "set_psi":
                    value = _finite_command_value(cmd)
                    if value is not None:
                        self.monitor.psi_driver = value
                        self._bump_runtime_counter("command_frames")
                elif action == "set_pac_gamma":
                    value = _finite_command_value(cmd)
                    if value is not None:
                        self.monitor.pac_gamma = value
                        self._bump_runtime_counter("command_frames")
                elif action == "reset":
                    self.monitor.reset(seed=cmd.get("seed", 42))
                    self._bump_runtime_counter("command_frames")
                elif action == "stop":
                    self._running = False
                    self._bump_runtime_counter("command_frames")
        finally:
            self._clients.discard(websocket)
            self._client_windows.pop(websocket, None)
            logger.info("Client disconnected (%d remain)", len(self._clients))

    async def _close_client(self, websocket: Any, *, code: int, reason: str) -> None:
        close = getattr(websocket, "close", None)
        if close is None:
            return
        with contextlib.suppress(Exception):
            await asyncio.wait_for(close(code=code, reason=reason), timeout=self.client_send_timeout_s)

    async def _close_backpressured_client(self, websocket: Any) -> None:
        self._bump_runtime_counter("backpressure_disconnects")
        await self._close_client(websocket, code=1011, reason="broadcast backpressure timeout")

    async def _send_frame_or_dead(self, websocket: Any, frame: str) -> Any | None:
        try:
            await asyncio.wait_for(websocket.send(frame), timeout=self.client_send_timeout_s)
            transport = getattr(websocket, "transport", None)
            get_write_buffer_size = getattr(transport, "get_write_buffer_size", None)
            if get_write_buffer_size is not None:
                buffered = int(get_write_buffer_size())
                if buffered > self.max_client_write_buffer_bytes:
                    await self._close_backpressured_client(websocket)
                    return websocket
        except asyncio.TimeoutError:
            await self._close_backpressured_client(websocket)
            return websocket
        except (ConnectionError, OSError):
            return websocket
        return None

    async def _send_response(self, websocket: Any, payload: dict[str, str]) -> bool:
        return await self._send_frame_or_dead(websocket, json.dumps(payload)) is None

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
            self._bump_runtime_counter("broadcast_frames", len(send_results) - len(dead))
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
            max_queue=4,
            write_limit=self.max_client_write_buffer_bytes,
        ):
            scheme = "wss" if ssl_context is not None else "ws"
            logger.info("Phase stream listening on %s://%s:%d", scheme, host, port)
            await tick_task

    def serve_sync(self, host: str = "127.0.0.1", port: int = 8765, ssl_context: ssl.SSLContext | None = None) -> None:
        """Blocking entry point."""
        asyncio.run(self.serve(host, port, ssl_context))


def _validate_websocket_runtime_payload(
    payload: Mapping[str, Any],
    *,
    require_facility_claim: bool,
) -> WebSocketRuntimeEvidence:
    if payload.get("schema_version") != WEBSOCKET_RUNTIME_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("WebSocket runtime evidence schema_version is unsupported")
    declared_digest = payload.get("payload_sha256")
    if not _is_sha256(declared_digest):
        raise ValueError("WebSocket runtime evidence payload_sha256 must be a SHA-256 hex digest")
    if declared_digest != _payload_sha256(payload):
        raise ValueError("WebSocket runtime evidence payload_sha256 does not match payload")
    generated_utc = payload.get("generated_utc")
    if not isinstance(generated_utc, str) or not generated_utc.endswith("Z"):
        raise ValueError("WebSocket runtime evidence generated_utc must be a UTC timestamp ending in Z")
    deployment_id = payload.get("deployment_id")
    bind_host = payload.get("bind_host")
    if not isinstance(deployment_id, str) or not deployment_id.strip():
        raise ValueError("deployment_id must be non-empty")
    if not isinstance(bind_host, str) or not bind_host.strip():
        raise ValueError("bind_host must be non-empty")
    bool_fields = (
        "uses_tls",
        "require_client_auth",
        "api_key_configured",
        "require_tls",
        "allow_query_token_auth",
        "allow_insecure_remote",
        "facility_claim_allowed",
    )
    for name in bool_fields:
        if not isinstance(payload.get(name), bool):
            raise ValueError(f"{name} must be boolean")
    api_key_min_bytes = _require_positive_int("api_key_min_bytes", payload.get("api_key_min_bytes"))
    command_rate_limit = _require_positive_int("command_rate_limit", payload.get("command_rate_limit"))
    max_payload_bytes = _require_positive_int("max_payload_bytes", payload.get("max_payload_bytes"))
    max_clients = _require_positive_int("max_clients", payload.get("max_clients"))
    max_client_write_buffer_bytes = _require_positive_int(
        "max_client_write_buffer_bytes",
        payload.get("max_client_write_buffer_bytes"),
    )
    command_rate_window_s = _require_finite_positive(
        "command_rate_window_s",
        payload.get("command_rate_window_s"),
    )
    client_send_timeout_s = _require_finite_positive(
        "client_send_timeout_s",
        payload.get("client_send_timeout_s"),
    )
    allowed_actions = _tuple_of_nonempty_strings("allowed_actions", payload.get("allowed_actions"))
    if not allowed_actions or set(allowed_actions) - _KNOWN_ACTIONS:
        raise ValueError("allowed_actions must contain known WebSocket commands")
    allowed_origins = _tuple_of_nonempty_strings("allowed_origins", payload.get("allowed_origins"))
    config_payload = {
        "bind_host": bind_host,
        "uses_tls": bool(payload["uses_tls"]),
        "require_client_auth": bool(payload["require_client_auth"]),
        "api_key_configured": bool(payload["api_key_configured"]),
        "api_key_min_bytes": api_key_min_bytes,
        "require_tls": bool(payload["require_tls"]),
        "allow_query_token_auth": bool(payload["allow_query_token_auth"]),
        "allow_insecure_remote": bool(payload["allow_insecure_remote"]),
        "command_rate_limit": command_rate_limit,
        "command_rate_window_s": command_rate_window_s,
        "max_payload_bytes": max_payload_bytes,
        "client_send_timeout_s": client_send_timeout_s,
        "max_clients": max_clients,
        "max_client_write_buffer_bytes": max_client_write_buffer_bytes,
        "allowed_actions": allowed_actions,
        "allowed_origins": allowed_origins,
    }
    config_sha256 = payload.get("config_sha256")
    if not _is_sha256(config_sha256):
        raise ValueError("config_sha256 must be a SHA-256 hex digest")
    if config_sha256 != _runtime_config_sha256(config_payload):
        raise ValueError("config_sha256 does not match WebSocket runtime configuration")
    counter_values = {
        name: _require_nonnegative_int(name, payload.get(name))
        for name in (
            "auth_successes",
            "command_frames",
            "broadcast_frames",
            "peak_connected_clients",
            "authentication_failures",
            "origin_rejections",
            "payload_rejections",
            "rate_limit_rejections",
            "capacity_rejections",
            "action_rejections",
            "backpressure_disconnects",
        )
    }
    facility_claim_allowed = bool(payload["facility_claim_allowed"])
    expected_status = (
        WEBSOCKET_RUNTIME_EVIDENCE_QUALIFIED if facility_claim_allowed else WEBSOCKET_RUNTIME_EVIDENCE_LOCAL_ONLY
    )
    if payload.get("claim_status") != expected_status:
        raise ValueError("WebSocket runtime evidence claim_status does not match facility_claim_allowed")
    if require_facility_claim:
        if not facility_claim_allowed:
            raise ValueError("WebSocket runtime evidence is local-only and cannot support a facility claim")
        if not bool(payload["require_client_auth"]) or not bool(payload["api_key_configured"]):
            raise ValueError("qualified WebSocket evidence requires authenticated clients")
        if api_key_min_bytes < _MIN_API_KEY_BYTES:
            raise ValueError("qualified WebSocket evidence requires a hardened API-key policy")
        if not bool(payload["uses_tls"]):
            raise ValueError("qualified WebSocket evidence requires TLS")
        if not bool(payload["require_tls"]):
            raise ValueError("qualified WebSocket evidence requires TLS enforcement")
        if bool(payload["allow_query_token_auth"]):
            raise ValueError("qualified WebSocket evidence forbids query-token authentication")
        if bool(payload["allow_insecure_remote"]):
            raise ValueError("qualified WebSocket evidence forbids insecure remote binding")
        if max_payload_bytes > 65536:
            raise ValueError("qualified WebSocket evidence requires payload cap <= 65536 bytes")
        if counter_values["auth_successes"] <= 0:
            raise ValueError("qualified WebSocket evidence requires at least one authenticated session")
        if counter_values["command_frames"] <= 0:
            raise ValueError("qualified WebSocket evidence requires observed authenticated commands")
        if counter_values["broadcast_frames"] <= 0:
            raise ValueError("qualified WebSocket evidence requires observed broadcast frames")
        if counter_values["peak_connected_clients"] <= 0:
            raise ValueError("qualified WebSocket evidence requires observed client connectivity")
        if counter_values["backpressure_disconnects"] != 0:
            raise ValueError("qualified WebSocket evidence cannot contain backpressure disconnects")
    return WebSocketRuntimeEvidence(
        schema_version=str(payload["schema_version"]),
        generated_utc=generated_utc,
        deployment_id=deployment_id,
        bind_host=bind_host,
        uses_tls=bool(payload["uses_tls"]),
        require_client_auth=bool(payload["require_client_auth"]),
        api_key_configured=bool(payload["api_key_configured"]),
        api_key_min_bytes=api_key_min_bytes,
        require_tls=bool(payload["require_tls"]),
        allow_query_token_auth=bool(payload["allow_query_token_auth"]),
        allow_insecure_remote=bool(payload["allow_insecure_remote"]),
        command_rate_limit=command_rate_limit,
        command_rate_window_s=command_rate_window_s,
        max_payload_bytes=max_payload_bytes,
        client_send_timeout_s=client_send_timeout_s,
        max_clients=max_clients,
        max_client_write_buffer_bytes=max_client_write_buffer_bytes,
        allowed_actions=allowed_actions,
        allowed_origins=allowed_origins,
        config_sha256=str(config_sha256),
        auth_successes=counter_values["auth_successes"],
        command_frames=counter_values["command_frames"],
        broadcast_frames=counter_values["broadcast_frames"],
        peak_connected_clients=counter_values["peak_connected_clients"],
        authentication_failures=counter_values["authentication_failures"],
        origin_rejections=counter_values["origin_rejections"],
        payload_rejections=counter_values["payload_rejections"],
        rate_limit_rejections=counter_values["rate_limit_rejections"],
        capacity_rejections=counter_values["capacity_rejections"],
        action_rejections=counter_values["action_rejections"],
        backpressure_disconnects=counter_values["backpressure_disconnects"],
        facility_claim_allowed=facility_claim_allowed,
        claim_status=str(payload["claim_status"]),
        payload_sha256=str(declared_digest),
    )


def websocket_runtime_evidence(
    server: PhaseStreamServer,
    *,
    deployment_id: str,
    bind_host: str = "127.0.0.1",
    uses_tls: bool = False,
    generated_utc: str | None = None,
    counters: Mapping[str, int] | None = None,
    facility_claim_allowed: bool = False,
) -> WebSocketRuntimeEvidence:
    """Build tamper-evident runtime evidence for a phase WebSocket deployment."""

    if not isinstance(server, PhaseStreamServer):
        raise ValueError("server must be PhaseStreamServer")
    if not isinstance(deployment_id, str) or not deployment_id.strip():
        raise ValueError("deployment_id must be non-empty")
    if not isinstance(bind_host, str) or not bind_host.strip():
        raise ValueError("bind_host must be non-empty")
    if not isinstance(uses_tls, bool):
        raise ValueError("uses_tls must be boolean")
    runtime_counters = server.runtime_counters()
    if counters is not None:
        runtime_counters.update({str(key): int(value) for key, value in counters.items()})
    config = _ws_config_payload(server, bind_host=bind_host.strip(), uses_tls=uses_tls)
    payload: dict[str, Any] = {
        "schema_version": WEBSOCKET_RUNTIME_EVIDENCE_SCHEMA_VERSION,
        "generated_utc": generated_utc or _utc_now(),
        "deployment_id": deployment_id.strip(),
        **config,
        "config_sha256": _runtime_config_sha256(config),
        "auth_successes": int(runtime_counters.get("auth_successes", 0)),
        "command_frames": int(runtime_counters.get("command_frames", 0)),
        "broadcast_frames": int(runtime_counters.get("broadcast_frames", 0)),
        "peak_connected_clients": int(runtime_counters.get("peak_connected_clients", 0)),
        "authentication_failures": int(runtime_counters.get("authentication_failures", 0)),
        "origin_rejections": int(runtime_counters.get("origin_rejections", 0)),
        "payload_rejections": int(runtime_counters.get("payload_rejections", 0)),
        "rate_limit_rejections": int(runtime_counters.get("rate_limit_rejections", 0)),
        "capacity_rejections": int(runtime_counters.get("capacity_rejections", 0)),
        "action_rejections": int(runtime_counters.get("action_rejections", 0)),
        "backpressure_disconnects": int(runtime_counters.get("backpressure_disconnects", 0)),
        "facility_claim_allowed": bool(facility_claim_allowed),
        "claim_status": (
            WEBSOCKET_RUNTIME_EVIDENCE_QUALIFIED if facility_claim_allowed else WEBSOCKET_RUNTIME_EVIDENCE_LOCAL_ONLY
        ),
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return _validate_websocket_runtime_payload(payload, require_facility_claim=bool(facility_claim_allowed))


def assert_websocket_runtime_claim_admissible(
    evidence: WebSocketRuntimeEvidence,
) -> WebSocketRuntimeEvidence:
    """Fail closed unless WebSocket evidence supports a facility runtime claim."""

    return _validate_websocket_runtime_payload(asdict(evidence), require_facility_claim=True)


def save_websocket_runtime_evidence(evidence: WebSocketRuntimeEvidence, output_path: str | Path) -> None:
    """Persist WebSocket runtime evidence as sorted JSON without secret material."""

    if not isinstance(evidence, WebSocketRuntimeEvidence):
        raise ValueError("evidence must be WebSocketRuntimeEvidence")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_websocket_runtime_evidence(
    path: str | Path,
    *,
    require_facility_claim: bool = False,
) -> WebSocketRuntimeEvidence:
    """Load WebSocket runtime evidence with duplicate-key and digest admission."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError("WebSocket runtime evidence must be a JSON object")
    return _validate_websocket_runtime_payload(payload, require_facility_claim=require_facility_claim)


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
    parser.add_argument("--max-client-write-buffer-bytes", type=int, default=262144)
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
        max_client_write_buffer_bytes=args.max_client_write_buffer_bytes,
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
