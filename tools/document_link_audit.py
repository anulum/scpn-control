# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Deterministic local and governed external document-link audit.

"""Audit public documentation links without exposing private operational paths."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import html
import ipaddress
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10 CI.
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "tools" / "document_link_policy.toml"
SCHEMA = "scpn-control.document-link-audit.v1"

_FENCE_RE = re.compile(r"(?ms)^\s*(```|~~~).*?^\s*\1\s*$")
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]\n]*\]\(\s*(<[^>\n]+>|[^\s)]+)")
_MARKDOWN_REFERENCE_RE = re.compile(r"(?m)^\s*\[[^\]\n]+\]:\s*(<[^>\n]+>|\S+)")
_HTML_LINK_RE = re.compile(r"(?i)\b(?:href|src)\s*=\s*([\"'])(.*?)\1")
_TEX_LINK_RE = re.compile(r"\\(?:href|url)\{([^}]+)\}")
_TEX_FILE_RE = re.compile(r"\\(?:includegraphics|input|include|bibliography)\{([^}]+)\}")
_URL_RE = re.compile(r"https?://[^\s<>\"'`]+", re.IGNORECASE)
_MKDOCS_NAV_RE = re.compile(r"(?m)^\s*-\s+[^:#\n]+:\s*([^#\s]+(?:#[^\s]+)?)\s*$")
_METADATA_FILE_RE = re.compile(
    r"(?P<quote>[\"'])(?P<target>(?:(?:\.\.?)/)*[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*"
    r"\.(?:md|markdown|tex|bib|cff|json|toml|ya?ml|html?|png|jpe?g|svg|pdf))(?P=quote)"
)
_SECRET_NAME_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, order=True)
class LinkRef:
    """One extracted link and its public source location."""

    source: str
    line: int
    target: str
    kind: str


@dataclass(frozen=True, order=True)
class Finding:
    """One deterministic local-link or URL-policy failure."""

    source: str
    line: int
    target: str
    reason: str


@dataclass(frozen=True)
class ExternalResult:
    """One bounded external crawl outcome."""

    url: str
    classification: str
    status_code: int | None
    attempts: int
    checked_at: str
    final_url: str | None
    detail: str
    cached: bool = False


@dataclass(frozen=True)
class Policy:
    """Validated link-audit policy."""

    source_sha256: str
    include_suffixes: tuple[str, ...]
    exclude_prefixes: tuple[str, ...]
    exclude_globs: tuple[str, ...]
    require_tracked_targets: bool
    check_markdown_anchors: bool
    allowed_schemes: tuple[str, ...]
    ignored_schemes: tuple[str, ...]
    timeout_seconds: float
    retries: int
    retry_backoff_seconds: float
    per_host_delay_seconds: float
    cache_ttl_seconds: int
    transient_cache_ttl_seconds: int
    max_urls: int
    user_agent: str
    transient_statuses: frozenset[int]
    restricted_statuses: frozenset[int]
    permanent_statuses: frozenset[int]
    secret_query_keys: frozenset[str]


class UnsafeExternalTarget(RuntimeError):
    """Raised before crawling or recording a non-public redirect target."""


def _read_policy(path: Path) -> Policy:
    source_bytes = path.read_bytes()
    payload = tomllib.loads(source_bytes.decode("utf-8"))
    scan = payload["scan"]
    local = payload["local"]
    external = payload["external"]
    return Policy(
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        include_suffixes=tuple(str(item) for item in scan["include_suffixes"]),
        exclude_prefixes=tuple(str(item) for item in scan["exclude_prefixes"]),
        exclude_globs=tuple(str(item) for item in scan["exclude_globs"]),
        require_tracked_targets=bool(local["require_tracked_targets"]),
        check_markdown_anchors=bool(local["check_markdown_anchors"]),
        allowed_schemes=tuple(str(item) for item in external["allowed_schemes"]),
        ignored_schemes=tuple(str(item) for item in external["ignored_schemes"]),
        timeout_seconds=float(external["timeout_seconds"]),
        retries=int(external["retries"]),
        retry_backoff_seconds=float(external["retry_backoff_seconds"]),
        per_host_delay_seconds=float(external["per_host_delay_seconds"]),
        cache_ttl_seconds=int(external["cache_ttl_seconds"]),
        transient_cache_ttl_seconds=int(external["transient_cache_ttl_seconds"]),
        max_urls=int(external["max_urls"]),
        user_agent=str(external["user_agent"]),
        transient_statuses=frozenset(int(item) for item in external["transient_statuses"]),
        restricted_statuses=frozenset(int(item) for item in external["restricted_statuses"]),
        permanent_statuses=frozenset(int(item) for item in external["permanent_statuses"]),
        secret_query_keys=frozenset(str(item) for item in external["secret_query_keys"]),
    )


def _tracked_paths(root: Path) -> tuple[Path, ...]:
    result = subprocess.run(  # noqa: S603
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return tuple(root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item)


def _is_public_source(relative: str, policy: Policy) -> bool:
    if relative.startswith(policy.exclude_prefixes):
        return False
    if any(fnmatch.fnmatch(relative, pattern) for pattern in policy.exclude_globs):
        return False
    return Path(relative).suffix.lower() in policy.include_suffixes


def public_sources(root: Path, policy: Policy) -> tuple[Path, ...]:
    """Return deterministic tracked public text sources covered by the audit."""
    return tuple(path for path in _tracked_paths(root) if _is_public_source(path.relative_to(root).as_posix(), policy))


def _prose(text: str) -> str:
    def blank(match: re.Match[str]) -> str:
        return "\n" * match.group(0).count("\n")

    return _INLINE_CODE_RE.sub("", _FENCE_RE.sub(blank, text))


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _clean_target(raw: str, *, trim_prose_punctuation: bool = False) -> str:
    target = html.unescape(raw.strip())
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if trim_prose_punctuation:
        target = target.rstrip(".,;:!?")
        for opening, closing in (("(", ")"), ("[", "]"), ("{", "}")):
            while target.endswith(closing) and target.count(closing) > target.count(opening):
                target = target[:-1]
    return target


def _append_matches(
    refs: set[LinkRef], source: str, text: str, pattern: re.Pattern[str], kind: str, group: int | str
) -> None:
    for match in pattern.finditer(text):
        target = _clean_target(match.group(group), trim_prose_punctuation=kind == "external")
        if target:
            refs.add(LinkRef(source, _line(text, match.start(group)), target, kind))


def _mask_matches(text: str, pattern: re.Pattern[str]) -> str:
    """Blank structured matches without changing offsets or line numbers."""

    def blank(match: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))

    return pattern.sub(blank, text)


def _mkdocs_nav_text(text: str) -> str:
    """Keep only the top-level ``nav`` block while preserving line numbers."""
    lines = text.splitlines(keepends=True)
    in_nav = False
    output: list[str] = []
    for line in lines:
        if line.rstrip() == "nav:":
            in_nav = True
            output.append(line)
            continue
        if in_nav and line and not line[0].isspace() and line.strip() and not line.lstrip().startswith("#"):
            in_nav = False
        output.append(line if in_nav else ("\n" if line.endswith("\n") else ""))
    return "".join(output)


def extract_links(path: Path, root: Path) -> tuple[LinkRef, ...]:
    """Extract local and external references from one tracked public source."""
    relative = path.relative_to(root).as_posix()
    original = path.read_text(encoding="utf-8", errors="strict")
    text = _prose(original)
    external_text = text
    refs: set[LinkRef] = set()
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        _append_matches(refs, relative, text, _MARKDOWN_LINK_RE, "markdown", 1)
        _append_matches(refs, relative, text, _MARKDOWN_REFERENCE_RE, "markdown-reference", 1)
    if suffix in {".html", ".htm"}:
        _append_matches(refs, relative, text, _HTML_LINK_RE, "html", 2)
    if suffix == ".tex":
        _append_matches(refs, relative, text, _TEX_LINK_RE, "tex-url", 1)
        _append_matches(refs, relative, text, _TEX_FILE_RE, "tex-file", 1)
        external_text = _mask_matches(text, _TEX_LINK_RE)
    if relative == "mkdocs.yml":
        _append_matches(refs, relative, _mkdocs_nav_text(text), _MKDOCS_NAV_RE, "mkdocs-nav", 1)
    if relative.startswith("papers/submissions/") and suffix in {".json", ".toml", ".yaml", ".yml", ".cff"}:
        _append_matches(refs, relative, text, _METADATA_FILE_RE, "metadata-file", "target")
    _append_matches(refs, relative, external_text, _URL_RE, "external", 0)
    return tuple(sorted(refs))


def _normalise_secret_name(name: str) -> str:
    return _SECRET_NAME_RE.sub("_", name.lower()).strip("_")


def _redact_url(url: str, policy: Policy) -> str:
    parts = urlsplit(url)
    host = parts.hostname or ""
    try:
        port = f":{parts.port}" if parts.port is not None else ""
    except ValueError:
        port = ""
    netloc = f"[REDACTED]@{host}{port}" if parts.username is not None or parts.password is not None else parts.netloc
    query = urlencode(
        [
            (name, "[REDACTED]" if _normalise_secret_name(name) in policy.secret_query_keys else value)
            for name, value in parse_qsl(parts.query, keep_blank_values=True)
        ],
        doseq=True,
    )
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def _url_policy_finding(ref: LinkRef, policy: Policy) -> Finding | None:
    parts = urlsplit(ref.target)
    if parts.scheme.lower() not in policy.allowed_schemes:
        return None
    if parts.username is not None or parts.password is not None:
        return Finding(ref.source, ref.line, _redact_url(ref.target, policy), "URL contains user-info credentials")
    host = parts.hostname
    if not host:
        return Finding(ref.source, ref.line, ref.target, "URL has no public host")
    normalised_host = host.rstrip(".").lower()
    if normalised_host == "localhost" or normalised_host.endswith((".localhost", ".local", ".internal")):
        return Finding(ref.source, ref.line, ref.target, "URL targets a non-public host")
    try:
        address = ipaddress.ip_address(normalised_host)
    except ValueError:
        pass
    else:
        if not address.is_global:
            return Finding(ref.source, ref.line, ref.target, "URL targets a non-public IP address")
    keys = {_normalise_secret_name(name) for name, _ in parse_qsl(parts.query, keep_blank_values=True)}
    secret_keys = keys & policy.secret_query_keys
    if secret_keys:
        names = ", ".join(sorted(secret_keys))
        return Finding(
            ref.source,
            ref.line,
            _redact_url(ref.target, policy),
            f"URL contains secret-bearing query key(s): {names}",
        )
    return None


def _slug(text: str) -> str:
    value = re.sub(r"<[^>]+>", "", text).strip().lower()
    value = re.sub(r"[^\w\- ]", "", value, flags=re.UNICODE)
    return re.sub(r"\s", "-", value).strip("-")


def _markdown_anchors(path: Path) -> frozenset[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for line in _prose(path.read_text(encoding="utf-8")).splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not match:
            continue
        base = _slug(match.group(1))
        count = counts.get(base, 0)
        counts[base] = count + 1
        anchors.add(base if count == 0 else f"{base}-{count}")
    return frozenset(anchors)


def _resolve_source_target(ref: LinkRef, root: Path) -> tuple[Path, str]:
    parts = urlsplit(ref.target)
    fragment = unquote(parts.fragment)
    raw_path = unquote(parts.path)
    source_path = root / ref.source
    if not raw_path:
        return source_path, fragment
    if ref.kind == "mkdocs-nav":
        return root / "docs" / raw_path, fragment
    candidate = source_path.parent / raw_path
    if ref.kind == "tex-file" and not candidate.suffix:
        candidates = [candidate.with_suffix(ext) for ext in (".tex", ".bib", ".pdf", ".png", ".jpg", ".svg")]
        existing = next((path for path in candidates if path.exists()), candidate)
        return existing, fragment
    return candidate, fragment


def audit_local(root: Path, policy: Policy) -> tuple[tuple[Finding, ...], tuple[LinkRef, ...]]:
    """Validate tracked local targets, anchors, and non-secret external URLs."""
    tracked = {path.resolve() for path in _tracked_paths(root)}
    sources = public_sources(root, policy)
    refs = tuple(ref for path in sources for ref in extract_links(path, root))
    findings: set[Finding] = set()
    anchor_cache: dict[Path, frozenset[str]] = {}
    nav_targets = {_resolve_source_target(ref, root)[0].resolve() for ref in refs if ref.kind == "mkdocs-nav"}
    for source in sources:
        relative = source.relative_to(root).as_posix()
        if relative.startswith("docs/") and source.suffix.lower() in {".md", ".markdown"}:
            if source.resolve() not in nav_targets:
                findings.add(
                    Finding(relative, 1, relative.removeprefix("docs/"), "public page is absent from MkDocs nav")
                )
    for ref in refs:
        parts = urlsplit(ref.target)
        scheme = parts.scheme.lower()
        if scheme in policy.ignored_schemes:
            continue
        if scheme in policy.allowed_schemes:
            finding = _url_policy_finding(ref, policy)
            if finding:
                findings.add(finding)
            continue
        if scheme or parts.netloc or ref.target.startswith("/"):
            continue
        target, fragment = _resolve_source_target(ref, root)
        resolved = target.resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            findings.add(Finding(ref.source, ref.line, ref.target, "relative target escapes repository root"))
            continue
        if not resolved.exists():
            findings.add(Finding(ref.source, ref.line, ref.target, "relative target does not exist"))
            continue
        if policy.require_tracked_targets and resolved.is_file() and resolved not in tracked:
            findings.add(Finding(ref.source, ref.line, ref.target, "relative target is not tracked"))
            continue
        if fragment and policy.check_markdown_anchors and resolved.suffix.lower() in {".md", ".markdown"}:
            anchors = anchor_cache.setdefault(resolved, _markdown_anchors(resolved))
            if fragment not in anchors:
                findings.add(Finding(ref.source, ref.line, ref.target, "Markdown anchor does not exist"))
    return tuple(sorted(findings)), refs


def audit_site(site_dir: Path, base_path: str = "") -> tuple[Finding, ...]:
    """Validate rendered HTML links inside a generated MkDocs site tree."""
    findings: set[Finding] = set()
    site_root = site_dir.resolve()
    for source in sorted(site_dir.rglob("*.html")):
        text = source.read_text(encoding="utf-8", errors="strict")
        for match in _HTML_LINK_RE.finditer(text):
            target = _clean_target(match.group(2))
            parts = urlsplit(target)
            if parts.scheme or parts.netloc or target.startswith(("#", "mailto:", "tel:")):
                continue
            raw_path = unquote(parts.path)
            if not raw_path:
                continue
            if target.startswith("/"):
                normalised_base = "/" + base_path.strip("/") if base_path.strip("/") else ""
                if normalised_base and not (raw_path == normalised_base or raw_path.startswith(f"{normalised_base}/")):
                    continue
                public_path = raw_path[len(normalised_base) :].lstrip("/")
                candidate = site_root / public_path
            else:
                candidate = source.parent / raw_path
            if raw_path.endswith("/") or candidate.is_dir() or not candidate.suffix and not candidate.exists():
                candidate = candidate / "index.html"
            try:
                candidate.resolve().relative_to(site_root)
            except ValueError:
                reason = "rendered target escapes site root"
            else:
                reason = "rendered target does not exist" if not candidate.exists() else ""
            if reason:
                findings.add(
                    Finding(
                        source.relative_to(site_root).as_posix(),
                        _line(text, match.start(2)),
                        target,
                        reason,
                    )
                )
    return tuple(sorted(findings))


def _external_urls(refs: Iterable[LinkRef], policy: Policy) -> tuple[str, ...]:
    urls: set[str] = set()
    for ref in refs:
        parts = urlsplit(ref.target)
        if parts.scheme.lower() not in policy.allowed_schemes or _url_policy_finding(ref, policy) is not None:
            continue
        urls.add(urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, "")))
    return tuple(sorted(urls))


def _classify_http(code: int, policy: Policy) -> str:
    if 200 <= code < 400:
        return "reachable"
    if code in policy.restricted_statuses:
        return "restricted"
    if code in policy.transient_statuses:
        return "transient"
    if code in policy.permanent_statuses or 400 <= code < 500:
        return "permanent-failure"
    return "transient"


def _request_once(url: str, policy: Policy, method: str) -> tuple[int, str]:
    headers = {"User-Agent": policy.user_agent, "Accept": "text/html,application/json;q=0.9,*/*;q=0.1"}
    if method == "GET":
        headers["Range"] = "bytes=0-0"
    request = Request(url, headers=headers, method=method)
    try:
        with urlopen(request, timeout=policy.timeout_seconds) as response:  # nosec B310
            final_url = response.geturl()
            if _url_policy_finding(LinkRef("<redirect>", 0, final_url, "external"), policy):
                raise UnsafeExternalTarget("redirect target violates public URL policy")
            return int(response.status), final_url
    except HTTPError as exc:
        final_url = exc.geturl()
        if _url_policy_finding(LinkRef("<redirect>", 0, final_url, "external"), policy):
            raise UnsafeExternalTarget("redirect target violates public URL policy") from None
        return int(exc.code), final_url


def _check_external(url: str, policy: Policy) -> ExternalResult:
    checked_at = datetime.now(timezone.utc).isoformat()
    attempts = 0
    last_detail = ""
    for attempt in range(policy.retries + 1):
        attempts += 1
        try:
            status, final_url = _request_once(url, policy, "HEAD")
            if status in {403, 405, 501}:
                status, final_url = _request_once(url, policy, "GET")
            classification = _classify_http(status, policy)
            if classification != "transient" or attempt == policy.retries:
                return ExternalResult(url, classification, status, attempts, checked_at, final_url, "HTTP response")
            last_detail = f"transient HTTP {status}"
        except (TimeoutError, URLError, OSError) as exc:
            last_detail = f"{type(exc).__name__}: {exc}"
            if attempt == policy.retries:
                return ExternalResult(url, "transient", None, attempts, checked_at, None, last_detail)
        except UnsafeExternalTarget:
            return ExternalResult(
                url,
                "policy-failure",
                None,
                attempts,
                checked_at,
                None,
                "redirect target violates public URL policy",
            )
        time.sleep(policy.retry_backoff_seconds * (attempt + 1))
    return ExternalResult(url, "transient", None, attempts, checked_at, None, last_detail)


def _tool_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _load_cache(path: Path, policy: Policy) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if payload.get("schema_version") != SCHEMA:
        return {}
    provenance = payload.get("provenance", {})
    if provenance.get("policy_sha256") != policy.source_sha256 or provenance.get("tool_sha256") != _tool_sha256():
        return {}
    return {str(item["url"]): item for item in payload.get("results", []) if isinstance(item, dict) and "url" in item}


def _cached_result(item: dict[str, Any], now: datetime, policy: Policy) -> ExternalResult | None:
    try:
        checked = datetime.fromisoformat(str(item["checked_at"]))
        classification = str(item["classification"])
        ttl = policy.transient_cache_ttl_seconds if classification == "transient" else policy.cache_ttl_seconds
        if (now - checked).total_seconds() > ttl:
            return None
        return ExternalResult(
            url=str(item["url"]),
            classification=classification,
            status_code=int(item["status_code"]) if item.get("status_code") is not None else None,
            attempts=int(item["attempts"]),
            checked_at=str(item["checked_at"]),
            final_url=str(item["final_url"]) if item.get("final_url") else None,
            detail=str(item.get("detail", "cached result")),
            cached=True,
        )
    except (KeyError, TypeError, ValueError):
        return None


def audit_external(urls: Sequence[str], policy: Policy, cache_path: Path) -> tuple[ExternalResult, ...]:
    """Check bounded public URLs with per-host delay and TTL cache reuse."""
    if len(urls) > policy.max_urls:
        raise ValueError(f"external URL count {len(urls)} exceeds policy maximum {policy.max_urls}")
    cache = _load_cache(cache_path, policy)
    now = datetime.now(timezone.utc)
    last_host_at: dict[str, float] = {}
    results: list[ExternalResult] = []
    for url in urls:
        cached = _cached_result(cache[url], now, policy) if url in cache else None
        if cached:
            results.append(cached)
            continue
        host = urlsplit(url).netloc.lower()
        elapsed = time.monotonic() - last_host_at.get(host, 0.0)
        if elapsed < policy.per_host_delay_seconds:
            time.sleep(policy.per_host_delay_seconds - elapsed)
        results.append(_check_external(url, policy))
        last_host_at[host] = time.monotonic()
    return tuple(results)


def _digest_lines(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_report(
    path: Path,
    root: Path,
    policy_path: Path,
    sources: Sequence[Path],
    refs: Sequence[LinkRef],
    findings: Sequence[Finding],
    external_results: Sequence[ExternalResult],
) -> None:
    payload = {
        "schema_version": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            "tool_sha256": _tool_sha256(),
            "source_set_sha256": _digest_lines(path.relative_to(root).as_posix() for path in sources),
            "external_url_set_sha256": _digest_lines(result.url for result in external_results),
        },
        "source_count": len(sources),
        "reference_count": len(refs),
        "findings": [asdict(item) for item in findings],
        "results": [asdict(item) for item in external_results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--site-dir", type=Path)
    parser.add_argument("--external", action="store_true")
    parser.add_argument("--cache", type=Path, default=Path("artifacts/document_link_audit.json"))
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--list-external", action="store_true")
    return parser


def _mkdocs_base_path(root: Path) -> str:
    mkdocs = root / "mkdocs.yml"
    if not mkdocs.exists():
        return ""
    match = re.search(r"(?m)^site_url:\s*(\S+)\s*$", mkdocs.read_text(encoding="utf-8"))
    return urlsplit(match.group(1)).path if match else ""


def main(argv: Sequence[str] | None = None) -> int:
    """Run deterministic local checks and optional bounded external crawling."""
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    policy_path = args.policy.resolve()
    policy = _read_policy(policy_path)
    sources = public_sources(root, policy)
    findings, refs = audit_local(root, policy)
    site_findings = audit_site(args.site_dir.resolve(), _mkdocs_base_path(root)) if args.site_dir else ()
    findings = tuple(sorted((*findings, *site_findings)))
    urls = _external_urls(refs, policy)
    if args.list_external:
        for url in urls:
            print(url)
    external_results: tuple[ExternalResult, ...] = ()
    if args.external:
        external_results = audit_external(urls, policy, args.cache.resolve())
    written_reports: set[Path] = set()
    if args.external:
        cache_path = args.cache.resolve()
        _write_report(cache_path, root, policy_path, sources, refs, findings, external_results)
        written_reports.add(cache_path)
    if args.json_out:
        report_path = args.json_out.resolve()
        if report_path not in written_reports:
            _write_report(report_path, root, policy_path, sources, refs, findings, external_results)
    for finding in findings:
        print(f"{finding.source}:{finding.line}: {finding.reason}: {finding.target}")
    for result in external_results:
        marker = "cached" if result.cached else f"attempts={result.attempts}"
        print(f"{result.classification}: {result.url} ({marker}, status={result.status_code})")
    permanent = [result for result in external_results if result.classification == "permanent-failure"]
    policy_failures = [result for result in external_results if result.classification == "policy-failure"]
    if findings or permanent or policy_failures:
        print(
            f"Document link audit FAILED: {len(findings)} local/policy finding(s), "
            f"{len(permanent)} permanent URL failure(s), {len(policy_failures)} redirect-policy failure(s)."
        )
        return 1
    print(f"Document link audit passed: {len(sources)} sources, {len(refs)} references, {len(urls)} external URLs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
