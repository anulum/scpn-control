# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Python release metadata transport contracts.
"""Verify real HTTP metadata identity, refusal and bounded resource behaviour."""

from __future__ import annotations

import pytest
from dependency_metadata_server import metadata_service, release_metadata

from tools.python_dependency_metadata import MAX_METADATA_BYTES, DependencyMetadataError, read_python_constraint


@pytest.mark.parametrize("constraint", [">=3.11,!=3.12.0", "", None])
def test_metadata_binds_normalised_release_through_actual_http(constraint: str | None) -> None:
    """Retain explicit null/empty constraints and match normalised identities."""
    with metadata_service(release_metadata("Example_Distribution", "1.0.0", constraint)) as (url, requests):
        assert read_python_constraint("example.distribution", "1.0", base_url=url) == constraint
    assert requests == ["/pypi/example-distribution/1.0/json"]


@pytest.mark.parametrize(
    "body,diagnostic",
    [
        (b"", "empty"),
        (b" " * (MAX_METADATA_BYTES + 1), "exceeds"),
        (b"\xff", "UTF-8"),
        (b"{", "JSON"),
        (b"[" * 20000 + b"0" + b"]" * 20000, "JSON"),
        (b'{"integer":' + b"9" * 5000 + b"}", "JSON"),
        (b"[]", "info object"),
        (b'{"info": []}', "info object"),
        (b'{"info": {}}', "name and version"),
        (b'{"info": {"name": "?", "version": "1.0"}}', "invalid identity"),
        (b'{"info": {"name": "example-distribution", "version": "bad"}}', "invalid identity"),
        (release_metadata("other"), "does not match"),
        (release_metadata(version="2"), "does not match"),
        (b'{"info": {"name": "example-distribution", "version": "1.0"}}', "Python constraint"),
        (b'{"info": {"name": "example-distribution", "version": "1.0", "requires_python": 3}}', "Python constraint"),
        (b'{"info": {}, "info": {}}', "duplicate"),
        (b'{"info": {"name": "x", "name": "y"}}', "duplicate"),
        (b'{"info": NaN}', "non-JSON"),
    ],
    ids=[
        "empty",
        "oversized",
        "invalid-utf8",
        "invalid-json",
        "excessive-json-nesting",
        "decoder-integer-limit",
        "array",
        "info-array",
        "absent-identity",
        "invalid-name",
        "invalid-version",
        "different-name",
        "different-version",
        "absent-constraint",
        "nonstring-constraint",
        "duplicate-info",
        "duplicate-name",
        "non-json-constant",
    ],
)
def test_metadata_refuses_actual_invalid_response_bodies(body: bytes, diagnostic: str) -> None:
    """No malformed, oversized or differently identified response yields a constraint."""
    with metadata_service(body) as (url, _requests), pytest.raises(DependencyMetadataError, match=diagnostic):
        read_python_constraint("example-distribution", "1.0", base_url=url)


@pytest.mark.parametrize("status", [301, 404, 503])
def test_metadata_refuses_http_status_without_redirect_or_retry(status: int) -> None:
    """An unavailable release produces one request and a fixed refusal."""
    with (
        metadata_service(release_metadata(), status=status) as (url, requests),
        pytest.raises(DependencyMetadataError, match="HTTP 200"),
    ):
        read_python_constraint("example-distribution", "1.0", base_url=url)
    assert len(requests) == 1


def test_metadata_refuses_wrong_actual_http_media_type() -> None:
    """A syntactically valid JSON body cannot override the response media contract."""
    with (
        metadata_service(release_metadata(), media_type="text/html") as (url, _requests),
        pytest.raises(DependencyMetadataError, match="application/json"),
    ):
        read_python_constraint("example-distribution", "1.0", base_url=url)


def test_metadata_translates_an_actual_socket_timeout() -> None:
    """A server withholding headers cannot become a passing metadata observation."""
    with (
        metadata_service(release_metadata(), delay_s=0.2) as (url, requests),
        pytest.raises(DependencyMetadataError, match="Could not read"),
    ):
        read_python_constraint("example-distribution", "1.0", base_url=url, timeout_s=0.05)
    assert len(requests) == 1


def test_metadata_translates_an_actual_tls_handshake_failure() -> None:
    """TLS against a plain HTTP listener refuses without exposing SSL exception text."""
    with (
        metadata_service(release_metadata()) as (url, _requests),
        pytest.raises(DependencyMetadataError, match="^Could not read the requested release metadata.$"),
    ):
        read_python_constraint("example-distribution", "1.0", base_url=url.replace("http:", "https:"), timeout_s=1)


def test_metadata_refuses_an_unencoded_origin_path_without_a_traceback() -> None:
    """The HTTP writer's real Unicode failure becomes an authored refusal."""
    with (
        metadata_service(release_metadata()) as (url, requests),
        pytest.raises(DependencyMetadataError, match="^Could not read the requested release metadata.$"),
    ):
        read_python_constraint("example-distribution", "1.0", base_url=url + "/\u017e")
    assert requests == []


@pytest.mark.parametrize(
    "url",
    [
        "https://[",
        "https://localhost:99999",
        "https:///pypi",
        "file:///pypi",
        "http://example.org/pypi",
        "https://user@localhost/pypi",
        "https://localhost/pypi?query=1",
        "https://localhost/pypi#fragment",
    ],
)
def test_metadata_refuses_unusable_origins_before_a_request(url: str) -> None:
    """Unsupported or ambiguous origins cannot reach release interpretation."""
    with pytest.raises(DependencyMetadataError, match="origin"):
        read_python_constraint("example-distribution", "1.0", base_url=url)


@pytest.mark.parametrize("name,version", [("", "1.0"), ("?", "1.0"), ("example", "bad")])
def test_metadata_refuses_invalid_requested_release_identity(name: str, version: str) -> None:
    """Invalid requested names and versions refuse before opening a connection."""
    with pytest.raises(DependencyMetadataError):
        read_python_constraint(name, version)


@pytest.mark.parametrize("timeout", [True, 0.0, -1.0, float("nan"), float("inf")])
def test_metadata_refuses_nonpositive_or_nonfinite_timeout(timeout: float) -> None:
    """No invalid timeout can create an unbounded metadata connection."""
    with pytest.raises(DependencyMetadataError, match="timeout"):
        read_python_constraint("example-distribution", "1.0", timeout_s=timeout)
