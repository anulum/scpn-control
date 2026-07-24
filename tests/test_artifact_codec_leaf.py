# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for artifact compact codec leaf

"""Drive production compact packed-weight codec leaf and owner re-exports."""

from __future__ import annotations

import pytest

import scpn_control.scpn.artifact as owner
import scpn_control.scpn.artifact_codec as codec
from scpn_control.scpn.artifact_validate import ArtifactValidationError


def test_owner_reexports_codec_surface_by_identity() -> None:
    """Owner product surface re-exports the codec leaf by identity."""
    assert owner.encode_u64_compact is codec.encode_u64_compact
    assert owner.decode_u64_compact is codec.decode_u64_compact
    assert owner._encode_u64_compact is codec._encode_u64_compact
    assert owner._decode_u64_compact is codec._decode_u64_compact


def test_roundtrip_empty_and_nonempty_words() -> None:
    """Public codec round-trips empty and non-empty uint64 word lists."""
    for words in ([], [0], [1, 2, 3], [2**64 - 1, 42]):
        encoded = codec.encode_u64_compact(words)
        assert encoded["encoding"] == "u64-le-zlib-base64"
        assert encoded["count"] == len(words)
        assert isinstance(encoded["data_u64_b64_zlib"], str)
        assert codec.decode_u64_compact(encoded) == words


def test_decode_rejects_unsupported_encoding() -> None:
    """Unknown encoding tags fail closed with ArtifactValidationError."""
    with pytest.raises(ArtifactValidationError, match="Unsupported packed encoding"):
        codec.decode_u64_compact({"encoding": "raw-hex", "count": 0, "data_u64_b64_zlib": ""})


def test_decode_rejects_missing_payload_string() -> None:
    """Missing compact payload string is rejected."""
    with pytest.raises(ArtifactValidationError, match="Missing compact packed payload"):
        codec.decode_u64_compact({"encoding": "u64-le-zlib-base64", "count": 0})
