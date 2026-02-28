# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Artifact Compact Codec Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Edge case tests for the u64 compact codec and artifact validation."""
from __future__ import annotations

import pytest

from scpn_control.scpn.artifact import (
    ArtifactValidationError,
    _decode_u64_compact,
    decode_u64_compact,
    encode_u64_compact,
)


class TestCompactCodecRoundTrip:
    def test_empty_list(self):
        encoded = encode_u64_compact([])
        decoded = decode_u64_compact(encoded)
        assert decoded == []

    def test_single_value(self):
        encoded = encode_u64_compact([42])
        decoded = decode_u64_compact(encoded)
        assert decoded == [42]

    def test_large_values(self):
        data = [0, 1, 2**63 - 1, 2**64 - 1]
        encoded = encode_u64_compact(data)
        decoded = decode_u64_compact(encoded)
        assert decoded == data

    def test_many_values(self):
        data = list(range(500))
        encoded = encode_u64_compact(data)
        decoded = decode_u64_compact(encoded)
        assert decoded == data

    def test_encoding_format(self):
        encoded = encode_u64_compact([1, 2, 3])
        assert encoded["encoding"] == "u64-le-zlib-base64"
        assert encoded["count"] == 3
        assert "data_u64_b64_zlib" in encoded


class TestDecodeErrors:
    def test_wrong_encoding_tag(self):
        with pytest.raises(ArtifactValidationError, match="Unsupported packed encoding"):
            _decode_u64_compact({"encoding": "raw", "data_u64_b64_zlib": "AA=="})

    def test_missing_payload(self):
        with pytest.raises(ArtifactValidationError, match="Missing compact packed payload"):
            _decode_u64_compact({"encoding": "u64-le-zlib-base64", "data_u64_b64_zlib": 123})

    def test_invalid_base64(self):
        with pytest.raises(ArtifactValidationError, match="Invalid base64"):
            _decode_u64_compact({
                "encoding": "u64-le-zlib-base64",
                "data_u64_b64_zlib": "!!!not_base64!!!",
            })

    def test_invalid_zlib_payload(self):
        import base64
        bad_bytes = base64.b64encode(b"not_valid_zlib").decode("ascii")
        with pytest.raises(ArtifactValidationError, match="Invalid compact"):
            _decode_u64_compact({
                "encoding": "u64-le-zlib-base64",
                "data_u64_b64_zlib": bad_bytes,
            })

    def test_byte_length_not_multiple_of_8(self):
        import base64, zlib
        raw = b"\x00" * 7  # 7 bytes, not divisible by 8
        compressed = zlib.compress(raw)
        payload = base64.b64encode(compressed).decode("ascii")
        with pytest.raises(ArtifactValidationError, match="not divisible by 8"):
            _decode_u64_compact({
                "encoding": "u64-le-zlib-base64",
                "data_u64_b64_zlib": payload,
                "count": None,
            })

    def test_negative_count(self):
        encoded = encode_u64_compact([1, 2])
        encoded["count"] = -1
        with pytest.raises(ArtifactValidationError, match="exceeds limit"):
            _decode_u64_compact(encoded)

    def test_count_exceeds_available(self):
        encoded = encode_u64_compact([1])
        encoded["count"] = 99
        with pytest.raises(ArtifactValidationError, match="Invalid compact packed count"):
            _decode_u64_compact(encoded)

    def test_invalid_count_type(self):
        import base64, zlib
        raw = b"\x00" * 8
        compressed = zlib.compress(raw)
        payload = base64.b64encode(compressed).decode("ascii")
        with pytest.raises(ArtifactValidationError, match="Invalid compact packed count type"):
            _decode_u64_compact({
                "encoding": "u64-le-zlib-base64",
                "data_u64_b64_zlib": payload,
                "count": "bad",
            })

    def test_none_count_uses_available(self):
        encoded = encode_u64_compact([10, 20, 30])
        del encoded["count"]  # force None path
        decoded = _decode_u64_compact(encoded)
        assert decoded == [10, 20, 30]
