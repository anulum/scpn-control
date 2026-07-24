# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for artifact JSON schema leaf

"""Drive production artifact JSON schema leaf and owner re-exports."""

from __future__ import annotations

import scpn_control.scpn.artifact as owner
import scpn_control.scpn.artifact_schema as schema


def test_owner_reexports_schema_surface_by_identity() -> None:
    """Owner product surface re-exports the schema leaf by identity."""
    assert owner.get_artifact_json_schema is schema.get_artifact_json_schema
    assert owner._object_schema is schema._object_schema
    assert owner._packed_weight_schema is schema._packed_weight_schema


def test_get_artifact_json_schema_is_closed_draft07() -> None:
    """Emitted schema is Draft-07, closed, and names the controller artifact."""
    doc = schema.get_artifact_json_schema()
    assert doc["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert doc["title"] == "SCPN Controller Artifact"
    assert doc["type"] == "object"
    assert doc["additionalProperties"] is False
    assert "meta" in doc["properties"]
    assert "topology" in doc["properties"]
    assert "weights" in doc["properties"]
    assert "formal_verification" in doc["definitions"]


def test_object_schema_helper_is_closed() -> None:
    """Private object helper rejects additional properties by default."""
    obj = schema._object_schema({"a": {"type": "string"}}, required=("a",))
    assert obj["additionalProperties"] is False
    assert obj["required"] == ["a"]
