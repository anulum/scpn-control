# Reactor-semantic fixture provenance

`torax_runtime_review_envelope_v1.json` is a byte-for-byte test copy of the
canonical SCPN-FUSION-CORE review envelope committed in evidence commit
`0f5c0108a4ac111f7c01c2d862ac50551c1eb44f`. Its producer commit is
`314463489c95692d851cf6b9102ca733d878ca8a`; its SHA-256 is
`b594e2f8b72056426d628b638f6a849ef39e75daddc827305002b109365596c4`.

The fixture is portable test evidence, not copied sibling code. Tests verify
the digest before passing its exact bytes through the installed public SPO
adapter and CONTROL admission API. Updating it requires a new immutable FUSION
producer/evidence receipt, a matching SPO receipt, and an explicit digest
change in the dedicated tests.

`mif_merge_compression_observation_v1.json` is a byte-for-byte copy of the
canonical SCPN-MIF-CORE merge-compression observation committed at producer
revision `f60dbae4b2ea3344ac0cb086a3b7d248d65cf92f`. Its SHA-256 is
`c780706abd5a0b185a95e85767e623248388664da61126d196fcb3d528b0c0ca`.
The installed SPO `1.3.0` public adapter deterministically converts these 2,475
bytes into the 101,652-byte review-only handoff with SHA-256
`c0f03b7c49346c39342598275556e8ac28c93138ba14f6e21d6739400e0edeb2`.

The MIF fixture describes serialized simulation state, not measured plasma
phase, physical compression actuation, fusion yield, or plant readiness. Only
the two serialized model angles become `numerical_phase`; all other atoms
remain bounded or categorical nonphase evidence. Updating it requires a new
MIF producer receipt, matching SPO adapter receipt, and explicit CONTROL policy
digest changes.
