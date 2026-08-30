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
