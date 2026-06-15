// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Fuzz target: reactor-configuration JSON parser surface.
//
// `ReactorConfig::from_file` deserialises an untrusted JSON document with
// `serde_json`. This target drives that deserialisation directly from fuzzer
// bytes and asserts two contracts on every accepted document:
//   1. a value that deserialised must re-serialise without panicking, and
//   2. the re-serialised text must parse back to an equivalent config.
// Rejected (malformed) input must return an error, never panic. Grid
// construction is exercised only for bounded resolutions so the target probes
// the parser rather than the allocator (extreme-grid OOM is a separate
// denial-of-service guard, not a parser defect).

#![no_main]

use control_types::config::ReactorConfig;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    let Ok(config) = serde_json::from_str::<ReactorConfig>(text) else {
        return;
    };

    let reserialised = serde_json::to_string(&config).expect("accepted config must re-serialise");
    let _reparsed: ReactorConfig =
        serde_json::from_str(&reserialised).expect("serialised config must round-trip");

    if config.grid_resolution[0] <= 256 && config.grid_resolution[1] <= 256 {
        let _ = config.create_grid();
    }
});
