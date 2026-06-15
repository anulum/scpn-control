// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Fuzz target: BOUT++ linear-stability output text parser.
//
// `parse_bout_stability` consumes the untrusted stdout of an external BOUT++
// linear-stability run (`n = ...`, `gamma = ...`, `omega = ...`,
// `amplitude = a, b, c`). The parser does its own float/int parsing and field
// validation; this target only requires that no input — truncated, mojibake,
// or adversarial numeric tokens — makes it panic. Errors are the expected
// outcome for malformed text.

#![no_main]

use control_core::bout_interface::parse_bout_stability;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    let _ = parse_bout_stability(text);
});
