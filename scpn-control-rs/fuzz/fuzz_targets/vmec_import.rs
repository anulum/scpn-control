// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Fuzz target: VMEC-like fixed-boundary text-import parser.
//
// `import_vmec_like_text` parses an untrusted free-text stellarator boundary
// description (keyword lines plus Fourier-mode rows). This target asserts that
// any accepted boundary state both validates and survives an export/import
// round-trip without panicking. Malformed text must return an error.

#![no_main]

use control_core::vmec_interface::{export_vmec_like_text, import_vmec_like_text};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    if let Ok(state) = import_vmec_like_text(text) {
        // A parser that admits a boundary must produce one its own validator and
        // exporter accept; a mismatch is a parser/exporter inconsistency.
        let _ = state.validate();
        if let Ok(exported) = export_vmec_like_text(&state) {
            let _ = import_vmec_like_text(&exported);
        }
    }
});
