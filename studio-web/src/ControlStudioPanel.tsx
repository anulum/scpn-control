// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Control Studio UI remote — the exposed ControlStudioPanel

import type { ReactElement } from 'react';

import {
  claimRendersAsValidated,
  CONTROL_CLAIMS,
  CONTROL_VERBS,
  requiresLiveHardwareGate,
} from './domain.js';

/**
 * The SCPN-CONTROL Control Studio panel — the federated UI module the Hub loads.
 *
 * It surfaces CONTROL's verbs (each with its safety tier, side-effect class, and
 * timing — a realtime verb shows its hard deadline, a live-hardware verb is marked
 * as Hub-gated) and a claims section that renders each claim's boundary verbatim,
 * marking only a reference-validated claim as validated. The same honesty grading
 * the Python vertical emits is shown here as UI.
 */
export default function ControlStudioPanel(): ReactElement {
  return (
    <section className="control-studio">
      <header className="control-studio__header">
        <h2>SCPN-CONTROL — Control Studio</h2>
      </header>

      <table className="control-studio__verbs">
        <thead>
          <tr>
            <th>Verb</th>
            <th>Safety tier</th>
            <th>Side effect</th>
            <th>Timing</th>
            <th>Hub gate</th>
          </tr>
        </thead>
        <tbody>
          {CONTROL_VERBS.map((verb) => {
            const gated = requiresLiveHardwareGate(verb);
            const timing =
              verb.deadlineUs === undefined
                ? verb.timingClass
                : `${verb.timingClass} (${verb.deadlineUs.toString()} µs)`;
            return (
              <tr key={verb.name} data-distinctive={verb.domainDistinctive ? 'domain' : 'core'}>
                <td>{verb.name}</td>
                <td>{verb.safetyTier}</td>
                <td>{verb.sideEffect}</td>
                <td>{timing}</td>
                <td>{gated ? 'live-hardware (per-tenant)' : '—'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <div className="control-studio__claims">
        <h3>Claims</h3>
        <ul>
          {CONTROL_CLAIMS.map((claim) => {
            const validated = claimRendersAsValidated(claim.status, claim.admission);
            return (
              <li key={claim.schema} data-validated={validated ? 'yes' : 'no'}>
                {claim.schema} — {claim.kind} — {validated ? 'validated' : claim.status}
              </li>
            );
          })}
        </ul>
      </div>
    </section>
  );
}
