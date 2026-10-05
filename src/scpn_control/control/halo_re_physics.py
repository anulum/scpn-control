# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Halo Runaway Physics
r"""Physics-based halo current and runaway electron models.

Halo Current Model (Fitzpatrick-style L/R circuit)
---------------------------------------------------
During a vertical displacement event (VDE) or disruption, the plasma column
contacts the wall and drives a halo current through the vessel structure.
The model uses an L/R circuit analogy:

    L_h dI_h/dt + R_h I_h = M dI_p/dt

where I_h is the halo current, I_p is the plasma current, M is the mutual
inductance between plasma and halo region, and L_h, R_h are the halo circuit
inductance and resistance.

The toroidal peaking factor (TPF) captures non-uniform wall-contact geometry:

    TPF × I_h / I_p  ≤  0.75   (ITER design limit)

Reference: Fitzpatrick, R., "Halo Current and Error Field Interaction",
           Phys. Plasmas 9, 3459 (2002).

Runaway Electron Model (Connor-Hastie + Rosenbluth-Putvinski)
--------------------------------------------------------------
Primary (Dreicer) generation:

    γ_D = (n_e / τ_coll) · C_D · (E_D / E)^{h(Z_eff)} · exp(-E_D/(4E) - √(ν_eff))

where E_D is the Dreicer field, E is the toroidal electric field, and h(Z) is
a Z_eff-dependent exponent.

Secondary (avalanche) generation:

    γ_av = n_RE · (E/E_c - 1) / (τ_av · ln Λ)

with E_c the critical (Connor-Hastie) field for runaway sustainment.

References
----------
    Connor, J.W. & Hastie, R.J., Nucl. Fusion 15, 415 (1975).
    Rosenbluth, M.N. & Putvinski, S.V., Nucl. Fusion 37, 1355 (1997).
"""

from __future__ import annotations

import numpy as np

from scpn_control.control._disruption_claims import (
    DisruptionMitigationClaimEvidence,
    assert_disruption_mitigation_claim_admissible,
    disruption_mitigation_claim_evidence,
    save_disruption_mitigation_claim_evidence,
)
from scpn_control.control._disruption_claims import (
    _finite_nonnegative_or_none as _finite_nonnegative_or_none,
)
from scpn_control.control._disruption_claims import (
    _finite_unit_interval as _finite_unit_interval,
)
from scpn_control.control._disruption_claims import (
    _non_empty_text as _non_empty_text,
)
from scpn_control.control._disruption_ensemble import DisruptionMitigationReport, run_disruption_ensemble
from scpn_control.control._halo_current_model import _MU0 as _MU0
from scpn_control.control._halo_current_model import HaloCurrentModel, HaloCurrentResult
from scpn_control.control._runaway_electron_model import (
    _C_LIGHT as _C_LIGHT,
)
from scpn_control.control._runaway_electron_model import (
    _E_CHARGE as _E_CHARGE,
)
from scpn_control.control._runaway_electron_model import (
    _EPSILON0 as _EPSILON0,
)
from scpn_control.control._runaway_electron_model import (
    _LN_LAMBDA as _LN_LAMBDA,
)
from scpn_control.control._runaway_electron_model import (
    _M_ELECTRON as _M_ELECTRON,
)
from scpn_control.control._runaway_electron_model import RunawayElectronModel, RunawayElectronResult

__all__ = [
    "DisruptionMitigationClaimEvidence",
    "DisruptionMitigationReport",
    "HaloCurrentModel",
    "HaloCurrentResult",
    "RunawayElectronModel",
    "RunawayElectronResult",
    "assert_disruption_mitigation_claim_admissible",
    "disruption_mitigation_claim_evidence",
    "run_disruption_ensemble",
    "save_disruption_mitigation_claim_evidence",
    "np",
]
