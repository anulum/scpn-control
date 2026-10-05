# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption Contracts.
"""Stable public imports for bounded disruption mitigation contracts.

The physics proxies, synthetic episode and labelled-shot replay live in focused
modules; this facade preserves their established public import paths.
"""

from __future__ import annotations

from scpn_control.control._disruption_episode_physics import (
    impurity_transport_response as impurity_transport_response,
)
from scpn_control.control._disruption_episode_physics import (
    mcnp_lite_tbr as mcnp_lite_tbr,
)
from scpn_control.control._disruption_episode_physics import (
    post_disruption_halo_runaway as post_disruption_halo_runaway,
)
from scpn_control.control._disruption_episode_physics import (
    synthetic_disruption_signal as synthetic_disruption_signal,
)
from scpn_control.control._disruption_episode_runtime import run_disruption_episode as run_disruption_episode
from scpn_control.control._disruption_shot_replay import run_real_shot_replay as run_real_shot_replay
from scpn_control.control.disruption_predictor import predict_disruption_risk as predict_disruption_risk
from scpn_control.core._validators import require_1d_array as require_1d_array
from scpn_control.core._validators import require_finite_float as require_finite_float
from scpn_control.core._validators import require_fraction as require_fraction
from scpn_control.core._validators import require_int as require_int
from scpn_control.core._validators import require_positive_float as require_positive_float

__all__ = [
    "impurity_transport_response",
    "mcnp_lite_tbr",
    "post_disruption_halo_runaway",
    "require_fraction",
    "require_int",
    "run_disruption_episode",
    "run_real_shot_replay",
    "synthetic_disruption_signal",
]
