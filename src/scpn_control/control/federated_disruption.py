# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Federated Disruption.

"""Federated learning framework for cross-machine disruption prediction.

Trains a shared MLP disruption classifier across heterogeneous tokamak
datasets (DIII-D, JET, KSTAR, EAST, SPARC). The server receives model updates;
callers decide where client arrays are created and retained.
Supports FedAvg (McMahan et al., AISTATS 2017) and FedProx (Li et al.,
MLSys 2020) aggregation strategies.

Disruption features: Ip, beta_N, q95, n/n_GW, li, dBp/dt,
locked_mode_amplitude, n1_rms — 8-dimensional input space whose
distributions differ across machines (JET: higher Ip; DIII-D: more
shaping; KSTAR: longer pulses).
"""

from __future__ import annotations

import logging

from scpn_control.control._federated_benchmark import FacilityBenchmarkSummary, run_synthetic_multifacility_benchmark
from scpn_control.control._federated_clients import (
    MACHINE_PROFILES,
    MachineClient,
    create_facility_clients_from_arrays,
    create_machine_clients,
)
from scpn_control.control._federated_clients import _generate_disruption_data as _generate_disruption_data
from scpn_control.control._federated_config import FederatedConfig
from scpn_control.control._federated_model import FEATURE_NAMES as FEATURE_NAMES
from scpn_control.control._federated_model import N_FEATURES
from scpn_control.control._federated_model import (
    _apply_weight_delta as _apply_weight_delta,
)
from scpn_control.control._federated_model import (
    _binary_cross_entropy as _binary_cross_entropy,
)
from scpn_control.control._federated_model import (
    _init_mlp_weights as _init_mlp_weights,
)
from scpn_control.control._federated_model import (
    _l2_norm as _l2_norm,
)
from scpn_control.control._federated_model import (
    _mlp_forward as _mlp_forward,
)
from scpn_control.control._federated_model import (
    _mlp_gradients as _mlp_gradients,
)
from scpn_control.control._federated_model import (
    _relu as _relu,
)
from scpn_control.control._federated_model import (
    _sigmoid as _sigmoid,
)
from scpn_control.control._federated_model import (
    _weight_delta as _weight_delta,
)
from scpn_control.control._federated_privacy import (
    DifferentialPrivacyConfig,
    PrivacyLedgerEntry,
    compose_privacy_epsilon,
    differential_privacy_clip,
    gaussian_mechanism_epsilon,
)
from scpn_control.control._federated_privacy import (
    _require_positive_float as _require_positive_float,
)
from scpn_control.control._federated_privacy import (
    _require_positive_int as _require_positive_int,
)
from scpn_control.control._federated_server import FederatedServer

logger = logging.getLogger(__name__)

__all__ = [
    "DifferentialPrivacyConfig",
    "FacilityBenchmarkSummary",
    "FederatedConfig",
    "FederatedServer",
    "MACHINE_PROFILES",
    "MachineClient",
    "N_FEATURES",
    "PrivacyLedgerEntry",
    "compose_privacy_epsilon",
    "create_facility_clients_from_arrays",
    "create_machine_clients",
    "differential_privacy_clip",
    "gaussian_mechanism_epsilon",
    "run_synthetic_multifacility_benchmark",
]
