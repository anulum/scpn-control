# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI reference-artifact validation commands

"""Register existing reference commands and preserve their original import surface.

Cohesive family modules own each callback. This facade reexports the same Click
objects used by the root group, retaining registration order and hidden legacy
compatibility. Source-validator contracts govern declaration/runtime admission.
"""

from __future__ import annotations

import click

from scpn_control.cli_reference_engineering import (
    validate_burn_reference_command as validate_burn_reference_command,
)
from scpn_control.cli_reference_engineering import (
    validate_current_drive_reference_command as validate_current_drive_reference_command,
)
from scpn_control.cli_reference_engineering import (
    validate_density_reference_command as validate_density_reference_command,
)
from scpn_control.cli_reference_engineering import (
    validate_volt_second_reference_command as validate_volt_second_reference_command,
)
from scpn_control.cli_reference_equilibrium import (
    _ReportOutputPath as _ReportOutputPath,
)
from scpn_control.cli_reference_equilibrium import (
    validate_ida_same_case_command as validate_ida_same_case_command,
)
from scpn_control.cli_reference_equilibrium import (
    validate_neural_equilibrium_reference_command as validate_neural_equilibrium_reference_command,
)
from scpn_control.cli_reference_equilibrium import (
    validate_orbit_reference_command as validate_orbit_reference_command,
)
from scpn_control.cli_reference_equilibrium import (
    validate_uncertainty_reference_command as validate_uncertainty_reference_command,
)
from scpn_control.cli_reference_equilibrium import (
    validate_vmec_reference_command as validate_vmec_reference_command,
)
from scpn_control.cli_reference_instabilities import (
    validate_disruption_reference_command as validate_disruption_reference_command,
)
from scpn_control.cli_reference_instabilities import (
    validate_elm_reference_command as validate_elm_reference_command,
)
from scpn_control.cli_reference_instabilities import (
    validate_eped_reference_command as validate_eped_reference_command,
)
from scpn_control.cli_reference_instabilities import (
    validate_marfe_reference_command as validate_marfe_reference_command,
)
from scpn_control.cli_reference_instabilities import (
    validate_ntm_reference_command as validate_ntm_reference_command,
)
from scpn_control.cli_reference_kinetic import (
    _split_csv_option as _split_csv_option,
)
from scpn_control.cli_reference_kinetic import (
    validate_gk_crosscode_command as validate_gk_crosscode_command,
)
from scpn_control.cli_reference_kinetic import (
    validate_gk_geometry_reference_command as validate_gk_geometry_reference_command,
)
from scpn_control.cli_reference_kinetic import (
    validate_gk_interface_artifacts_command as validate_gk_interface_artifacts_command,
)
from scpn_control.cli_reference_kinetic import (
    validate_gk_ood_calibration_command as validate_gk_ood_calibration_command,
)
from scpn_control.cli_reference_kinetic import (
    validate_gk_species_reference_command as validate_gk_species_reference_command,
)
from scpn_control.cli_reference_kinetic import (
    validate_jax_gk_parity_command as validate_jax_gk_parity_command,
)
from scpn_control.cli_reference_static_mu import (
    _run_static_mu_analysis_reference as _run_static_mu_analysis_reference,
)
from scpn_control.cli_reference_static_mu import (
    validate_mu_synthesis_reference_compatibility_command as validate_mu_synthesis_reference_compatibility_command,
)
from scpn_control.cli_reference_static_mu import (
    validate_static_mu_analysis_reference_command as validate_static_mu_analysis_reference_command,
)
from scpn_control.cli_reference_tracking import (
    validate_digital_twin_reference_command as validate_digital_twin_reference_command,
)
from scpn_control.cli_reference_tracking import (
    validate_free_boundary_reference_command as validate_free_boundary_reference_command,
)
from scpn_control.cli_reference_tracking import (
    validate_rzip_reference_command as validate_rzip_reference_command,
)
from scpn_control.cli_reference_transport import (
    validate_blob_transport_reference_command as validate_blob_transport_reference_command,
)
from scpn_control.cli_reference_transport import (
    validate_neural_transport_reference_command as validate_neural_transport_reference_command,
)
from scpn_control.cli_reference_transport import (
    validate_neural_turbulence_reference_command as validate_neural_turbulence_reference_command,
)
from scpn_control.cli_reference_transport import (
    validate_soc_reference_command as validate_soc_reference_command,
)

REFERENCE_VALIDATOR_COMMANDS: tuple[click.Command, ...] = (
    validate_gk_crosscode_command,
    validate_gk_geometry_reference_command,
    validate_gk_species_reference_command,
    validate_jax_gk_parity_command,
    validate_gk_ood_calibration_command,
    validate_gk_interface_artifacts_command,
    validate_blob_transport_reference_command,
    validate_elm_reference_command,
    validate_eped_reference_command,
    validate_marfe_reference_command,
    validate_ntm_reference_command,
    validate_neural_equilibrium_reference_command,
    validate_neural_transport_reference_command,
    validate_neural_turbulence_reference_command,
    validate_orbit_reference_command,
    validate_uncertainty_reference_command,
    validate_vmec_reference_command,
    validate_rzip_reference_command,
    validate_current_drive_reference_command,
    validate_static_mu_analysis_reference_command,
    validate_mu_synthesis_reference_compatibility_command,
    validate_volt_second_reference_command,
    validate_burn_reference_command,
    validate_density_reference_command,
    validate_free_boundary_reference_command,
    validate_disruption_reference_command,
    validate_digital_twin_reference_command,
    validate_soc_reference_command,
    validate_ida_same_case_command,
)
