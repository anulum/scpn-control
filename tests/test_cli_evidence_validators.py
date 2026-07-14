# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI evidence-validator module contract tests

"""Module-surface tests for :mod:`scpn_control.cli_evidence_validators`.

The per-command behaviour is exercised through the root ``scpn-control`` group in
the ``test_cli_*`` suites; this file guards the module's own contract: the
``EVIDENCE_VALIDATOR_COMMANDS`` registration tuple that `scpn_control.cli` folds
onto the root group.
"""

from __future__ import annotations

import click

from scpn_control.cli import main
from scpn_control.cli_evidence_validators import EVIDENCE_VALIDATOR_COMMANDS


def test_evidence_validator_commands_are_click_commands() -> None:
    assert isinstance(EVIDENCE_VALIDATOR_COMMANDS, tuple)
    assert len(EVIDENCE_VALIDATOR_COMMANDS) == 6
    assert all(isinstance(command, click.Command) for command in EVIDENCE_VALIDATOR_COMMANDS)


def test_evidence_validator_command_names_are_unique_and_validate_prefixed() -> None:
    names = [command.name for command in EVIDENCE_VALIDATOR_COMMANDS]
    assert all(name is not None and name.startswith("validate") for name in names)
    assert len(names) == len(set(names)), "duplicate command name in EVIDENCE_VALIDATOR_COMMANDS"


def test_evidence_validator_commands_are_registered_on_root_group() -> None:
    for command in EVIDENCE_VALIDATOR_COMMANDS:
        assert main.commands.get(command.name) is command
