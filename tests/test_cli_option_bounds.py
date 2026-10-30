# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — upper bounds on CLI options that size work

"""CLI options that size work, memory or sockets declare an inclusive upper bound.

A value one past the bound is refused while the command line is parsed, so no
simulation, benchmark, campaign or server starts.
"""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from scpn_control.cli import main

BOUNDED = {
    ("demo", "steps"): 1_000_000,
    ("benchmark", "n_bench"): 10_000_000,
    ("benchmark", "n_warmup"): 10_000_000,
    ("run-hardware-campaign", "neurons"): 65_536,
    ("run-hardware-campaign", "transport_port"): 65_535,
    ("run-hardware-campaign", "transport_max_queue"): 4_096,
    ("run-hardware-campaign", "heartbeat_port"): 65_535,
    ("run-hardware-campaign", "steps"): 100_000_000,
    ("run-hardware-campaign", "formal_depth"): 64,
    ("run-hardware-campaign", "formal_max_marking"): 1_000_000,
    ("run-hardware-campaign", "formal_channel_capacity"): 4_096,
    ("live", "port"): 65_535,
    ("live", "layers"): 1_024,
    ("live", "n_per"): 100_000,
}


def _option(command: str, name: str) -> click.Option:
    """Return one option of one subcommand of the CLI."""
    group = main
    assert isinstance(group, click.Group)
    sub = group.commands[command]
    found = next(p for p in sub.params if p.name == name)
    assert isinstance(found, click.Option)
    return found


@pytest.mark.parametrize(("key", "bound"), BOUNDED.items(), ids=lambda v: str(v))
def test_work_sizing_option_declares_its_upper_bound(key: tuple[str, str], bound: int) -> None:
    """Declare the inclusive maximum on every work-sizing option."""
    option = _option(*key)
    assert isinstance(option.type, click.IntRange)
    assert option.type.max == bound
    assert not option.type.max_open


@pytest.mark.parametrize(("key", "bound"), BOUNDED.items(), ids=lambda v: str(v))
def test_one_past_the_bound_is_refused_before_work(key: tuple[str, str], bound: int) -> None:
    """Refuse the bound plus one with a usage error, before the command body runs."""
    command, name = key
    flag = "--" + name.replace("_", "-")
    result = CliRunner().invoke(main, [command, flag, str(bound + 1)])
    assert result.exit_code == 2
    assert "Invalid value" in result.output
    assert f"{bound}" in result.output
