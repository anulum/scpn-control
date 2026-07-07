# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Studio package facade tests
"""Studio package facade contract: lazy export resolution, unknown-attribute
rejection, dir() listing, export-map consistency, and the SDK-free import
guarantee for the sealed-claim submodule."""

from __future__ import annotations

import importlib
import sys

import pytest

import scpn_control.studio as studio


def test_sealed_claim_submodule_imports_without_the_sdk() -> None:
    """The sealed-claim module must import even when the SDK is absent.

    Reloads the package and imports the submodule under an import hook that
    blocks ``scpn_studio_platform`` — the exact CI situation on matrices
    without the ``studio`` extra.
    """

    class _BlockSdk:
        def find_spec(self, name: str, *args: object, **kwargs: object) -> None:
            if name.split(".")[0] == "scpn_studio_platform":
                raise ModuleNotFoundError(f"No module named {name!r} (blocked for test)")
            return None

    blocker = _BlockSdk()
    # Purge the SDK AND every already-imported studio submodule: earlier tests
    # in a full-suite run import studio.evidence with the SDK present, and both
    # importlib.reload and the lazy facade's globals cache would otherwise
    # satisfy the attribute access without touching the blocked SDK import.
    purge_roots = ("scpn_studio_platform",)
    purge_exact = {
        "scpn_control.studio.adapters",
        "scpn_control.studio.evidence",
        "scpn_control.studio.feed",
        "scpn_control.studio.manifest",
        "scpn_control.studio.verbs",
    }
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.split(".")[0] in purge_roots or k in purge_exact}
    sys.meta_path.insert(0, blocker)
    try:
        pkg = importlib.reload(studio)
        for name in list(vars(pkg)):
            if name in pkg._EXPORT_MODULES:
                delattr(pkg, name)  # drop lazily cached symbols from prior tests
        mod = importlib.import_module("scpn_control.studio.sealed_claim")
        assert mod.SEALED_SAFETY_CLAIM_SCHEMA == "scpn-control.sealed-safety-claim.v1"
        with pytest.raises(ModuleNotFoundError):
            _ = pkg.safety_certificate_evidence
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)
        importlib.reload(studio)


def test_lazy_export_resolves_and_caches_sdk_free_symbol() -> None:
    pkg = importlib.reload(studio)
    fn = pkg.build_safety_certificate_sealed_claim
    assert callable(fn)
    # second access must hit the cached global, not the lazy path
    assert pkg.build_safety_certificate_sealed_claim is fn
    assert "build_safety_certificate_sealed_claim" in vars(pkg)


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'nonexistent_symbol'"):
        _ = studio.nonexistent_symbol


def test_dir_lists_every_public_export() -> None:
    listing = dir(studio)
    for name in studio.__all__:
        assert name in listing


def test_export_map_and_all_are_consistent() -> None:
    assert sorted(studio.__all__) == sorted(studio._EXPORT_MODULES)
    for name, module_name in studio._EXPORT_MODULES.items():
        assert module_name.startswith("scpn_control.studio."), (name, module_name)
