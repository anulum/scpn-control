# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Core Package Init
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Core solver and plant model.

Unlike the upstream core/__init__.py, this does NOT eagerly import
equilibrium_3d, gpu_runtime, gyro_swin_surrogate, or pretrained_surrogates.
Only the minimal FusionKernel is loaded.
"""
try:
    from scpn_control.core._rust_compat import FusionKernel, RUST_BACKEND
except ImportError:
    from scpn_control.core.fusion_kernel import FusionKernel
    RUST_BACKEND = False

__all__ = ["FusionKernel", "RUST_BACKEND"]
