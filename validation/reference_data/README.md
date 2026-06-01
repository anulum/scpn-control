<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Validation Reference Data

Experimental and design reference data for cross-validating SCPN Fusion Core
simulation outputs against real tokamak parameters.

## Datasets

### SPARC (Commonwealth Fusion Systems)

Equilibrium files from the [SPARCPublic](https://github.com/cfs-energy/SPARCPublic)
repository. SPARC is a compact, high-field (B = 12.2 T) tokamak under construction
by CFS, designed to achieve Q > 2 in D-T plasmas.

| File | Description | Grid | I_p (MA) | B_T (T) |
|------|-------------|------|----------|---------|
| `lmode_vv.geqdsk` | L-mode lower single-null, vertical-vertical divertor | 129x129 | 8.5 | 12.2 |
| `lmode_vh.geqdsk` | L-mode lower single-null, vertical-horizontal divertor | 129x129 | 8.5 | 12.2 |
| `lmode_hv.geqdsk` | L-mode lower single-null, horizontal-vertical divertor | 129x129 | 8.5 | 12.2 |
| `sparc_1300.eqdsk` | EQ library entry 1300 (low current, 0.2 MA) | 61x129 | 0.2 | 12.2 |
| `sparc_1305.eqdsk` | EQ library entry 1305 | 61x129 | — | 12.2 |
| `sparc_1310.eqdsk` | EQ library entry 1310 | 61x129 | — | 12.2 |
| `sparc_1315.eqdsk` | EQ library entry 1315 | 61x129 | — | 12.2 |
| `sparc_1349.eqdsk` | EQ library entry 1349 (full current, 8.0 MA) | 61x129 | 8.0 | 12.2 |
| `device_description.json` | IMAS-format device description (coils, wall) | — | — | — |
| `prd_popcon.csv` | Primary Reference Discharge POPCON data | — | — | — |

**License:** See https://github.com/cfs-energy/SPARCPublic for terms.

### ITPA H-Mode Confinement Database

Curated subset of the International Tokamak Physics Activity (ITPA) global
H-mode confinement database, covering 18 tokamaks worldwide.

| File | Description |
|------|-------------|
| `hmode_confinement.csv` | Machine parameters and measured τ_E for 20 entries |
| `ipb98y2_coefficients.json` | IPB98(y,2) scaling law coefficients and uncertainties |

**Source:** Verdoolaege et al., Nuclear Fusion 61 (2021) 076006

### QLKNN and QuaLiKiz Public Acquisition Metadata

Public Zenodo metadata and small auxiliary files are mirrored under
`qlknn/` for neural-transport acquisition planning:

| Directory | Dataset | DOI | Local status |
|-----------|---------|-----|--------------|
| `qlknn/zenodo_3497066/` | QLKNN10D training set | `10.5281/zenodo.3497066` | Normalised file manifest acquired |
| `qlknn/zenodo_7418108/` | QuaLiKiz v2.6.2 JET linear-instability spectra | `10.5281/zenodo.7418108` | Normalised file manifest acquired |
| `qlknn/zenodo_8017522/` | QLKNN11D training set | `10.5281/zenodo.8017522` | Normalised file manifest acquired |

The multi-GB NetCDF/HDF5 tensor payloads are intentionally not stored in the
repository. Raw Zenodo record payloads are also not vendored because their HTML
descriptions are third-party publication text. Each `files_manifest.json`
records the Zenodo API download URL, size, MD5 checksum, source DOI, record
digest, and a deferred-download policy. Validate the acquisition metadata with:

```bash
python validation/validate_public_data_acquisition.py --json-out
```

These manifests are acquisition readiness evidence only. They do not constitute
trained neural-transport weights, quantitative QuaLiKiz agreement, or measured
facility validation until the large numeric files are downloaded on an admitted
storage target and processed into strict reference-artifact evidence.

### ITER Configurations (existing)

Four ITER-scale validation configurations with different coil current optimisations:

| File | Description |
|------|-------------|
| `../iter_validated_config.json` | Human-designed baseline (I_p = 15 MA) |
| `../iter_analytic_config.json` | Analytically optimised coil currents |
| `../iter_force_balanced.json` | Newton-Raphson force-balanced equilibrium |
| `../iter_genetic_config.json` | Genetic algorithm optimised |

## Usage

```python
from scpn_control.core.eqdsk import read_geqdsk

eq = read_geqdsk("validation/reference_data/sparc/lmode_vv.geqdsk")
print(f"R_axis = {eq.rmaxis:.3f} m, B_T = {eq.bcentr:.2f} T")
print(f"ψ(R,Z) shape: {eq.psirz.shape}")
```
