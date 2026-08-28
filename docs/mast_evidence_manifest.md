# MAST evidence manifest

This manifest exposes the reviewable public boundary of the bounded FAIR-MAST
campaign. It publishes method pages, schemas, report digests, and reproduction
commands. It does not publish raw traces, local dataset locations, coordination
paths, or operational reports.

Every listed gate currently reports `blocked`. A digest proves the identity of
a report if an operator supplies that report for review; it does not make the
private report itself public and does not promote any training, scientific,
facility-prediction, or control claim.

| Gate | Public method | Report schema | Payload SHA-256 | Report-file SHA-256 |
| --- | --- | --- | --- | --- |
| Campaign reconciliation | [Method and reproduction](mast_campaign_reconciliation.md) | `scpn-control.mast-campaign-lineage-reconciliation.v1.0.0` | `e80d62fde41765a7069a90bb2713f8d1b59fc263b80741fee6a68231c3532e46` | `39df8485da98a08b92be4b7b67a5b65586e3d70c84c8bdcce7f45721fa297988` |
| Toroidal field | [Method and reproduction](mast_toroidal_field_authority.md) | `scpn-control.mast-toroidal-field-authority.v1.0.0` | `5d65b0395f599d20cc63129c515ec99aa292e3af36aff50bfb1023c9888b3900` | `653835ec1d520b6aebd67bb6cd2e47f1b1797db8de60aae8b2ca7970b84f202b` |
| Normalised beta | [Method and reproduction](mast_normalised_beta_authority.md) | `scpn-control.mast-normalised-beta-authority.v1.0.0` | `6c8e8b4e2f394123106563da7dd5a5de264679d3a6417eb162a9a6ce3c9706a1` | `5dc6c759f34befb814587f9ddc455e5687bf4dc65c39c403c1e68736596a425e` |
| Saddle modal | [Method and reproduction](mast_saddle_modal_authority.md) | `scpn-control.mast-saddle-modal-authority.v1.0.0` | `0785c0e39e2dc4de826a32bbf38546bf52af5ad7e075bdd55956e36390188177` | `b321b337565ffe4a3e361752c5bdae426dc5d2d1a9910e5faa7d5bf15bab2919` |
| Locked mode | [Method and reproduction](mast_locked_mode_authority.md) | `scpn-control.mast-locked-mode-authority.v1.0.0` | `54c9fccb5d9308680fc8650b611e1c1e75b809eff23a33a779eca0b6bbe31f70` | `1ddc292e069da7939998fc4c2ef5d3c5c89a9508968c2e2c25b78de31e9cac2e` |
| dB/dt source | [Method and reproduction](mast_dbdt_authority.md) | `scpn-control.mast-dbdt-authority.v1.0.0` | `39c4d603e5275d6f1902d26f032dd870e7f319c936b9506b25c0e5e505604736` | `8ae8c8e5c43b621995727f562d8097673910f3c9c8cc44bae720b6f790617a84` |

## Review boundary

- The method pages link directly to the pinned FAIR-MAST, IMAS, DOI, and UKAEA
  sources that support their public statements.
- Reproduction commands require a caller-supplied, verified external dataset
  root. They never imply that private operational storage is publicly
  reachable.
- Each gate verifies its own schema and canonical payload digest. File digests
  above bind the exact bounded reports reviewed on 22–23 July 2026.
- The repository publishes no access token, signed URL, authenticated endpoint,
  workstation path, storage-host path, or coordination-tree reference.

The deterministic document-link audit checks this public graph locally on
every documentation build. A separate scheduled job checks public external
URLs with bounded retries, per-host pacing, cache provenance, and explicit
transient or restricted classifications.
