# Validation Deficiencies — scpn-control

This document records physics and model-data alignment deficiencies revealed by
tightened validation thresholds in v0.15.0. These thresholds were tightened to
meet publication standards (e.g., JOSS, Nuclear Fusion).

## 1. Equilibrium Self-Consistency (Lane 1)

**Threshold:** `psi_nrmse < 2.5%` (Tightened from 5.0%)
**Result:** 7/13 pass (54%) — **FAIL**

### Identified Deficiencies

1. **DIII-D High-Beta and Shaped Shots:**
   - `diiid_hmode_1p5MA.geqdsk` (6.98%)
   - `diiid_hmode_2MA.geqdsk` (6.81%)
   - `diiid_negdelta.geqdsk` (8.82%)
   - **Root Cause:** The validation proxy for NRMSE computes the $\Delta^* \psi$
     operator without subtracting the plasma source term ($R p' + FF'/\mu_0 R$).
     In highly shaped or high-beta DIII-D shots, the neglected source term
     leads to a large residual, which the proxy misinterprets as non-consistency.
   - **Impact:** Tightened thresholds reveal that the current validation script
     is inadequate for highly active plasma regions.

2. **SPARC Start-up Phase (`sparc_1300.eqdsk`):**
   - NRMSE: 7.84%
   - q95: -36.67 (Anomalous)
   - **Root Cause:** This equilibrium represents a start-up or transition phase
     where the magnetic axis is poorly defined or the plasma current is
     extremely low. The $q$-profile in the reference GEQDSK is likely
     unphysical or uses a different normalization convention.

### Mitigation Plan (v2.1)
- Refactor `validate_real_shots.py` to reconstruct the toroidal current
  density $J_\phi$ from GEQDSK $p'$ and $FF'$ profiles.
- Implement a true GS residual check: $|| \Delta^* \psi + \mu_0 R J_\phi || / || \psi ||$.

## 2. Disruption Prediction (Lane 3)

**Threshold:** `recall > 80%`, `FPR < 25%`
**Result:** Recall 100%, FPR 0% — **PASS**

*Note:* Although the current test set passes, the synthetic ROC analysis
(Task V4) suggests that the FPR may degrade under more diverse noise
conditions.

## 3. Transport Scaling (Lane 2)

**Threshold:** `80% within 2-sigma`
**Result:** 95% within 2-sigma — **PASS**

The IPB98(y,2) scaling remains robust across the ITPA database for the
current parameter range.
