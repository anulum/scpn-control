# FAIR-MAST Disruption Evaluation

- **Status**: blocked (admission_ready=False)
- **Dataset**: mast-disruption-campaign01 (synthetic=False)
- **Shots**: 93 (65 disruptive)
- **Window**: 128 samples · alarm threshold 0.65
- **AUC**: 0.5349

## Warning-time recall (disruptive shots alarmed ≥ N ms before disruption)

| Warning (ms) | Recall |
| --- | --- |
| 10 | 0.6769 |
| 20 | 0.6769 |
| 30 | 0.6769 |
| 50 | 0.6769 |
| 100 | 0.6769 |

> Bounded internal ROC/AUC for the fixed-weight disruption heuristic; the disruption-risk claim boundary remains locked pending a real fit and a sealed reference artefact.

Predictor: scores the dBdt window plus n=1/n=2 toroidal observables; does not consume the Rea-2019 q95/beta_N/li/Greenwald/P_rad features.
