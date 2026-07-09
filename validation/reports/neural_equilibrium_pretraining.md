# Neural equilibrium synthetic pretraining benchmark

- Evidence kind: synthetic pretraining
- Samples: 2048
- Grid: 65 x 65
- PCA components: 20
- Explained variance: 0.99706694
- Train MSE: 1.14853121e-02
- Validation MSE: 1.14849560e-02
- Test MSE: 1.25451889e-02
- Test max error: 7.64689374e-01
- GS residual: 2.83134332e-05
- Weights SHA-256: `1984585aa58569b4cfffeefe4173f80b2617679f4602734fa098c5572e45830b`
- Claim admission: synthetic pretraining evidence only; predictive EFIT/P-EFIT claims blocked
- Facility claim allowed: False

Claim boundary: this benchmark trains JAX-compatible PCA plus MLP
weights on bounded synthetic Solovev-like equilibria. It is useful
for pretraining and performance plumbing, but it is not real EFIT
or P-EFIT validation. Real fine-tuning remains gated by persisted
reference artefacts in `validation/reports/neural_equilibrium_reference/`.
