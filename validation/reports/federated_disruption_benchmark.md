# Federated disruption benchmark

- Evidence kind: synthetic multi-facility
- Aggregation: `fedprox`
- Facilities: DIII-D, JET, KSTAR, EAST
- Rounds: 4
- Mean accuracy: 0.666667
- Mean loss: 0.705369
- Differential privacy epsilon: 7.751688
- Differential privacy delta: 1.0e-05

Per-facility final accuracy:

- `DIII-D`: 0.716667
- `EAST`: 0.633333
- `JET`: 0.600000
- `KSTAR`: 0.716667

Claim boundary: this report exercises the production federation,
heterogeneity, and facility-update differential privacy contracts on
deterministic synthetic facility distributions. It does not claim
measured cross-facility validation against DIII-D, JET, KSTAR, or EAST
shot databases.
