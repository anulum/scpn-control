# Neural Transport Surrogate Training Recipe

This guide describes how to train or retrain the neural transport surrogate
used in `scpn-control`. The surrogate is a compact Multi-Layer Perceptron (MLP)
that replaces expensive gyrokinetic simulations (like TGLF or QuaLiKiz)
with millisecond-scale inference.

## 1. Prerequisites

- **Python**: 3.9 or newer.
- **Framework**: PyTorch (recommended) or a raw NumPy-based trainer.
- **Data Science Stack**: `numpy`, `pandas`, `scikit-learn`.
- **Hardware**: CPU is sufficient for this compact architecture (< 100k parameters).

## 2. Download Dataset

The default surrogate is trained on the **QLKNN-10D** dataset, which provides
turbulent fluxes for a wide range of tokamak plasma parameters.

- **Source**: Zenodo ([doi:10.5281/zenodo.3700755](https://doi.org/10.5281/zenodo.3700755))
- **File Format**: CSV or HDF5.
- **Columns**: 10 input features and 3--7 output fluxes.
- **Units**: Normalised gradients (R/L), temperatures (keV), and fluxes
  (Gyro-Bohm units or m²/s).

## 3. Data Preparation

### Feature Selection
The 10 input features required by the `TransportInputs` class are:
1. `R/L_Ti`: Ion temperature gradient.
2. `R/L_Te`: Electron temperature gradient.
3. `R/L_ne`: Electron density gradient.
4. `q`: Safety factor.
5. `s_hat`: Magnetic shear.
6. `alpha_MHD`: Pressure gradient parameter.
7. `Te/Ti`: Temperature ratio.
8. `Z_eff`: Effective charge.
9. `nu_star`: Collisionality.
10. `beta_e`: Electron beta.

### Processing Pipeline
1. **Filtering**: Remove non-physical samples (e.g., negative temperatures).
2. **Normalisation**: Use `StandardScaler` to reach zero mean and unit variance.
   **Note:** Store the mean and scale values; they must be provided to
   `NeuralTransportModel` for inference.
3. **Split**: 80% Train, 10% Validation, 10% Test.

## 4. Architecture

We use a three-layer MLP: **10 → 128 → 64 → 3**.

- **Hidden Layers**: 128 and 64 neurons with **ReLU** activation.
- **Output Layer**: 3 neurons (`chi_e`, `chi_i`, `D_e`) with **Softplus**
  activation to ensure positive diffusivities.
- **Design Goal**: Compactness. This architecture achieves < 25 µs
  inference time on a single CPU thread, enabling integration into
  10kHz control loops.

## 5. Training Loop

### Hyperparameters
- **Batch Size**: 512
- **Optimizer**: Adam
- **Initial Learning Rate**: 1e-3
- **Scheduler**: `ReduceLROnPlateau` (factor 0.5, patience 10)
- **Early Stopping**: Patience 20 on validation loss.

### Code Snippet (PyTorch)
```python
model = nn.Sequential(
    nn.Linear(10, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 3),
    nn.Softplus()
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

## 6. Weight Export

The `NeuralTransportModel` expects weights in a specific `.npz` format.
After training, export the state dictionary using NumPy:

```python
import numpy as np

# Extract weights from PyTorch model
w1 = model[0].weight.detach().numpy().T
b1 = model[0].bias.detach().numpy()
w2 = model[2].weight.detach().numpy().T
b2 = model[2].bias.detach().numpy()
w3 = model[4].weight.detach().numpy().T
b3 = model[4].bias.detach().numpy()

np.savez("neural_transport_custom.npz",
         w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3,
         input_mean=scaler.mean_,
         input_scale=scaler.scale_)
```

## 7. Validation

After exporting, verify the new weights using the provided test suite:

1. Update the weight path in your local config or environment variable.
2. Run `pytest tests/test_neural_transport_core.py` to check for shape
   and consistency errors.
3. Run `pytest tests/test_neural_transport_physics.py` to ensure the
   new model respects fundamental plasma trends (monotonicity, thresholds).

## 8. Retraining from Custom Data

To adapt the surrogate for different regimes:
- **ETG-dominant**: Include higher resolution in $R/L_{Te}$ and focus
  training on electron flux columns.
- **Stellarators**: You may need to add additional geometry features
  (e.g., helical ripple) and increase hidden layer width to 256.
- **Data Sources**: For custom tokamak configurations, use the
  `scpn_control.core.tglf_adapter` (if available) to generate
  local simulation batches.
