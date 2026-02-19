# Tutorials

This project ships three executable notebooks under `examples/`:

1. `neuro_symbolic_control_demo.ipynb`  
   End-to-end closed-loop controller walkthrough.
2. `q10_breakeven_demo.ipynb`  
   Transport, breakeven, and robust control analysis.
3. `snn_compiler_walkthrough.ipynb`  
   Stochastic Petri net to SNN compilation and artifact export.

## Prerequisites

```bash
python -m pip install -e ".[viz]" jupyter nbconvert
```

## Run one notebook

```bash
jupyter nbconvert --to notebook --execute examples/q10_breakeven_demo.ipynb --output-dir artifacts/notebook-exec
```

## Run CI-equivalent notebook smoke set

```bash
jupyter nbconvert --to notebook --execute --output-dir artifacts/notebook-exec examples/q10_breakeven_demo.ipynb
jupyter nbconvert --to notebook --execute --output-dir artifacts/notebook-exec examples/snn_compiler_walkthrough.ipynb
```

## Run full notebook set (optional dependency)

`neuro_symbolic_control_demo.ipynb` requires `sc_neurocore`.

```bash
jupyter nbconvert --to notebook --execute --output-dir artifacts/notebook-exec examples/neuro_symbolic_control_demo.ipynb
```

## Notes

- Notebook outputs are written to `artifacts/notebook-exec/`.
- CI runs the smoke set by default and skips the neuro-symbolic notebook when
  `sc_neurocore` is not available.
- If Rust bindings are available, parity checks can be run from the validation
  workflow described in [Validation and QA](validation.md).
