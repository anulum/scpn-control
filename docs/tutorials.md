# Tutorials

This project ships three executable notebooks under `examples/`:

1. `neuro_symbolic_control_demo.ipynb`  
   End-to-end closed-loop controller walkthrough.
2. `q10_breakeven_demo.ipynb`  
   Transport, breakeven, and robust control analysis.
3. `snn_compiler_walkthrough.ipynb`  
   Stochastic Petri net to SNN compilation and artifact export.

## Run one notebook

```bash
jupyter nbconvert --to notebook --execute examples/neuro_symbolic_control_demo.ipynb --output-dir artifacts/notebook-exec
```

## Run all notebooks

```bash
jupyter nbconvert --to notebook --execute --output-dir artifacts/notebook-exec examples/*.ipynb
```

## Notes

- Notebook outputs are written to `artifacts/notebook-exec/`.
- If Rust bindings are available, parity checks can be run from the validation
  workflow described in [Validation and QA](validation.md).
