# Release and PyPI

This page documents the release flow before tagging and publishing.

## Pre-release checklist

1. Ensure version is updated in `pyproject.toml`.
2. Ensure `CITATION.cff` version/date are updated.
3. Update `CHANGELOG.md`.
4. Run docs build:

```bash
mkdocs build --strict
```

5. Run validation baseline:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p hypothesis.extra.pytestplugin tests/ -q
cd scpn-control-rs
cargo build --workspace
cargo clippy --workspace -- -D warnings
cargo test --workspace
cargo bench --workspace
```

6. Run notebook execution:

```bash
# Self-contained notebooks
jupyter nbconvert --to notebook --execute --output-dir artifacts/notebook-exec examples/q10_breakeven_demo.ipynb
jupyter nbconvert --to notebook --execute --output-dir artifacts/notebook-exec examples/snn_compiler_walkthrough.ipynb

# Optional: requires sc_neurocore availability
jupyter nbconvert --to notebook --execute --output-dir artifacts/notebook-exec examples/neuro_symbolic_control_demo.ipynb
```

## Build package locally

```bash
python -m pip install --upgrade pip build twine
python -m build
python -m twine check dist/*
```

## Tag and release

After merging release prep to `main`:

```bash
git checkout main
git pull
git tag v0.1.0
git push origin v0.1.0
```

Then create a GitHub Release for that tag and paste relevant changelog entries.

## PyPI publishing options

### Recommended: Trusted Publisher (GitHub Actions)

This repository includes `.github/workflows/publish-pypi.yml`.

Configure trusted publishing on PyPI (and optionally TestPyPI):

1. PyPI -> `scpn-control` -> Publishing -> Add Trusted Publisher.
2. Use repository `anulum/scpn-control`.
3. Workflow file: `.github/workflows/publish-pypi.yml`.
4. Leave environment unset unless you also bind the workflow job to a GitHub environment.

After configuration:

- Push tag `v*` to publish to PyPI automatically.
- Or manually trigger workflow dispatch with `testpypi` first.

### Manual fallback: Twine token upload

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

Use `TWINE_USERNAME=__token__` and `TWINE_PASSWORD=<pypi-token>` when prompted or via environment.

## Post-release checks

1. Verify install from PyPI:

```bash
python -m pip install --upgrade scpn-control
python -m scpn_control.cli --version
```

2. Verify docs and badges are green:

- CI badge
- Docs Pages badge
- PyPI package page
