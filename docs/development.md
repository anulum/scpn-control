# Development

## Local setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Docs site preview

```bash
python -m pip install mkdocs
mkdocs serve
```

Then open `http://127.0.0.1:8000/`.

## Release-quality checklist

1. Execute all notebooks in `examples/`.
2. Run Python test suite.
3. Run Rust build, clippy, and tests.
4. Run Rust/Python parity tests if bindings are enabled.
5. Confirm CI and Pages workflows pass on `main`.
