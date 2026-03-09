# SPDX-License-Identifier: MIT OR Apache-2.0
.PHONY: test test-rust test-all lint fmt bandit sast preflight preflight-fast docs docs-build bench bench-rust bridge build install-hooks docker-build docker-run clean

test:
	pytest tests/ -v --cov=scpn_control --cov-report=term --cov-fail-under=85

test-rust:
	cd scpn-control-rs && cargo test --workspace

test-all: test test-rust

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

fmt:
	ruff check --fix src/ tests/
	ruff format src/ tests/
	cd scpn-control-rs && cargo fmt

bandit:
	bandit -r src/scpn_control/ -c pyproject.toml -q

sast: bandit

preflight:
	python tools/preflight.py

preflight-fast:
	python tools/preflight.py --no-tests

docs:
	mkdocs serve

docs-build:
	mkdocs build --strict

bench:
	scpn-control benchmark --n-bench 5000

bench-rust:
	cd scpn-control-rs && cargo bench --workspace

bridge:
	cd scpn-control-rs/crates/control-python && maturin develop --release

build:
	python -m build

install-hooks:
	git config core.hooksPath .githooks
	@echo "Git hooks installed (.githooks/pre-push)"

docker-build:
	docker build -t scpn-control:latest .

docker-run:
	docker run --rm -it scpn-control:latest

clean:
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	cd scpn-control-rs && cargo clean
