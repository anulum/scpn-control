# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.23.x  | :white_check_mark: |
| < 0.23  | :x:                |

Only the latest `0.x` release receives security fixes. Upgrade with:

```bash
pip install --upgrade scpn-control
```

## Reporting a Vulnerability

If you discover a security vulnerability in SCPN Control, please report it
responsibly:

1. **Preferred:** [GitHub Security Advisories](https://github.com/anulum/scpn-control/security/advisories/new)
2. **Email:** protoscience@anulum.li (subject: `[SECURITY] SCPN Control`)
3. **Do not** open a public GitHub issue for security vulnerabilities.

We will acknowledge receipt within 48 hours and aim to provide a fix within
7 days for critical issues.

## Scope

SCPN Control is a simulation and control library. Its default installation
handles no user authentication or financial data and exposes no network
service. The optional `ws` extra provides a WebSocket phase-stream server
(`scpn_control.phase.ws_phase_stream`) with HMAC API-key authentication and
optional TLS; it binds `127.0.0.1` by default and is a network service only
once a user installs that extra and starts the server. Security concerns are
primarily:

- Malicious input files (JSON configs, GEQDSK equilibria, NumPy `.npz`)
- Unsafe deserialization (serde, pickle, NumPy load)
- Numerical overflow / denial of service via pathological inputs
- Native code memory safety (Rust crates via PyO3)
- Supply chain integrity (dependency audit via `cargo-deny`)

## Hardening Measures

- **Input validation:** Public API boundaries enforce finite-float, integer,
  fraction, and 1D-array checks via `core/_validators.py`.
- **Rust:** `cargo deny` supply-chain policy enforced in CI.
- **Checkpoint hygiene:** `torch.load(..., weights_only=True)` by default.
- **RNG isolation:** All stochastic modules use scoped `numpy.random.Generator`.
- **Pre-commit:** ruff, mypy, cargo fmt, merge-conflict, private-key detection.

## Known Limitations

- No fuzzing harness yet (Hypothesis covers many paths).
- No third-party security audit.
- No CVE history.

Contributions to improve security coverage are welcome.
