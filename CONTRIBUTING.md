# Contributing to vector-novelty

Thank you for your interest in the Cocapn Fleet. This document covers how to set up the project, run tests, and submit changes.

## Setup

```bash
git clone https://github.com/SuperInstance/vector-novelty.git
cd vector-novelty
pip install -e ".[dev]"
```

Requires **Python 3.10+** and **NumPy**.

## Running Tests

```bash
# Full suite
make test

# With coverage (threshold: 75%)
make coverage

# Verbose
make test-verbose
```

## Lint & Type Check

```bash
make lint       # ruff check
make format     # ruff format
make type-check # mypy
```

## Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Submitting Changes

1. **Branch**: Create a feature branch (`git checkout -b feature/name`)
2. **Tests**: All tests must pass. New features need new tests.
3. **Coverage**: Maintain or improve coverage (current threshold: 75%)
4. **Commit**: Use conventional commits (`feat:`, `fix:`, `docs:`, `test:`)
5. **PR**: Open against `main` with a clear description

## Code Style

- **Ruff** for linting and formatting
- **MyPy** for type checking (ignore missing imports for external deps)
- Docstrings for public APIs
- Type hints on function signatures

## Architecture Notes

- **Pure NumPy**: No PyTorch, no JAX. Stay lightweight.
- **O(n) only**: Centroid-based, never pairwise O(n²).
- **API surface**: `AgentVector`, `VectorTable`, `compute_novelty`, `batch_novelty`, `cosine_distance`

## Security

- Never commit secrets, API keys, or private keys
- Run `make security` before submitting
- The CI runs `trufflehog` to catch leaked credentials

## Questions?

Open an issue or reach out in `#cocapn-build` on Matrix.
