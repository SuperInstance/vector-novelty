.PHONY: test coverage lint format type-check security install clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -q --tb=short

test-verbose:
	python -m pytest tests/ -v --tb=short

coverage:
	python -m pytest tests/ --cov=vector_novelty --cov-report=term-missing --cov-fail-under=75

lint:
	ruff check vector_novelty/

format:
	ruff format vector_novelty/

type-check:
	mypy vector_novelty/ --ignore-missing-imports

security:
	bandit -r vector_novelty/ -ll
	pip-audit --desc .

dev: install lint type-check test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/
