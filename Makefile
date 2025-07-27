# Makefile for Observer Coordinator Insights
# Provides standardized build commands across different environments

.PHONY: help install install-dev clean test lint format typecheck security build docker docker-compose up down logs shell docs serve-docs release

# Default target
help: ## Show this help message
	@echo "Observer Coordinator Insights - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Environment Setup
install: ## Install production dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev,docs,testing]"
	pre-commit install

install-all: ## Install all dependencies including optional ones
	pip install -e ".[dev,docs,testing]"
	pip install -r requirements-dev.txt
	pre-commit install

# Cleaning
clean: ## Clean build artifacts and temporary files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-all: clean ## Clean everything including logs and outputs
	rm -rf logs/
	rm -rf output/
	rm -rf reports/
	rm -rf cache/
	rm -rf .env

# Testing
test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v -m unit

test-integration: ## Run integration tests only
	pytest tests/integration/ -v -m integration

test-e2e: ## Run end-to-end tests only
	pytest tests/e2e/ -v -m e2e

test-performance: ## Run performance tests only
	pytest tests/performance/ -v -m performance --benchmark-only

test-security: ## Run security tests only
	pytest tests/security/ -v -m security

test-coverage: ## Run tests with coverage report
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing --cov-report=xml

test-fast: ## Run tests excluding slow ones
	pytest tests/ -v -m "not slow"

test-parallel: ## Run tests in parallel
	pytest tests/ -v -n auto

# Code Quality
lint: ## Run linting checks
	ruff check src/ tests/
	mypy src/
	bandit -r src/ -f json -o bandit-report.json || true

lint-fix: ## Run linting with auto-fix
	ruff check --fix src/ tests/
	ruff format src/ tests/

format: ## Format code
	ruff format src/ tests/

format-check: ## Check code formatting
	ruff format --check src/ tests/

typecheck: ## Run type checking
	mypy src/

typecheck-report: ## Generate type checking report
	mypy src/ --html-report mypy-report/

# Security
security: ## Run security scans
	bandit -r src/ -f json -o bandit-report.json
	safety check
	pip-audit

security-full: ## Run comprehensive security scan
	bandit -r src/ -f json -o bandit-report.json
	safety check --full-report
	pip-audit --format=json --output=pip-audit-report.json
	semgrep --config=auto src/

# Building
build: ## Build distribution packages
	python -m build

build-wheel: ## Build wheel package only
	python -m build --wheel

build-sdist: ## Build source distribution only
	python -m build --sdist

# Docker
docker: ## Build Docker image
	docker build -t observer-coordinator-insights:latest .

docker-dev: ## Build development Docker image
	docker build -f Dockerfile.dev -t observer-coordinator-insights:dev .

docker-compose: ## Build all services with docker-compose
	docker-compose build

up: ## Start services with docker-compose
	docker-compose up -d

down: ## Stop services with docker-compose
	docker-compose down

logs: ## View docker-compose logs
	docker-compose logs -f

shell: ## Open shell in development container
	docker-compose exec app bash

# Documentation
docs: ## Build documentation
	cd docs && sphinx-build -b html . _build/html

docs-clean: ## Clean documentation build
	cd docs && rm -rf _build/

docs-live: ## Build and serve documentation with auto-reload
	cd docs && sphinx-autobuild . _build/html --host 0.0.0.0 --port 8000

serve-docs: ## Serve built documentation
	cd docs/_build/html && python -m http.server 8000

# Development
dev: ## Start development server
	python src/main.py

dev-debug: ## Start development server with debug logging
	DEBUG=true LOG_LEVEL=DEBUG python src/main.py

dev-watch: ## Start development server with file watching
	watchmedo auto-restart --directory=src/ --pattern="*.py" --recursive -- python src/main.py

# Data Management
sample-data: ## Generate sample data for testing
	python scripts/generate_sample_data.py

validate-data: ## Validate existing data files
	python scripts/validate_data.py data/

clean-data: ## Clean and anonymize data files
	python scripts/clean_data.py

# Analysis
analyze: ## Run full analysis pipeline
	python autonomous_orchestrator.py

analyze-sample: ## Run analysis on sample data
	python autonomous_orchestrator.py --config config/sample.yml --input data/sample.csv

benchmark: ## Run performance benchmarks
	python scripts/benchmark.py

profile: ## Run performance profiling
	python -m cProfile -o profile.stats src/main.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Release Management
release-patch: ## Create patch release
	bump2version patch
	git push origin main --tags

release-minor: ## Create minor release
	bump2version minor
	git push origin main --tags

release-major: ## Create major release
	bump2version major
	git push origin main --tags

release-dry-run: ## Dry run release process
	bump2version --dry-run --verbose patch

# CI/CD Support
ci-install: ## Install dependencies for CI
	pip install -e ".[dev,testing]"

ci-test: ## Run tests in CI environment
	pytest tests/ --junitxml=test-results.xml --cov=src --cov-report=xml

ci-lint: ## Run linting in CI environment
	ruff check src/ tests/ --format=github
	mypy src/ --junit-xml=mypy-results.xml

ci-security: ## Run security checks in CI environment
	bandit -r src/ -f json -o bandit-report.json
	safety check --json --output safety-report.json

ci-build: ## Build in CI environment
	python -m build
	twine check dist/*

# Utilities
check-deps: ## Check for dependency updates
	pip list --outdated

licenses: ## Generate license report
	pip-licenses --format=json --output-file=licenses.json

tree: ## Show project structure
	tree -I '__pycache__|*.pyc|.git|.pytest_cache|htmlcov|.mypy_cache|.ruff_cache|build|dist|*.egg-info'

size: ## Show code statistics
	find src/ -name "*.py" -exec wc -l {} + | tail -1

# Environment Info
info: ## Show environment information
	@echo "Python Version: $$(python --version)"
	@echo "Pip Version: $$(pip --version)"
	@echo "Virtual Environment: $$VIRTUAL_ENV"
	@echo "Current Directory: $$(pwd)"
	@echo "Git Branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repo')"
	@echo "Git Commit: $$(git rev-parse --short HEAD 2>/dev/null || echo 'Not a git repo')"

# Load Testing
load-test: ## Run load tests
	locust -f tests/performance/locustfile.py --host=http://localhost:8000

# Database (Future Use)
db-migrate: ## Run database migrations
	@echo "Database migrations not yet implemented"

db-seed: ## Seed database with sample data
	@echo "Database seeding not yet implemented"

# Monitoring
health-check: ## Check application health
	curl -f http://localhost:8080/health || exit 1

metrics: ## Show application metrics
	curl -s http://localhost:9090/metrics | head -20

# Git Hooks
hooks-install: ## Install git hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

hooks-update: ## Update git hooks
	pre-commit autoupdate

hooks-run: ## Run all git hooks manually
	pre-commit run --all-files

# Configuration
config-validate: ## Validate configuration files
	python scripts/validate_config.py

config-example: ## Generate example configuration
	python scripts/generate_config_example.py > .env.example

# Backup
backup: ## Create backup of important files
	tar -czf backup-$$(date +%Y%m%d_%H%M%S).tar.gz src/ docs/ tests/ *.yml *.toml *.md

# Quick Development Setup
setup: install-dev hooks-install ## Quick setup for new developers
	@echo "Development environment setup complete!"
	@echo "Next steps:"
	@echo "  1. Copy .env.example to .env and configure"
	@echo "  2. Run 'make test' to verify installation"
	@echo "  3. Run 'make dev' to start development server"