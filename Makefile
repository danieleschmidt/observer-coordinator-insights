# Observer Coordinator Insights - Build and Development Automation
.PHONY: help install test lint format type-check security-scan build clean dev prod docker-build docker-run docs

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
PROJECT_NAME := observer-coordinator-insights
DOCKER_IMAGE := $(PROJECT_NAME)
DOCKER_TAG := latest
TEST_ARGS := tests/ -v
COVERAGE_THRESHOLD := 80

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)$(PROJECT_NAME) - Development Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(GREEN)<target>$(NC)\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# Development Setup
install: ## Install dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .[dev]
	pre-commit install

install-prod: ## Install production dependencies only
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .

# Code Quality
lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	ruff check src/ tests/
	@echo "$(GREEN)Linting completed!$(NC)"

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	ruff format src/ tests/
	@echo "$(GREEN)Code formatting completed!$(NC)"

type-check: ## Run type checking
	@echo "$(BLUE)Running type checks...$(NC)"
	mypy src/
	@echo "$(GREEN)Type checking completed!$(NC)"

security-scan: ## Run security scanning
	@echo "$(BLUE)Running security scan...$(NC)"
	bandit -r src/ -f json -o security-report.json || true
	bandit -r src/ -f console
	@echo "$(GREEN)Security scan completed!$(NC)"

# Testing
test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTHON) -m pytest $(TEST_ARGS)

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTHON) -m pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTHON) -m pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests
	@echo "$(BLUE)Running end-to-end tests...$(NC)"
	$(PYTHON) -m pytest tests/e2e/ -v

test-performance: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	$(PYTHON) -m pytest tests/performance/ -v -m performance

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTHON) -m pytest --cov=src --cov-report=html --cov-report=term --cov-fail-under=$(COVERAGE_THRESHOLD)
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

test-all: lint type-check security-scan test-coverage ## Run all quality checks and tests

# Building and Packaging
build: ## Build package
	@echo "$(BLUE)Building package...$(NC)"
	$(PYTHON) -m build
	@echo "$(GREEN)Package built successfully!$(NC)"

build-wheel: ## Build wheel package only
	@echo "$(BLUE)Building wheel package...$(NC)"
	$(PYTHON) -m build --wheel

build-sdist: ## Build source distribution only
	@echo "$(BLUE)Building source distribution...$(NC)"
	$(PYTHON) -m build --sdist

# Docker Operations
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "$(GREEN)Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"

docker-build-dev: ## Build development Docker image
	@echo "$(BLUE)Building development Docker image...$(NC)"
	docker build --target development -t $(DOCKER_IMAGE):dev .
	@echo "$(GREEN)Development Docker image built: $(DOCKER_IMAGE):dev$(NC)"

docker-run: ## Run application in Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run --rm -it \
		-v $(PWD)/data:/app/data:ro \
		-v $(PWD)/output:/app/output \
		-p 8080:8080 \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-compose-up: ## Start all services with docker-compose
	@echo "$(BLUE)Starting services with docker-compose...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Services started!$(NC)"

docker-compose-dev: ## Start development environment
	@echo "$(BLUE)Starting development environment...$(NC)"
	docker-compose --profile dev up -d
	@echo "$(GREEN)Development environment started!$(NC)"

docker-compose-down: ## Stop all docker-compose services
	@echo "$(BLUE)Stopping docker-compose services...$(NC)"
	docker-compose down
	@echo "$(GREEN)Services stopped!$(NC)"

# Development Commands
dev: ## Start development server
	@echo "$(BLUE)Starting development server...$(NC)"
	$(PYTHON) src/main.py --dev

prod: ## Start production server
	@echo "$(BLUE)Starting production server...$(NC)"
	$(PYTHON) -m src.main

debug: ## Start application in debug mode
	@echo "$(BLUE)Starting application in debug mode...$(NC)"
	$(PYTHON) -m debugpy --listen 5678 --wait-for-client -m src.main

# Data Processing
sample-run: ## Run with sample data
	@echo "$(BLUE)Running with sample data...$(NC)"
	$(PYTHON) -m src.main --input tests/fixtures/sample_insights_data.csv --output output/

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	$(PYTHON) -m pytest tests/performance/ -v --benchmark-only

# Documentation
docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	@echo "$(YELLOW)Documentation build not yet implemented$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	@echo "$(YELLOW)Documentation serve not yet implemented$(NC)"

# Cleanup
clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf output/
	rm -rf logs/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)Cleanup completed!$(NC)"

clean-docker: ## Clean Docker images and containers
	@echo "$(BLUE)Cleaning Docker images and containers...$(NC)"
	docker system prune -f
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) 2>/dev/null || true
	docker rmi $(DOCKER_IMAGE):dev 2>/dev/null || true
	@echo "$(GREEN)Docker cleanup completed!$(NC)"

# Release Management
version: ## Show current version
	@echo "$(BLUE)Current version:$(NC)"
	@$(PYTHON) -c "import src; print(getattr(src, '__version__', 'unknown'))" 2>/dev/null || echo "Version not found"

release-check: ## Check if ready for release
	@echo "$(BLUE)Checking release readiness...$(NC)"
	@$(MAKE) test-all
	@$(MAKE) build
	@echo "$(GREEN)Release check completed!$(NC)"

# Environment Management
setup-dev: ## Set up development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	python -m venv venv
	. venv/bin/activate && $(MAKE) install
	@echo "$(GREEN)Development environment set up!$(NC)"
	@echo "$(YELLOW)Activate with: source venv/bin/activate$(NC)"

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

# CI/CD Support
ci-test: ## Run tests in CI environment
	@echo "$(BLUE)Running CI tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v --junitxml=test-results.xml --cov=src --cov-report=xml

ci-build: ## Build in CI environment
	@echo "$(BLUE)Running CI build...$(NC)"
	$(MAKE) install-prod
	$(MAKE) build
	$(MAKE) docker-build

# Monitoring and Health Checks
health-check: ## Run application health check
	@echo "$(BLUE)Running health check...$(NC)"
	$(PYTHON) -c "from src.main import main; print('Health check passed')" || echo "$(RED)Health check failed$(NC)"

metrics: ## Show application metrics
	@echo "$(BLUE)Application metrics:$(NC)"
	@echo "$(YELLOW)Metrics endpoint not yet implemented$(NC)"

# Database Operations (if applicable)
init-db: ## Initialize database
	@echo "$(BLUE)Initializing database...$(NC)"
	@echo "$(YELLOW)Database initialization not yet implemented$(NC)"

migrate-db: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	@echo "$(YELLOW)Database migrations not yet implemented$(NC)"