# Development Setup

## Prerequisites

* Python 3.8+
* Node.js 16+ (for tooling)
* Git
* Docker (optional)

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd <repository-name>

# Install dependencies
pip install -r requirements-dev.txt
npm install  # For pre-commit hooks

# Setup pre-commit
pre-commit install

# Run tests
make test

# Start development
make dev
```

## Project Structure

* `src/` - Core application code
* `tests/` - Test suites (unit, integration, e2e)
* `docs/` - Documentation and ADRs
* `requirements*.txt` - Python dependencies

## Development Resources

* Architecture: `../ARCHITECTURE.md`
* Contributing: `../CONTRIBUTING.md`
* [Python Development Guide](https://docs.python.org/3/tutorial/)