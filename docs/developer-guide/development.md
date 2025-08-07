# Development Setup Guide

This guide provides comprehensive instructions for setting up a development environment for Observer Coordinator Insights, including local development, testing procedures, debugging tools, and contribution workflows.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [IDE Configuration](#ide-configuration)
4. [Database Development Setup](#database-development-setup)
5. [Testing Framework](#testing-framework)
6. [Debugging & Profiling](#debugging--profiling)
7. [Code Quality Tools](#code-quality-tools)
8. [Development Workflows](#development-workflows)
9. [Performance Testing](#performance-testing)
10. [Contribution Guidelines](#contribution-guidelines)

## Prerequisites

### System Requirements
- **Python**: 3.11 or higher
- **Git**: Latest version
- **Docker**: For containerized development (optional but recommended)
- **PostgreSQL**: 13+ (for database development)
- **Redis**: 6+ (for caching features)
- **Node.js**: 18+ (for frontend development tools)

### Recommended Tools
- **IDE**: VS Code, PyCharm, or Vim/Neovim
- **Terminal**: iTerm2 (macOS), Windows Terminal, or any modern terminal
- **Git Client**: Command line or GUI tool like GitKraken
- **Database Client**: pgAdmin, DBeaver, or TablePlus
- **API Client**: Postman, Insomnia, or httpie

### Platform-Specific Setup

#### macOS
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required tools
brew install python@3.11 git postgresql redis node
brew install --cask docker
```

#### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install Python 3.11
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Install other dependencies
sudo apt install git postgresql postgresql-contrib redis-server
sudo apt install build-essential libpq-dev

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install nodejs
```

#### Windows
```powershell
# Install using Chocolatey (recommended)
# First install Chocolatey: https://chocolatey.org/install

choco install python311 git postgresql redis-64 nodejs docker-desktop

# Or use Windows Subsystem for Linux (WSL2) with Ubuntu
```

## Local Development Setup

### Clone and Setup Repository

```bash
# Clone the repository
git clone https://github.com/terragon-labs/observer-coordinator-insights.git
cd observer-coordinator-insights

# Create and activate virtual environment
python3.11 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Upgrade pip and install development dependencies
pip install --upgrade pip
pip install -e .
pip install -r requirements-dev.txt
```

### Environment Configuration

Create a `.env` file in the root directory:

```bash
# Development environment variables
ENVIRONMENT=development
DEBUG=true

# Database configuration
DATABASE_URL=sqlite:///dev_insights.db
# For PostgreSQL development:
# DATABASE_URL=postgresql://insights_dev:dev_password@localhost:5432/insights_dev

# Redis configuration (optional for development)
REDIS_URL=redis://localhost:6379/0

# Security settings (development)
SECRET_KEY=development-secret-key-change-in-production
REQUIRE_AUTH=false
SECURE_MODE=false

# API configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Logging
LOG_LEVEL=DEBUG
LOG_FORMAT=detailed

# Testing
TEST_DATABASE_URL=sqlite:///test_insights.db
```

### Development Dependencies

The `requirements-dev.txt` includes:

```txt
# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
pytest-xdist>=3.0.0
httpx>=0.24.0
factory-boy>=3.2.0

# Code Quality
ruff>=0.0.280
black>=23.0.0
isort>=5.12.0
mypy>=1.4.0
pre-commit>=3.3.0

# Documentation
mkdocs>=1.5.0
mkdocs-material>=9.0.0
mkdocs-mermaid2-plugin>=1.0.0

# Development Tools
ipython>=8.0.0
jupyter>=1.0.0
rich>=13.0.0
watchdog>=3.0.0

# Profiling & Debugging  
py-spy>=0.3.0
memory_profiler>=0.60.0
line_profiler>=4.0.0
pdb-attach>=3.2.0

# Database Development
alembic>=1.11.0
sqlalchemy-utils>=0.41.0
```

### Initialize Development Database

```bash
# Initialize SQLite database (default)
python scripts/init_database.py --mode development

# Or for PostgreSQL development
createdb insights_dev
python scripts/init_database.py --mode development --database postgresql://insights_dev:dev_password@localhost:5432/insights_dev

# Run database migrations
alembic upgrade head

# Load sample data (optional)
python scripts/load_sample_data.py
```

### Verify Installation

```bash
# Run quality gates to verify setup
python scripts/run_quality_gates.py

# Start development server
python src/main.py --development

# Or start API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Test basic functionality
curl http://localhost:8000/api/health
```

## IDE Configuration

### Visual Studio Code Setup

Install recommended extensions by creating `.vscode/extensions.json`:

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.pylint",
    "ms-python.black-formatter",
    "ms-python.isort",
    "charliermarsh.ruff",
    "ms-python.mypy-type-checker",
    "ms-toolsai.jupyter",
    "eamodio.gitlens",
    "ms-vscode.test-adapter-converter",
    "ms-python.pytest",
    "redhat.vscode-yaml",
    "yzhang.markdown-all-in-one",
    "shd101wyy.markdown-preview-enhanced"
  ]
}
```

Configure VS Code settings in `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "88"],
  "python.sortImports.args": ["--profile", "black"],
  "python.testing.pytestEnabled": true,
  "python.testing.pytestPath": "./venv/bin/pytest",
  "python.testing.pytestArgs": ["tests/"],
  "python.testing.autoTestDiscoverOnSaveEnabled": true,
  
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    },
    "editor.rulers": [88],
    "editor.tabSize": 4
  },
  
  "[json]": {
    "editor.formatOnSave": true
  },
  
  "[yaml]": {
    "editor.formatOnSave": true,
    "editor.tabSize": 2
  },
  
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".pytest_cache": true,
    ".coverage": true,
    "htmlcov": true
  },
  
  "python.analysis.extraPaths": ["./src"],
  "python.envFile": "${workspaceFolder}/.env"
}
```

Create debug configuration in `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: API Server",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "src.api.main:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000"
      ],
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.env"
    },
    {
      "name": "Python: Main Application",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/src/main.py",
      "args": ["tests/fixtures/sample_data.csv", "--clusters", "4"],
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.env"
    },
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.env"
    },
    {
      "name": "Python: Pytest Current File",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["${file}", "-v"],
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.env"
    }
  ]
}
```

### PyCharm Setup

1. **Open Project**: File → Open → Select project directory
2. **Configure Interpreter**: Settings → Python Interpreter → Add → Existing Environment → Point to `venv/bin/python`
3. **Configure Code Style**: Settings → Code Style → Python → Set line length to 88
4. **Enable Plugins**: Enable Python, Git, Docker, Database tools
5. **Configure Test Runner**: Settings → Tools → Python Integrated Tools → Set pytest as default

### Vim/Neovim Setup

Example configuration for Neovim with modern Python development:

```lua
-- ~/.config/nvim/init.lua
-- Python development configuration

-- Install packer.nvim for plugin management
-- Plugin setup
require('packer').startup(function(use)
  use 'wbthomason/packer.nvim'
  
  -- LSP and completion
  use 'neovim/nvim-lspconfig'
  use 'hrsh7th/nvim-cmp'
  use 'hrsh7th/cmp-nvim-lsp'
  use 'hrsh7th/cmp-buffer'
  use 'hrsh7th/cmp-path'
  
  -- Python-specific
  use 'psf/black'
  use 'fisadev/vim-isort'
  use 'nvim-treesitter/nvim-treesitter'
  
  -- Git integration
  use 'tpope/vim-fugitive'
  use 'lewis6991/gitsigns.nvim'
  
  -- File explorer
  use 'nvim-tree/nvim-tree.lua'
  
  -- Testing
  use 'vim-test/vim-test'
end)

-- LSP configuration for Python
require'lspconfig'.pylsp.setup{
  settings = {
    pylsp = {
      plugins = {
        pycodestyle = {enabled = false},
        pyflakes = {enabled = false},
        pylint = {enabled = false},
        ruff = {enabled = true},
        black = {enabled = true},
      }
    }
  }
}

-- Key mappings for development
vim.keymap.set('n', '<leader>t', ':TestNearest<CR>')
vim.keymap.set('n', '<leader>T', ':TestFile<CR>')
vim.keymap.set('n', '<leader>r', ':TestLast<CR>')
```

## Database Development Setup

### PostgreSQL Development

```bash
# Start PostgreSQL service
sudo systemctl start postgresql  # Linux
brew services start postgresql   # macOS

# Create development database and user
sudo -u postgres psql << EOF
CREATE DATABASE insights_dev;
CREATE USER insights_dev WITH ENCRYPTED PASSWORD 'dev_password';
GRANT ALL PRIVILEGES ON DATABASE insights_dev TO insights_dev;
ALTER USER insights_dev CREATEDB;  -- For running tests
EOF

# Update environment variables
echo "DATABASE_URL=postgresql://insights_dev:dev_password@localhost:5432/insights_dev" >> .env
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Description of changes"

# Review generated migration file in migrations/versions/
# Edit if necessary

# Apply migration
alembic upgrade head

# Downgrade if needed
alembic downgrade -1
```

### Database Debugging

```python
# Enable SQL logging in development
# In src/database/connection.py

engine = create_engine(
    database_url,
    echo=True,  # Log all SQL statements
    echo_pool=True,  # Log connection pool events
)
```

## Testing Framework

### Test Structure

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures
├── unit/                      # Unit tests
│   ├── __init__.py
│   ├── test_clustering.py
│   ├── test_validation.py
│   └── test_agents.py
├── integration/               # Integration tests
│   ├── __init__.py
│   ├── test_api_endpoints.py
│   ├── test_database.py
│   └── test_full_pipeline.py
├── performance/               # Performance tests
│   ├── __init__.py
│   ├── test_clustering_benchmarks.py
│   └── test_load_testing.py
├── security/                  # Security tests
│   ├── __init__.py
│   ├── test_authentication.py
│   └── test_data_protection.py
└── fixtures/                  # Test data
    ├── sample_data.csv
    └── test_config.yml
```

### Test Configuration

`conftest.py` with shared fixtures:

```python
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import numpy as np

from src.database.models.base import Base
from src.database.connection import get_database
from src.config import get_settings

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_database():
    """Create test database for the session."""
    # Create temporary database file
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db_url = f"sqlite:///{temp_db.name}"
    
    # Create engine and tables
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    
    yield db_url
    
    # Cleanup
    temp_db.close()
    Path(temp_db.name).unlink(missing_ok=True)

@pytest.fixture
async def db_session(test_database):
    """Create database session for each test."""
    engine = create_engine(test_database)
    SessionLocal = sessionmaker(bind=engine)
    
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture
def sample_personality_data():
    """Generate sample personality data for testing."""
    np.random.seed(42)  # Reproducible results
    
    n_employees = 100
    data = []
    
    for i in range(n_employees):
        # Generate realistic personality profiles
        # Each profile sums to approximately 100
        base_energies = np.random.dirichlet([1, 1, 1, 1]) * 100
        
        # Add some noise
        noise = np.random.normal(0, 2, 4)
        energies = np.clip(base_energies + noise, 0, 100)
        
        data.append({
            'employee_id': f'EMP{i:03d}',
            'red_energy': energies[0],
            'blue_energy': energies[1],
            'green_energy': energies[2],
            'yellow_energy': energies[3]
        })
    
    return pd.DataFrame(data)

@pytest.fixture
def temp_csv_file(sample_personality_data):
    """Create temporary CSV file with sample data."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
    sample_personality_data.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    yield temp_file.name
    
    Path(temp_file.name).unlink(missing_ok=True)

@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        'environment': 'testing',
        'database': {'url': 'sqlite:///test.db'},
        'clustering': {
            'default_method': 'esn',
            'default_clusters': 4
        },
        'security': {
            'require_auth': False,
            'secure_mode': False
        }
    }

@pytest.fixture
async def api_client():
    """Create FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.api.main import app
    
    return TestClient(app)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html --cov-report=term

# Run specific test types
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests only
pytest tests/performance/             # Performance tests only

# Run tests with verbose output
pytest -v

# Run tests in parallel
pytest -n auto

# Run specific test
pytest tests/unit/test_clustering.py::test_esn_clustering

# Run tests matching pattern
pytest -k "clustering"

# Run tests and stop on first failure
pytest -x

# Run tests with debugging
pytest --pdb

# Generate coverage report
pytest --cov=src --cov-report=html
# Open htmlcov/index.html to view coverage report
```

### Example Test Cases

#### Unit Test Example
```python
# tests/unit/test_clustering.py
import pytest
import numpy as np
from src.insights_clustering.neuromorphic_clustering import NeuromorphicClusterer

class TestNeuromorphicClusterer:
    
    def test_esn_clusterer_initialization(self):
        """Test ESN clusterer initialization."""
        clusterer = NeuromorphicClusterer(method="esn", n_clusters=4)
        
        assert clusterer.method == "esn"
        assert clusterer.n_clusters == 4
        assert clusterer.clusterer is not None
    
    @pytest.mark.asyncio
    async def test_clustering_with_sample_data(self, sample_personality_data):
        """Test clustering with sample data."""
        clusterer = NeuromorphicClusterer(method="esn", n_clusters=3)
        
        # Convert DataFrame to numpy array
        data = sample_personality_data[['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']].values
        
        # Perform clustering
        result = await clusterer.fit_async(data)
        
        # Verify results
        assert 'cluster_labels' in result
        assert len(result['cluster_labels']) == len(data)
        assert len(np.unique(result['cluster_labels'])) <= 3
        assert 'metrics' in result
        assert result['metrics']['silhouette_score'] >= -1
        assert result['metrics']['silhouette_score'] <= 1
    
    def test_invalid_parameters(self):
        """Test clusterer with invalid parameters."""
        with pytest.raises(ValueError):
            NeuromorphicClusterer(method="invalid_method")
        
        with pytest.raises(ValueError):
            NeuromorphicClusterer(n_clusters=0)
```

#### Integration Test Example
```python
# tests/integration/test_api_endpoints.py
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

class TestAPIEndpoints:
    
    def test_health_endpoint(self, api_client):
        """Test health check endpoint."""
        response = api_client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_upload_endpoint_with_valid_data(self, api_client, temp_csv_file):
        """Test file upload with valid data."""
        with open(temp_csv_file, 'rb') as f:
            files = {'file': ('test_data.csv', f, 'text/csv')}
            data = {'n_clusters': 4, 'method': 'esn'}
            
            response = api_client.post("/api/analytics/upload", files=files, data=data)
        
        assert response.status_code == 200
        result = response.json()
        assert "job_id" in result
        assert result["status"] == "processing"
    
    def test_upload_endpoint_with_invalid_data(self, api_client):
        """Test file upload with invalid data."""
        # Create invalid CSV content
        invalid_csv = "invalid,csv,content\n1,2,3"
        files = {'file': ('invalid.csv', invalid_csv, 'text/csv')}
        
        response = api_client.post("/api/analytics/upload", files=files)
        
        assert response.status_code == 400
        error = response.json()
        assert "error" in error
```

## Debugging & Profiling

### Debugging Tools

#### Python Debugger (pdb)
```python
# Insert breakpoint in code
import pdb; pdb.set_trace()

# Or use built-in breakpoint() function (Python 3.7+)
breakpoint()

# Remote debugging with pdb-attach
from pdb_attach import listen
listen(50000)  # Then connect with: pdb-attach 50000
```

#### Visual Debugging
```bash
# Install debugging tools
pip install ipdb pudb

# Use ipdb for enhanced debugging
import ipdb; ipdb.set_trace()

# Use pudb for full-screen debugging
import pudb; pudb.set_trace()
```

#### Async Debugging
```python
# For async functions
import asyncio
async def debug_async():
    import pdb; pdb.set_trace()
    await some_async_function()

# Or use adb for async debugging
pip install aiomonitor
from aiomonitor import start_monitor
start_monitor(loop, host='127.0.0.1', port=50101)
```

### Performance Profiling

#### CPU Profiling with py-spy
```bash
# Install py-spy
pip install py-spy

# Profile running application
py-spy record -o profile.svg -- python src/main.py data.csv

# Live profiling
py-spy top --pid <PID>

# Profile pytest runs
py-spy record -o test_profile.svg -- python -m pytest tests/performance/
```

#### Memory Profiling
```bash
# Install memory profiler
pip install memory-profiler psutil

# Profile specific function
@profile
def memory_intensive_function():
    # Function code here
    pass

# Run with memory profiler
python -m memory_profiler src/clustering.py

# Line-by-line memory usage
mprof run src/main.py data.csv
mprof plot
```

#### Line Profiling
```bash
# Install line profiler
pip install line_profiler

# Add @profile decorator to functions you want to profile
@profile
def slow_function():
    # Function code here
    pass

# Run line profiler
kernprof -l -v src/clustering.py
```

#### Custom Performance Monitoring
```python
# src/monitoring/dev_profiler.py
import time
import functools
import cProfile
import pstats
from typing import Callable, Any

class DevProfiler:
    """Development profiler for performance monitoring."""
    
    def __init__(self):
        self.timing_data = {}
        self.profiler = None
    
    def time_function(self, func_name: str = None):
        """Decorator to time function execution."""
        def decorator(func: Callable) -> Callable:
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    
                    if name not in self.timing_data:
                        self.timing_data[name] = []
                    self.timing_data[name].append(execution_time)
                    
                    print(f"⏱️ {name}: {execution_time:.4f}s")
            
            return wrapper
        return decorator
    
    def start_profiling(self):
        """Start CPU profiling."""
        self.profiler = cProfile.Profile()
        self.profiler.enable()
    
    def stop_profiling(self, output_file: str = "profile_results.prof"):
        """Stop profiling and save results."""
        if self.profiler:
            self.profiler.disable()
            self.profiler.dump_stats(output_file)
            
            # Print top functions
            stats = pstats.Stats(output_file)
            stats.sort_stats('cumulative')
            stats.print_stats(20)
    
    def get_timing_summary(self) -> dict:
        """Get summary of timing data."""
        summary = {}
        for func_name, times in self.timing_data.items():
            summary[func_name] = {
                'count': len(times),
                'total': sum(times),
                'avg': sum(times) / len(times),
                'min': min(times),
                'max': max(times)
            }
        return summary

# Usage example
profiler = DevProfiler()

@profiler.time_function()
async def example_clustering_function():
    # Your clustering code here
    pass
```

## Code Quality Tools

### Pre-commit Hooks

Install and configure pre-commit:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hook scripts
pre-commit install

# Run against all files (optional)
pre-commit run --all-files
```

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-json
      - id: check-merge-conflict
      - id: debug-statements
  
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.11
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
  
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: 'v0.0.280'
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.4.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        exclude: ^tests/
```

### Code Formatting

Configure `pyproject.toml` for code quality tools:

```toml
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
line_length = 88
known_first_party = ["src", "tests"]

[tool.ruff]
target-version = "py311"
line-length = 88
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade
]
ignore = [
    "E501",  # line too long, handled by black
    "B008",  # do not perform function calls in argument defaults
    "C901",  # too complex
]
exclude = [
    ".bzr",
    ".direnv",
    ".eggs",
    ".git",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pants.d",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pypackages__",
    "_build",
    "buck-out",
    "build",
    "dist",
    "node_modules",
    "venv",
]

[tool.ruff.per-file-ignores]
"tests/**/*" = ["S101"]  # Use of assert detected

[tool.mypy]
python_version = "3.11"
check_untyped_defs = true
ignore_missing_imports = true
warn_unused_ignores = true
warn_redundant_casts = true
warn_unused_configs = true

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --strict-markers --strict-config"
testpaths = ["tests"]
filterwarnings = [
    "ignore::UserWarning",
    "ignore::DeprecationWarning",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "security: marks tests as security tests",
]

[tool.coverage.run]
source = ["src"]
omit = [
    "src/migrations/*",
    "tests/*",
    "venv/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
```

### Quality Gates Script

Create `scripts/quality_gates.py`:

```python
#!/usr/bin/env python3
"""
Quality gates script for Observer Coordinator Insights
Runs all code quality checks and tests
"""

import subprocess
import sys
import os
from pathlib import Path

class QualityGates:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.failed_checks = []
    
    def run_command(self, command: str, description: str) -> bool:
        """Run a command and return success/failure."""
        print(f"🔍 {description}...")
        
        try:
            result = subprocess.run(
                command.split(),
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ {description} passed")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} failed:")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            self.failed_checks.append(description)
            return False
    
    def run_all_checks(self):
        """Run all quality gate checks."""
        print("🚀 Running Observer Coordinator Insights Quality Gates")
        print("=" * 60)
        
        checks = [
            ("ruff check src tests", "Code linting (ruff)"),
            ("black --check src tests", "Code formatting (black)"),
            ("isort --check-only src tests", "Import sorting (isort)"),
            ("mypy src", "Type checking (mypy)"),
            ("pytest tests/unit -v", "Unit tests"),
            ("pytest tests/integration -v", "Integration tests"),
            ("pytest tests/security -v", "Security tests"),
            ("pytest --cov=src --cov-report=term-missing --cov-fail-under=80", "Coverage check"),
        ]
        
        for command, description in checks:
            self.run_command(command, description)
        
        print("=" * 60)
        if self.failed_checks:
            print(f"❌ {len(self.failed_checks)} quality gate(s) failed:")
            for check in self.failed_checks:
                print(f"   • {check}")
            sys.exit(1)
        else:
            print("✅ All quality gates passed! 🎉")
    
    def fix_issues(self):
        """Attempt to auto-fix code issues."""
        print("🔧 Attempting to fix code issues...")
        
        fix_commands = [
            ("ruff --fix src tests", "Auto-fixing linting issues"),
            ("black src tests", "Auto-formatting code"),
            ("isort src tests", "Sorting imports"),
        ]
        
        for command, description in fix_commands:
            self.run_command(command, description)

if __name__ == "__main__":
    gates = QualityGates()
    
    if "--fix" in sys.argv:
        gates.fix_issues()
    
    gates.run_all_checks()
```

## Development Workflows

### Git Workflow

#### Branch Naming Convention
- `feature/description-of-feature`
- `bugfix/description-of-bug`
- `hotfix/critical-issue`
- `refactor/description-of-refactor`
- `docs/documentation-update`

#### Development Process

```bash
# 1. Create feature branch
git checkout -b feature/neuromorphic-snn-optimization

# 2. Make changes and commit frequently
git add .
git commit -m "feat: add SNN parameter optimization

- Add hyperparameter tuning for SNN
- Implement grid search optimization
- Add performance benchmarking"

# 3. Run quality gates before pushing
python scripts/quality_gates.py --fix
python scripts/quality_gates.py

# 4. Push branch
git push -u origin feature/neuromorphic-snn-optimization

# 5. Create pull request
# Use GitHub CLI or web interface
gh pr create --title "Add SNN optimization" --body "Detailed description..."

# 6. After review, merge and cleanup
git checkout main
git pull origin main
git branch -d feature/neuromorphic-snn-optimization
```

#### Commit Message Convention

Follow conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

Examples:
```bash
git commit -m "feat(clustering): add liquid state machine algorithm

Implement LSM-based clustering with 3D neuron topology
and synaptic plasticity for improved pattern recognition."

git commit -m "fix(api): resolve memory leak in file upload handler

The upload handler was not properly closing file handles,
causing memory leaks during large file uploads."

git commit -m "docs(api): add comprehensive API examples

- Add code examples for all endpoints
- Include error handling patterns
- Add performance optimization tips"
```

### Development Scripts

Create helpful development scripts in `scripts/`:

#### `scripts/dev_setup.sh`
```bash
#!/bin/bash
# Development environment setup script

set -e

echo "🚀 Setting up Observer Coordinator Insights development environment"

# Check Python version
python_version=$(python3.11 --version 2>/dev/null | cut -d' ' -f2)
if [[ -z "$python_version" ]]; then
    echo "❌ Python 3.11 not found. Please install Python 3.11 or higher."
    exit 1
fi
echo "✅ Found Python $python_version"

# Create virtual environment
if [[ ! -d "venv" ]]; then
    echo "📦 Creating virtual environment..."
    python3.11 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -e .
pip install -r requirements-dev.txt

# Setup pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install

# Initialize database
echo "🗄️ Initializing development database..."
python scripts/init_database.py --mode development

# Run quality gates to verify setup
echo "🔍 Running quality gates to verify setup..."
python scripts/quality_gates.py

echo "✅ Development environment setup complete!"
echo "🎯 Next steps:"
echo "   1. Activate virtual environment: source venv/bin/activate"
echo "   2. Start development server: uvicorn src.api.main:app --reload"
echo "   3. Open http://localhost:8000/docs for API documentation"
```

#### `scripts/test_runner.py`
```python
#!/usr/bin/env python3
"""
Enhanced test runner with parallel execution and reporting
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

def run_tests(test_type="all", parallel=True, coverage=True, verbose=True):
    """Run tests with specified options."""
    
    cmd = ["python", "-m", "pytest"]
    
    # Test selection
    if test_type == "unit":
        cmd.append("tests/unit")
    elif test_type == "integration":
        cmd.append("tests/integration")
    elif test_type == "security":
        cmd.append("tests/security")
    elif test_type == "performance":
        cmd.append("tests/performance")
    else:
        cmd.append("tests")
    
    # Options
    if parallel:
        cmd.extend(["-n", "auto"])
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    if verbose:
        cmd.append("-v")
    
    # Additional options
    cmd.extend([
        "--tb=short",
        "--strict-markers",
        "--strict-config"
    ])
    
    print(f"🧪 Running tests: {' '.join(cmd)}")
    start_time = time.time()
    
    result = subprocess.run(cmd)
    
    end_time = time.time()
    duration = end_time - start_time
    
    if result.returncode == 0:
        print(f"✅ Tests passed in {duration:.2f}s")
        if coverage:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print(f"❌ Tests failed after {duration:.2f}s")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Observer Coordinator Insights tests")
    parser.add_argument("--type", choices=["all", "unit", "integration", "security", "performance"], 
                       default="all", help="Type of tests to run")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel execution")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    run_tests(
        test_type=args.type,
        parallel=not args.no_parallel,
        coverage=not args.no_coverage,
        verbose=not args.quiet
    )
```

This comprehensive development setup guide provides everything needed to contribute effectively to Observer Coordinator Insights while maintaining high code quality standards.