#!/bin/bash
set -euo pipefail

# Development Environment Setup Script for Observer Coordinator Insights
# This script automates the setup of a complete development environment

echo "🚀 Setting up Observer Coordinator Insights development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.9+ is available
check_python() {
    log_info "Checking Python version..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        REQUIRED_VERSION="3.9"
        
        if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
            log_success "Python $PYTHON_VERSION detected"
            PYTHON_CMD="python3"
        else
            log_error "Python $PYTHON_VERSION is too old. Python $REQUIRED_VERSION+ is required."
            exit 1
        fi
    else
        log_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
}

# Create virtual environment
setup_venv() {
    log_info "Setting up virtual environment..."
    
    if [ -d "venv" ]; then
        log_warning "Virtual environment already exists. Removing old one..."
        rm -rf venv
    fi
    
    $PYTHON_CMD -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    log_success "Virtual environment created and activated"
}

# Install dependencies
install_dependencies() {
    log_info "Installing project dependencies..."
    
    # Install main dependencies
    pip install -e .
    
    # Install development dependencies
    pip install -e .[dev,testing,docs]
    
    log_success "Dependencies installed"
}

# Setup pre-commit hooks
setup_pre_commit() {
    log_info "Setting up pre-commit hooks..."
    
    if command -v pre-commit &> /dev/null; then
        pre-commit install
        pre-commit install --hook-type commit-msg
        log_success "Pre-commit hooks installed"
    else
        log_warning "pre-commit not found. Installing..."
        pip install pre-commit
        pre-commit install
        pre-commit install --hook-type commit-msg
        log_success "Pre-commit installed and configured"
    fi
}

# Create necessary directories
create_directories() {
    log_info "Creating project directories..."
    
    mkdir -p {data,output,reports,logs,temp,cache}
    mkdir -p docs/{api,examples,tutorials}
    mkdir -p tests/{fixtures,data}
    
    # Create .env.example if it doesn't exist
    if [ ! -f ".env.example" ]; then
        cat > .env.example << 'EOF'
# Environment Configuration for Observer Coordinator Insights

# Development settings
DEBUG=true
LOG_LEVEL=DEBUG

# Data settings
DATA_RETENTION_DAYS=180
ENABLE_ENCRYPTION=true

# API settings (if applicable in future)
API_HOST=localhost
API_PORT=8000

# Database settings (if applicable in future)
# DATABASE_URL=sqlite:///observer_insights.db

# Security settings
SECRET_KEY=your-secret-key-here
ENCRYPTION_KEY=your-encryption-key-here

# External service settings (placeholder)
# INSIGHTS_API_URL=https://api.insights.com
# INSIGHTS_API_KEY=your-api-key-here

# Testing settings
TEST_DATA_PATH=tests/fixtures
ENABLE_TEST_LOGGING=false
EOF
        log_success "Created .env.example template"
    fi
    
    log_success "Project directories created"
}

# Run initial tests to verify setup
verify_setup() {
    log_info "Verifying installation..."
    
    # Check if we can import the main module
    if $PYTHON_CMD -c "import src.main; print('✓ Main module imports successfully')" 2>/dev/null; then
        log_success "Main module import verification passed"
    else
        log_error "Main module import failed"
        exit 1
    fi
    
    # Run a quick linting check
    if command -v ruff &> /dev/null; then
        log_info "Running code quality checks..."
        ruff check src/ --select=E9,F63,F7,F82 --quiet && log_success "Basic code quality check passed"
    else
        log_warning "Ruff not available for code quality check"
    fi
    
    # Check if tests can be discovered
    if $PYTHON_CMD -m pytest --collect-only tests/ &> /dev/null; then
        log_success "Test discovery successful"
    else
        log_warning "Test discovery failed - this is normal for new projects"
    fi
}

# Display completion message
completion_message() {
    echo ""
    echo "🎉 Development environment setup complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Activate the virtual environment: source venv/bin/activate"
    echo "  2. Copy .env.example to .env and customize: cp .env.example .env"
    echo "  3. Run tests: python -m pytest tests/"
    echo "  4. Start developing!"
    echo ""
    echo "Available commands:"
    echo "  • Run tests: python -m pytest tests/"
    echo "  • Run linting: ruff check src/ tests/"
    echo "  • Run formatting: ruff format src/ tests/"
    echo "  • Run type checking: mypy src/"
    echo "  • Run security scan: bandit -r src/"
    echo "  • Run all quality checks: make lint"
    echo ""
    echo "Documentation:"
    echo "  • Architecture: docs/ARCHITECTURE.md"
    echo "  • Development: docs/DEVELOPMENT.md"
    echo "  • Contributing: CONTRIBUTING.md"
    echo ""
}

# Main execution
main() {
    echo "Observer Coordinator Insights - Development Environment Setup"
    echo "==========================================================="
    echo ""
    
    check_python
    setup_venv
    install_dependencies
    setup_pre_commit
    create_directories
    verify_setup
    completion_message
}

# Run main function
main "$@"