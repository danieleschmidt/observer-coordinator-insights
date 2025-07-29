#!/bin/bash
# Post-create script for devcontainer setup
# This script runs after the container is created but before the first use

set -e

echo "🚀 Setting up Observer Coordinator Insights development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[SETUP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Update system packages
print_status "Updating system packages..."
sudo apt-get update -qq

# Install additional development tools
print_status "Installing additional development tools..."
sudo apt-get install -y -qq \
    tree \
    htop \
    curl \
    wget \
    jq \
    vim \
    less \
    make \
    build-essential \
    software-properties-common

# Install Python dependencies
print_status "Installing Python dependencies..."
python -m pip install --upgrade pip

# Install the project in development mode
if [ -f "pyproject.toml" ]; then
    print_status "Installing project dependencies from pyproject.toml..."
    pip install -e ".[dev,docs,testing]"
elif [ -f "requirements.txt" ]; then
    print_status "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
    fi
fi

# Install pre-commit hooks
if [ -f ".pre-commit-config.yaml" ]; then
    print_status "Installing pre-commit hooks..."
    pre-commit install
    pre-commit install --hook-type commit-msg
    print_success "Pre-commit hooks installed"
else
    print_warning "No .pre-commit-config.yaml found, skipping pre-commit setup"
fi

# Set up git configuration if not already set
if ! git config --global user.name >/dev/null 2>&1; then
    print_status "Setting up default git configuration..."
    git config --global user.name "Dev Container User"
    git config --global user.email "dev@terragon-labs.com"
    git config --global init.defaultBranch main
    git config --global pull.rebase true
    git config --global push.autoSetupRemote true
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs data output cache reports visualizations

# Set up shell environment
print_status "Setting up shell environment..."
if [ "$SHELL" = "/bin/bash" ] || [ "$SHELL" = "/usr/bin/bash" ]; then
    # Add useful aliases to bashrc
    cat >> ~/.bashrc << 'EOF'

# Project-specific aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

# Python aliases
alias py='python'
alias pip='python -m pip'
alias pytest='python -m pytest'
alias mypy='python -m mypy'
alias ruff='python -m ruff'

# Project-specific aliases
alias test='make test'
alias lint='make lint'
alias format='make format'
alias dev='make dev'
alias logs='docker-compose logs -f'
alias up='docker-compose up -d'
alias down='docker-compose down'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git pull'
alias gd='git diff'
alias gco='git checkout'
alias gb='git branch'
alias glog='git log --oneline --graph --decorate'

EOF
fi

# Set up Python path
export PYTHONPATH="/workspaces/observer-coordinator-insights/src:$PYTHONPATH"
echo 'export PYTHONPATH="/workspaces/observer-coordinator-insights/src:$PYTHONPATH"' >> ~/.bashrc

# Install additional Python tools for development
print_status "Installing additional development tools..."
pip install --quiet \
    ipython \
    jupyter \
    jupyterlab \
    ipdb \
    rich \
    typer-cli \
    httpie \
    python-dotenv

# Verify installation
print_status "Verifying installation..."
python --version
pip --version
git --version

# Test imports
print_status "Testing key imports..."
python -c "import pytest; import mypy; import ruff; print('✅ Development tools imported successfully')" || print_warning "Some development tools may not be available"

# Set up Jupyter if available
if command -v jupyter >/dev/null 2>&1; then
    print_status "Setting up Jupyter configuration..."
    jupyter --generate-config --allow-root 2>/dev/null || true
    
    # Create jupyter config for development
    mkdir -p ~/.jupyter
    cat > ~/.jupyter/jupyter_lab_config.py << 'EOF'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.allow_origin = '*'
c.ServerApp.disable_check_xsrf = True
EOF
    print_success "Jupyter configured for development"
fi

# Create sample environment file if it doesn't exist
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    print_status "Creating development .env file from .env.example..."
    cp .env.example .env
    print_success "Development .env file created"
fi

# Run a quick health check
print_status "Running quick health check..."
if [ -f "src/main.py" ]; then
    python -c "import sys; sys.path.insert(0, 'src'); import main; print('✅ Main module imports successfully')" || print_warning "Main module import failed"
fi

# Display useful information
print_success "Development environment setup complete! 🎉"
echo ""
echo "📋 Quick Start Commands:"
echo "  make help          - Show available commands"  
echo "  make test          - Run tests"
echo "  make lint          - Run linting"
echo "  make dev           - Start development server"
echo "  make up            - Start services with docker-compose"
echo "  jupyter lab        - Start Jupyter Lab (port 8888)"
echo ""
echo "🔧 Development Tools Installed:"
echo "  ✅ Python $(python --version | cut -d' ' -f2)"
echo "  ✅ Pre-commit hooks"
echo "  ✅ Testing framework (pytest)"
echo "  ✅ Linting (ruff, mypy)"
echo "  ✅ Jupyter Lab"
echo "  ✅ Git configuration"
echo ""
echo "📁 Workspace: /workspaces/observer-coordinator-insights"
echo "🌐 Forwarded ports: 8000 (app), 8080 (health), 8888 (jupyter), 9090 (metrics)"
echo ""
print_success "Happy coding! 🚀"