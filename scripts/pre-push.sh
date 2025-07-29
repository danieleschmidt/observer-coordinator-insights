#!/bin/bash
# Pre-push hook for Observer Coordinator Insights
# Runs comprehensive checks before pushing code to remote repository

set -e

echo "🚀 Running pre-push checks..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
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

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not in a git repository"
    exit 1
fi

# Get the remote name and URL
remote=$(git remote)
url=$(git config --get remote.origin.url 2>/dev/null || echo "No remote URL found")

print_status "Pre-push checks for remote: $remote"
print_status "Remote URL: $url"

# Check for unstaged changes
if ! git diff-index --quiet HEAD --; then
    print_warning "You have unstaged changes. Consider committing them before pushing."
fi

# Check for untracked files
if [ -n "$(git ls-files --others --exclude-standard)" ]; then
    print_warning "You have untracked files. Consider adding them to git or .gitignore."
fi

# Run linting
print_status "Running code linting..."
if command -v ruff &> /dev/null; then
    if ! ruff check src/ tests/; then
        print_error "Linting failed. Fix the issues before pushing."
        exit 1
    fi
    print_success "Linting passed"
else
    print_warning "Ruff not found. Skipping linting."
fi

# Run type checking
print_status "Running type checking..."
if command -v mypy &> /dev/null; then
    if ! mypy src/; then
        print_error "Type checking failed. Fix the issues before pushing."
        exit 1
    fi
    print_success "Type checking passed"
else
    print_warning "MyPy not found. Skipping type checking."
fi

# Run security checks
print_status "Running security checks..."
if command -v bandit &> /dev/null; then
    if ! bandit -r src/ -f json -o bandit-report.json; then
        print_error "Security check failed. Review the issues before pushing."
        exit 1
    fi
    print_success "Security checks passed"
else
    print_warning "Bandit not found. Skipping security checks."
fi

# Run tests
print_status "Running tests..."
if command -v pytest &> /dev/null; then
    if ! pytest tests/ -x --tb=short; then
        print_error "Tests failed. Fix the failing tests before pushing."
        exit 1
    fi
    print_success "All tests passed"
else
    print_warning "Pytest not found. Skipping tests."
fi

# Check for secrets in code
print_status "Checking for potential secrets..."
if git grep -E "(password|secret|key|token)\s*=\s*['\"][^'\"]{8,}" -- '*.py' '*.yml' '*.yaml' '*.json' | grep -v -E "(test|example|sample)" | head -5; then
    print_warning "Potential secrets found in code. Please review:"
    git grep -E "(password|secret|key|token)\s*=\s*['\"][^'\"]{8,}" -- '*.py' '*.yml' '*.yaml' '*.json' | grep -v -E "(test|example|sample)" | head -5
    echo "If these are not real secrets, you can ignore this warning."
    echo "If they are real secrets, please remove them and use environment variables instead."
fi

# Check branch naming convention
current_branch=$(git symbolic-ref --short HEAD)
if [[ ! $current_branch =~ ^(main|develop|feature/|bugfix/|hotfix/|release/|terragon/) ]]; then
    print_warning "Branch name '$current_branch' doesn't follow naming conventions."
    print_warning "Consider using: feature/, bugfix/, hotfix/, release/, or terragon/ prefixes."
fi

# Check commit message format of recent commits
print_status "Checking recent commit messages..."
recent_commits=$(git log --oneline -5 --pretty=format:"%s")
while IFS= read -r commit_msg; do
    if [[ ! $commit_msg =~ ^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert|security|deps)(\(.+\))?: ]]; then
        print_warning "Commit message doesn't follow conventional commit format: '$commit_msg'"
        print_warning "Consider using format: type(scope): description"
    fi
done <<< "$recent_commits"

# Check for large files
print_status "Checking for large files..."
large_files=$(git ls-files | xargs ls -la | awk '$5 > 1048576 { print $9, $5 }' | head -5)
if [ -n "$large_files" ]; then
    print_warning "Large files detected (>1MB):"
    echo "$large_files"
    print_warning "Consider using Git LFS for large files."
fi

# Check for TODO/FIXME comments
print_status "Checking for TODO/FIXME comments..."
todo_count=$(git grep -E "(TODO|FIXME|XXX|HACK)" -- '*.py' | wc -l || echo "0")
if [ "$todo_count" -gt 0 ]; then
    print_warning "Found $todo_count TODO/FIXME comments. Consider addressing them:"
    git grep -E "(TODO|FIXME|XXX|HACK)" -- '*.py' | head -5 || true
fi

# Final check - ensure all files are properly formatted
print_status "Checking code formatting..."
if command -v ruff &> /dev/null; then
    if ! ruff format --check src/ tests/; then
        print_error "Code is not properly formatted. Run 'ruff format src/ tests/' to fix."
        exit 1
    fi
    print_success "Code formatting is correct"
fi

# Summary
print_success "All pre-push checks completed successfully! 🎉"
print_status "You can now push your changes with confidence."

# Optional: Show what will be pushed
print_status "Changes to be pushed:"
git log --oneline origin/$(git symbolic-ref --short HEAD)..HEAD || git log --oneline -5

echo ""
print_success "Pre-push validation complete. Proceeding with push..."