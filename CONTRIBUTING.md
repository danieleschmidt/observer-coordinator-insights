# Contributing Guidelines

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Set up development environment (see `docs/DEVELOPMENT.md`)
4. Make your changes with tests
5. Run linting and tests: `make test lint`
6. Submit a pull request

## Pull Request Process

* Ensure all tests pass
* Update documentation as needed
* Follow existing code style
* Write clear commit messages

## Issue Reporting

* Use GitHub Issues for bugs and feature requests
* Include reproduction steps for bugs
* Search existing issues first

## Development Resources

* [GitHub Flow](https://guides.github.com/introduction/flow/)
* [Conventional Commits](https://www.conventionalcommits.org/)
* Project architecture: `ARCHITECTURE.md`
* Setup guide: `docs/DEVELOPMENT.md`
# Contributing to Observer Coordinator Insights

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Development Setup

1. Fork and clone the repository
2. Create a development environment:
   ```bash
   make setup-dev
   source venv/bin/activate
   ```
3. Install dependencies: `make install`
4. Run tests: `make test`

## Code Standards

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write comprehensive tests
- Add docstrings for public APIs
- Run pre-commit hooks: `make pre-commit`

## Testing

- Write unit tests for new features
- Ensure integration tests pass
- Add performance tests for algorithms
- Maintain >80% code coverage

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Run the full test suite: `make test-all`
4. Update documentation if needed
5. Submit a pull request with:
   - Clear description of changes
   - Link to related issues
   - Test evidence

## Commit Messages

Use conventional commits format:
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation
- `test:` tests
- `refactor:` code refactoring
- `chore:` maintenance

## Code Review

All submissions require review. We may ask for:
- Code changes for clarity or performance
- Additional tests or documentation
- Compliance with security practices

## Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Follow our Code of Conduct

## Questions?

- Open a GitHub issue for bugs/features
- Join discussions for general questions
- Contact maintainers for security issues