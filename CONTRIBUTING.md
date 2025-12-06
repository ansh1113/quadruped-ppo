# Contributing to Quadruped PPO

Thank you for your interest in contributing to the Quadruped PPO project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/quadruped-ppo.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `pytest tests/`
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Check code style
black .
flake8 .
mypy src/
```

## Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting
- Maximum line length: 100 characters
- Add docstrings to all public functions and classes
- Add type hints where applicable

## Testing

- Write tests for new features
- Maintain test coverage above 80%
- Run full test suite before submitting PR
- Test on multiple Python versions (3.8, 3.9, 3.10)

## Documentation

- Update README.md if adding new features
- Add docstrings following Google style
- Update CHANGELOG.md
- Add examples for new functionality

## Pull Request Process

1. Update documentation
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Request review from maintainers
6. Address review feedback
7. Merge after approval

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Collaborate openly

## Questions?

Open an issue or reach out to the maintainers.

Thank you for contributing! 🚀
