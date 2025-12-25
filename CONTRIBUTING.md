# Contributing to Goal State Agent

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/goal-state-agent.git
   cd goal-state-agent
   ```
3. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

## Development Setup

Install development dependencies:

```bash
pip install -e ".[dev]"
```

This includes:
- `pytest` for testing
- `black` for code formatting
- `mypy` for type checking

## Code Style

- Use [Black](https://black.readthedocs.io/) for formatting (line length 88)
- Use type hints where possible
- Follow PEP 8 guidelines
- Write docstrings for public functions and classes

Format code before committing:

```bash
black .
```

## Running Tests

```bash
pytest
```

## Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Add tests for new functionality

4. Ensure tests pass:
   ```bash
   pytest
   ```

5. Format code:
   ```bash
   black .
   ```

6. Commit your changes:
   ```bash
   git commit -m "Add feature: description of your changes"
   ```

7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

8. Open a Pull Request

## Pull Request Guidelines

- Keep PRs focused on a single change
- Include tests for new functionality
- Update documentation as needed
- Write clear commit messages
- Reference any related issues

## Areas for Contribution

- **New environments**: Add support for other Gymnasium environments
- **Network architectures**: Implement deeper networks or different layer types
- **Goal-conditioned policies**: Extend to support multiple goal states
- **Visualization**: Add training visualization tools
- **Documentation**: Improve examples and tutorials
- **Tests**: Increase test coverage

## Questions?

Feel free to open an issue for questions or discussions about potential contributions.
