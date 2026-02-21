# Contributing to TokenKeeper

Thank you for your interest in contributing to TokenKeeper! This guide will help you get started.

## Development Setup

### Prerequisites

- **Python 3.10+** (up to 3.13)
- **[Ollama](https://ollama.com/)** installed and running (for embedding tests)
- **[uv](https://docs.astral.sh/uv/)** package manager

### Clone and Install

```bash
git clone https://github.com/admin-sosys/TokenKeeper.git
cd TokenKeeper
uv sync --dev
```

### Pull the Embedding Model

```bash
ollama pull nomic-embed-text
```

This downloads the `nomic-embed-text` model used for local embeddings (768 dimensions, runs on CPU).

## Running Tests

Run the full test suite:

```bash
uv run pytest tests/ -v
```

Run tests **without Ollama** (useful if Ollama is not installed):

```bash
uv run pytest tests/ -m "not ollama" -v
```

Run tests with **coverage reporting**:

```bash
uv run pytest tests/ -m "not ollama" --cov=src/tokenkeeper --cov-report=term-missing
```

## Code Style

- **Follow existing patterns** in the codebase. When in doubt, look at how similar code is structured nearby.
- **Type hints** are required for all public functions and methods.
- **Pydantic** models are used for configuration and data validation. Use them for new config or structured data types.
- Keep functions focused and small. Prefer composition over inheritance.

## Pull Request Process

1. **Fork** the repository and create a feature branch from `master`.
2. **Write tests** for any new functionality. We aim for 80%+ test coverage.
3. **Ensure all tests pass** before submitting:
   ```bash
   uv run pytest tests/ -m "not ollama" -v
   ```
4. **Update `CHANGELOG.md`** under the "Unreleased" section with a brief description of your changes.
5. **Submit a pull request** with a clear description of what you changed and why.

## Reporting Issues

We use GitHub Issues to track bugs and feature requests.

- **Found a bug?** Use the [Bug Report](https://github.com/admin-sosys/TokenKeeper/issues/new?template=bug_report.yml) template. Include steps to reproduce, expected vs. actual behavior, and your environment details.
- **Have an idea?** Use the [Feature Request](https://github.com/admin-sosys/TokenKeeper/issues/new?template=feature_request.yml) template. Describe the use case, your proposed solution, and any alternatives you considered.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
