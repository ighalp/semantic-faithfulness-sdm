# Contributing to Semantic Faithfulness SDM

Thank you for considering contributing to this project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Python version and package versions (`pip list`)
- Minimal code example if possible

### Suggesting Enhancements

For feature requests or enhancements:
- Open an issue with `[Feature Request]` in the title
- Describe the proposed functionality
- Explain the use case and benefits
- Provide examples if possible

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes**:
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed
3. **Test your changes**:
   ```bash
   pytest tests/
   ```
4. **Submit a pull request** with:
   - Clear description of changes
   - Link to related issues
   - Test results

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/semantic-faithfulness-sdm.git
cd semantic-faithfulness-sdm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Check code style
black sdm_package/
flake8 sdm_package/
```

## Code Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting
- Add type hints where appropriate
- Write docstrings in [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

### Example Docstring

```python
def compute_semantic_faithfulness(
    p_context: np.ndarray,
    p_question: np.ndarray,
    p_answer: np.ndarray,
    return_all: bool = False
) -> Dict[str, float]:
    """Compute Semantic Faithfulness (F_S) and Entropy Production (SEP) metrics.

    Args:
        p_context: Probability distribution over topics for context (N,)
        p_question: Probability distribution over topics for question (N,)
        p_answer: Probability distribution over topics for answer (N,)
        return_all: If True, return all intermediate metrics

    Returns:
        Dictionary containing:
            - F_S: Semantic Faithfulness score [0, 1]
            - SEP: Semantic Entropy Production (bits)
            - Ṡ: System Entropy Change = H(A) - H(C) (bits) (if return_all=True)
            - D_min: Minimal KL divergence (if return_all=True)
            - ... other metrics ...

    Raises:
        ValueError: If distributions have different dimensions
        RuntimeError: If optimization fails to converge

    Example:
        >>> results = compute_semantic_faithfulness(p_c, p_q, p_a, return_all=True)
        >>> print(f"Faithfulness: {results['F_S']:.3f}")
    """
```

## Testing

### Writing Tests

- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use descriptive test names

```python
def test_semantic_faithfulness_range():
    """F_S should be in range (0, 1]"""
    # Setup
    p_c = np.array([0.5, 0.3, 0.2])
    p_q = np.array([0.4, 0.4, 0.2])
    p_a = np.array([0.3, 0.5, 0.2])

    # Execute
    result = compute_semantic_faithfulness(p_c, p_q, p_a)

    # Assert
    assert 0 < result['F_S'] <= 1.0
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sdm_package tests/

# Run specific test file
pytest tests/test_semantic_faithfulness.py

# Run specific test
pytest tests/test_semantic_faithfulness.py::test_semantic_faithfulness_range
```

## Documentation

### Updating Documentation

When adding new features or changing APIs:
1. Update relevant files in `docs/`
2. Update docstrings in code
3. Update `README.md` if needed
4. Add examples if appropriate

### Building Documentation

(Future: Sphinx documentation)

## Commit Messages

Follow conventional commit format:

```
<type>: <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Example:
```
feat: add support for custom embedding models

Allow users to pass custom SentenceTransformer models
instead of model names for more flexibility.

Closes #42
```

## Code Review Process

1. All submissions require review
2. Maintainers will review PRs within 1 week
3. Address reviewer feedback
4. Once approved, maintainers will merge

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue with your question or contact the maintainers directly.

Thank you for contributing! 🙏
