# Contributing

## Development Setup

```bash
pip install -r requirements.txt -c constraints.txt
pre-commit install
```

## Running Tests

```bash
pytest -m "not gpu" -q
```

Please follow conventional commits for commit messages.
