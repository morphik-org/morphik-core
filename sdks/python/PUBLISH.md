# Publish to PyPI

- `cd` into the `sdks/python` directory
- Update the package version in `pyproject.toml`, `databridge/__init__.py`.
- Ensure you have the correct PyPI API key/certificates/ssh keys installed

```bash
# ensure you've activated the correct python environment
pip install build twine

rm -rf dist
python -m build
twine check dist/*
twine upload dist/*
```
