# Python Interface for GUNDAM

User friendly Python interface for GUNDAM.

The package is published as `gundam-interface` and imported as `gundam_interface`.

## Installation

```bash
pip install gundam-interface
```

For upgrading:
```bash
python -m pip install --upgrade gundam-interface
```

For local development:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Usage

```python
import gundam_interface

print(gundam_interface.__version__)
```

## Build

```bash
python -m build
python -m twine check dist/*
```

## Publish a new PyPI version

Publishing is handled by the GitHub Actions workflow
[`.github/workflows/publish.yml`](.github/workflows/publish.yml).

Before the first release, configure PyPI Trusted Publishing for this project:

- PyPI project name: `gundam-interface`
- Owner/repository: this GitHub repository
- Workflow name: `publish.yml`
- Environment name: `pypi`

To publish a release:

1. Make sure the `main` branch is ready and CI is passing.
2. Create a new GitHub Release with a semantic tag such as `v0.2.1`.
3. Publishing starts automatically when the release is published.
4. The workflow derives the package version from the tag, updates
   `pyproject.toml`, commits that version bump to `main`, builds the package,
   and uploads it to PyPI.

The workflow can also be started manually from the GitHub Actions page with a
`version` input such as `0.2.1`.


## License

This project is distributed under the GNU Lesser General Public License v2.1 or later.
See [LICENCE](LICENCE) for details.
