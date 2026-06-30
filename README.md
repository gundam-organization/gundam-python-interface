# Python Interface for GUNDAM

User friendly Python interface for GUNDAM.

The package is published as `gundam-interface` and imported as `gundam_interface`.

## Installation

```bash
pip install gundam-interface
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

## License

This project is distributed under the GNU Lesser General Public License v2.1 or later.
See [LICENCE](LICENCE) for details.
