# Python Interface for GUNDAM

User friendly Python interface for GUNDAM.

The package is published as `gundam-interface` and imported as `gundam_interface`.

## Installation

```bash
pip install gundam-interface
```

For upgrading:
```bash
pip install --upgrade gundam-interface
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

### Modify event variables and rebin

Enable dynamic binning on the desired samples after `configure()` and before
`initialize()`. GUNDAM then retains events outside the histogram and stores the
variables needed to assign bins.

```python
interface.configure()
model = interface.getModel()
sample = model.getSampleSet().getSampleList()[0]
sample.setUseDynamicBinning(True)
interface.initialize()

event = sample.getEventList()[0]
variables = event.getVariables()
variables["X"] = 1.2  # Access by variable name or integer index.

model.updateEventSampleBinning()  # Reuses the propagator's thread pool.
llh = interface.evaluateLlh()  # Refills histograms and evaluates the likelihood.
```

Assignments preserve the stored C++ numeric type and refresh its cached value.
Assign integers to integer variables. Rebinning is explicit and updates only
samples with dynamic binning enabled. Events outside all bins keep index `-1`
and remain available for later migrations. `sample.getNbBinnedEvents()` returns
the number currently assigned to a bin.

`sample.updateEventSampleBinning()` updates a single sample. Use the propagator method
for repeated updates to reuse its thread pool. `evaluateLlh()` refills the model
histograms. Event-to-dial associations remain unchanged by these operations.

These methods require GUNDAM bindings exposing `VariableHolder.setValue()`,
`Sample.setUseDynamicBinning()`, and `updateEventSampleBinning()` on samples and propagators.

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
