# Tests

This test suite checks the GUNDAM Python interface at two levels:

- interface unit tests using mock objects, in `test_package.py`;
- integration tests using the real GUNDAM Python bindings, in the numbered
  `test_00_...` through `test_04_...` files.

## Requirements

From the project root, install the development dependencies:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The integration tests also require:

- a GUNDAM installation whose Python bindings are importable;
- `uproot`, which is installed automatically with the package.

If the bindings are not already available in the Python environment, add the
GUNDAM `lib` directory to `PYTHONPATH` before running pytest:

```bash
export PYTHONPATH="/path/to/gundam/lib:$PYTHONPATH"
```

## Running the tests

From the repository root:

```bash
.venv/bin/pytest -q
```

Or, if the virtual environment is activated:

```bash
pytest -q
```

The full suite includes both unit and integration tests. The integration tests
require a usable GUNDAM installation. Pass its `lib` directory with
`--gundam-lib-path` when it is not already importable from the environment.

## Integration test sequence

The numbered files describe the workflow in order:

1. `test_00_generate_inputs.py` generates a temporary ROOT file and YAML
   configuration with `uproot`;
2. `test_01_initialize_interface.py` configures and initializes the interface;
3. `test_02_evaluate_likelihood.py` evaluates the likelihood;
4. `test_03_access_samples.py` checks access to model and data samples;
5. `test_04_access_histogram.py` checks histogram access.

The inputs are created in pytest's temporary directory and do not modify the
repository. Shared fixtures are defined in `conftest.py` and reuse one ROOT/YAML
input pair and one initialized interface instance for the test session.

## Running part of the suite

Run only the integration tests:

```bash
.venv/bin/pytest -q tests/test_0*.py
```

Run only the input-generation test:

```bash
.venv/bin/pytest -q tests/test_00_generate_inputs.py
```

Run the integration tests with an explicit GUNDAM installation path:

```bash
.venv/bin/pytest -q tests/test_0*.py \
  --gundam-lib-path /home/work/install/lib
```

Run the unit tests in `test_package.py`:

```bash
.venv/bin/pytest -q tests/test_package.py
```

To display detailed GUNDAM logs, remove `-q` and use `-s` to show standard
output directly:

```bash
.venv/bin/pytest tests/test_0*.py -s
```
