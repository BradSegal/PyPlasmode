# Contributing

Contributions to generators, evaluation functions, documentation and examples are welcome.
For a new signal or outcome family, describe the intended use and mathematical definition
in an issue before implementing it.

Install a supported Python version (3.12-3.15) and uv. From the repository root, set up the
environment with the validation examples and the documentation tools:

```bash
uv sync --extra validation --group docs
```

Add `--extra benchmark` to run the synthetic Cox/ridge/XGBoost example
(`examples/benchmark_recovery_demo.py`) and its integration test, which is otherwise skipped.

Before opening a pull request, run the checks that CI runs:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/pyplasmode
uv run mkdocs build --strict
```

Before a release, also build the distributions into an empty `dist/` and check them:

```bash
rm -rf dist
uv run python -m build
uv run twine check --strict dist/*
uv run check-wheel-contents dist/*.whl
```

Run `uv run python examples/validation_vignette.py --output validation-output` to regenerate the
synthetic validation figures and tables. The `validation` extra is included in the setup above.

The local uv lock records your development environment. Supported dependency ranges are
listed in `pyproject.toml`.

## Changes and support

Use the issue tracker for bug reports, questions and feature proposals. For a bug,
include dependency versions, the expected result and a small synthetic reproducer. Keep
source data and credentials out of reports. Send security-sensitive reports privately
to the repository maintainer.

Add a focused regression test for bug fixes. Generator changes also need checks of their
distributional properties, including null signals and a reference calculation or estimator.
Integration tests exercise complete user workflows; the examples show how to combine the
library with model fitting, tuning and explanation tools.

## Releases and compatibility

Record public changes in the changelog. Incompatible API or statistical-definition changes
increment the major version; compatible additions increment the minor version, and fixes
increment the patch version. Explain corrections that affect numerical results in the release notes.
Record random seeds and dependency versions when sharing reproducible analyses.

The CI matrix covers minimum numerical dependencies on Python 3.12 and current dependencies
on supported Python versions, with Linux, Windows and macOS jobs. Test dependency changes
against this matrix. Build release artifacts from the tagged version after CI passes.

Attach the checked wheel and source distribution to a draft GitHub release, then publish
the release. The `publish.yml` workflow validates those distributions and uploads them to
PyPI through the `pypi` trusted-publishing environment. For an existing release, run
`gh workflow run publish.yml --ref main -f tag=v1.0.0` with its tag. No PyPI API token is
required, and already-published versions must not be replaced.
