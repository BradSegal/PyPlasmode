# Contributing

Contributions to generators, evaluation functions, documentation and examples are welcome.
For a new signal or outcome family, describe the intended use and mathematical definition
in an issue before implementing it.

From a checkout, install a supported Python version (3.12-3.14) and uv, then run:

```bash
uv sync --extra validation --group docs
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/pyplasmode
uv run mkdocs build --strict
uv run python -m build
uv run twine check --strict dist/*
uv run check-wheel-contents dist/*.whl
```

Run `python examples/validation_vignette.py --output validation-output` to regenerate the
synthetic validation figures and tables. The `validation` extra is included in the setup above.

The local uv lock records your development environment. Supported dependency ranges are
listed in `pyproject.toml`.

## Changes and support

Use the repository's issue tracker for bug reports, questions and feature proposals. For a bug,
include dependency versions, the expected result and a small synthetic reproducer. Keep
participant data and credentials out of reports. Send security-sensitive reports privately
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
