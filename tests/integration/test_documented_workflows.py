"""Run the public tutorial blocks as a reader would, with their shared setup."""

import re
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:.*:FutureWarning:sklearn.*")
@pytest.mark.parametrize("page", ["tutorial", "outcomes", "structures-and-extensions"])
def test_documented_workflow(page: str) -> None:
    """Keep executable documentation aligned with the installed public API."""
    source = Path(__file__).parents[2] / "docs" / f"{page}.md"
    namespace: dict[str, object] = {}
    blocks = re.findall(r"```python\n(.*?)```", source.read_text(), re.DOTALL)
    assert blocks
    for block in blocks:
        exec(compile(block, str(source), "exec"), namespace)
    if page == "tutorial":
        assert namespace["matched"].chance_adjusted_recovery == 0.5
        assert namespace["exact"].recall == 1.0
    elif page == "outcomes":
        assert namespace["binary"].values.shape == (10_000,)
        assert namespace["survival"].time.shape == (10_000,)
    else:
        assert namespace["alignment"].unmapped == ("G",)
