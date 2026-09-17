"""Run the public tutorial blocks as a reader would, with their shared setup."""

import re
from pathlib import Path

import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:.*:FutureWarning:sklearn.*")
@pytest.mark.parametrize(
    "page", ["README", "tutorial", "outcomes", "structures-and-extensions", "evaluation"]
)
def test_documented_workflow(page: str) -> None:
    """Keep executable documentation aligned with the installed public API."""
    root = Path(__file__).parents[2]
    source = root / "README.md" if page == "README" else root / "docs" / f"{page}.md"
    namespace: dict[str, object] = {}
    blocks = re.findall(r"```python\n(.*?)```", source.read_text(), re.DOTALL)
    assert blocks
    for block in blocks:
        exec(compile(block, str(source), "exec"), namespace)
    if page == "tutorial":
        assert isinstance(namespace["model"], GridSearchCV)
        assert isinstance(namespace["model"].estimator, Pipeline)
        assert namespace["matched"].chance_adjusted_recovery == 0.5
        assert namespace["exact"].recall == 1.0
    elif page == "outcomes":
        assert namespace["binary"].values.shape == (10_000,)
        assert namespace["survival"].time.shape == (10_000,)
    elif page == "structures-and-extensions":
        assert namespace["alignment"].unmapped == ("G",)
    elif page == "evaluation":
        assert namespace["result"].at_depth[0].recall == 0.5
    else:
        assert namespace["recovery"].at_depth[0].recall == 1.0
