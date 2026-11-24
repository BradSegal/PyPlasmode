"""Keep the agent skill package loadable and its examples aligned with the public API."""

import re
from pathlib import Path

import pytest

SKILL = Path(__file__).parents[2] / "skills" / "pyplasmode"


@pytest.mark.integration
def test_skill_metadata_names_the_package() -> None:
    """The skill file opens with a name and a description, as skill loaders require."""
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    match = re.match(r"---\nname: (.+)\ndescription: (.+)\n---\n", text)
    assert match is not None
    assert match.group(1) == "pyplasmode"
    assert len(match.group(2)) > 200
    assert "references/recipes.md" in text
    assert len(text.splitlines()) < 500


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:.*:FutureWarning:sklearn.*")
@pytest.mark.parametrize("name", ["SKILL.md", "references/recipes.md"])
def test_skill_examples_run(name: str) -> None:
    """Every Python block in the skill runs on its own against the installed library."""
    source = SKILL / name
    blocks = re.findall(r"```python\n(.*?)```", source.read_text(encoding="utf-8"), re.DOTALL)
    assert blocks
    for block in blocks:
        exec(compile(block, str(source), "exec"), {})
