"""The release this branch is: above PyPI's 0.3.0, so that a pin of ``>=0.4.0`` never resolves PyPI's
0.3.0 (which has none of this branch's modules), with upstream's floors for pydantic and rich."""
import importlib.metadata
import tomllib
from pathlib import Path


def test_the_package_says_0_4_0():
    assert importlib.metadata.version("aisc-plugin-interface") == "0.4.0"


def test_it_needs_the_upstream_floors_of_pydantic_and_rich():
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert pyproject["project"]["dependencies"] == ["pydantic>=2.13.5", "rich>=15.0.0"]
