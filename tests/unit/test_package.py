from importlib.metadata import version

import evoworld


def test_package_exposes_version() -> None:
    assert evoworld.__version__ == version("evoworld")
