from prompt_diff_report.env import PACKAGE_DIR


def test__exist_package_dir() -> None:
    assert PACKAGE_DIR.exists() is True
