"""cf. https://docs.pytest.org/en/stable/example/simple.html"""
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--visualize",
        action="store_true",
        default=False,
        help="Enable interactive matplotlib tests",
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "visualize: mark test to enable interactive matplotlib tests")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--visualize"):
        return
    skip_visualize = pytest.mark.skip(reason="need --visualize option to run")
    for item in items:
        if "visualize" in item.keywords:
            item.add_marker(skip_visualize)
