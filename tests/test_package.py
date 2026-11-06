import importlib.util


def test_parrotly_package_is_importable():
    assert importlib.util.find_spec("parrotly") is not None
