"""The package imports and reports a version."""

import mrutils


def test_the_package_reports_a_version():
    assert isinstance(mrutils.__version__, str)
    assert mrutils.__version__
