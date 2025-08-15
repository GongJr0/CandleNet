import pytest


@pytest.fixture(scope="module")
def ticker():
    """
    Fixture to provide a ticker symbol for testing.
    """
    return "BRK.B"

@pytest.fixture(scope="module")
def index():
    return "GSPC"

@pytest.fixture(scope="module")
def period():
    return "1mo"

@pytest.fixture(scope="module")
def interval():
    return "1d"