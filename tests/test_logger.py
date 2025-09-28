import pytest
from utils import dummy_log


@pytest.mark.fast
@pytest.mark.unit
@pytest.mark.parametrize("i", range(200))
def test_logger_many(i, log_type, origin, msg):
    dummy_log(log_type, origin, msg)