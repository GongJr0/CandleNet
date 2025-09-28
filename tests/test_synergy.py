from CandleNet.synergy_matrix import Synergy
import numpy as np
import pytest


@pytest.mark.slow
def test_synergy(s: Synergy):
    mat = s.signed_synergy()
    vals = mat.dropna().values.flatten()
    assert_mask = np.where((-1.0 <= vals) & (vals <= 1.0), True, False)


    assert mat.shape[0] == mat.shape[1], "Synergy matrix is not square"
    assert np.all(assert_mask), "Synergy values out of bounds"
