import pytest
from CandleNet.autoreg import format_sparse_lag_sample
import numpy as np
import torch
import time


@pytest.mark.fast
@pytest.mark.unit
@pytest.mark.parametrize("i", range(100))
def test_lag_gen_clean_inp(i, arr):
    """Randomized input structured to the exact expected format."""

    lag_lim = len(arr)
    n_lags = np.random.randint(low=1, high=lag_lim//2)
    lags = np.random.choice(np.arange(1, lag_lim), n_lags, replace=False)
    sample = format_sparse_lag_sample(arr.flatten(), lags, ticker_id=-1)
    vals = sample['vals']

    assert vals.shape == (len(lags), 1)


@pytest.mark.fast
@pytest.mark.unit
@pytest.mark.parametrize("i", range(100))
def test_lag_gen_NaN_input(i, arr):
    """Randomized input with NaNs inserted."""

    lag_lim = len(arr)
    n_lags = np.random.randint(low=1, high=lag_lim // 2)
    lags = np.random.choice(np.arange(1, lag_lim), n_lags, replace=False)
    n_nans = np.random.randint(1, 5)
    nans = n_nans * [np.nan]
    lags = np.array([*lags, *nans])

    sample = format_sparse_lag_sample(arr.flatten(), lags, ticker_id=-1)

    assert sample['vals'].shape == (len(lags) - n_nans, 1)


@pytest.mark.fast
@pytest.mark.unit
@pytest.mark.parametrize("i", range(100))
def test_lag_gen_invalid_inp(i, arr):
    """Randomized input with invalid integers inserted."""

    lag_lim = len(arr)
    n_lags = np.random.randint(low=1, high=lag_lim // 2)
    lags = np.random.choice(np.arange(1, lag_lim), n_lags, replace=False)
    n_invalid = np.random.randint(1, 5)

    low_invalid = n_invalid * [0]
    high_invalid = n_invalid * [lag_lim + np.random.randint(1, 10_000)]
    neg_invalid = n_invalid * [-np.random.randint(1, 10_000)]
    pos_inf = n_invalid * [np.inf]
    neg_inf = n_invalid * [-np.inf]
    invalids = np.array([*low_invalid, *high_invalid, *neg_invalid, *pos_inf, *neg_inf])
    np.random.shuffle(invalids)


    lags = np.array([*lags, *invalids])
    sample = format_sparse_lag_sample(arr.flatten(), lags, ticker_id=-1)

    assert sample['vals'].shape == (len(lags) - 5*n_invalid, 1)


@pytest.mark.fast
@pytest.mark.unit
@pytest.mark.parametrize("i", range(100))
def test_non_integer_lags_rejected(i, arr):
    lags = [
        1.0,  # valid
        np.float64(2.000000000004),  # valid (near-integer)
        "3",  # valid (string integer)

        6.7,  # invalid
        np.pi,  # invalid
        np.sqrt(2),  # invalid
        "foo",  # invalid
    ]
    out = format_sparse_lag_sample(arr.flatten(), lags, ticker_id=-1)
    # only the integer-like ones remain
    assert out["lag_ids"].tolist() == [1, 2, 3]
    assert out["vals"].shape == (3, 1)


@pytest.mark.fast
@pytest.mark.unit
@pytest.mark.parametrize("i", range(100))
def test_pad_only_without_kpad(i, arr):
    n = len(arr)
    out = format_sparse_lag_sample(arr.flatten(), [n+1, n+2], ticker_id=-1, K_pad=None)
    assert out["vals"].shape == (1, 1)
    assert out["lag_ids"].tolist() == [0]   # pad_idx
    assert "pad_mask" not in out


@pytest.mark.fast
@pytest.mark.unit
@pytest.mark.parametrize("i", range(100))
def test_pad_only_with_kpad(i, arr):
    n = len(arr)
    pad = 4
    out = format_sparse_lag_sample(arr.flatten(), [n+1, n+2], ticker_id=-1, K_pad=pad)
    assert out["vals"].shape == (pad, 1)
    assert out["lag_ids"].tolist() == [0]*pad   # pad_idx
    assert out["pad_mask"].dtype == torch.bool
    assert out["pad_mask"].all().item() is True


@pytest.mark.slow
@pytest.mark.unit
@pytest.mark.parametrize("i", range(100))
def test_large_series_quick(i):
    t = time.time()
    n = 2_000_000
    y = np.random.randn(n).astype(float)
    lags = np.array(np.random.choice(np.arange(1, n), 1_000, replace=False))
    out = format_sparse_lag_sample(y, lags, ticker_id=0)
    assert out["vals"].shape == (len(lags), 1)
    assert time.time()-t <= 1.0
