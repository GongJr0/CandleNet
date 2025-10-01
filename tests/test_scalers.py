import pytest
from sklearn.preprocessing import StandardScaler as SKStandard, RobustScaler as SKRobust, MinMaxScaler as SKMinMax
from CandleNet.scalers import P2Scaler, RobustScaler as CNRobust, StandardScaler as CNStandard, MinMaxScaler as CNMinMax
import numpy as np

z_cutoff = 4  # \approx cumulative p=0.9999 for std normal dist (No IQR correction due to small sample size)


@pytest.mark.parametrize("i", range(100))
def test_p2_1d(i, arr):
    scaler = P2Scaler(arr, f"test_p2_1d - {np.random.randint(0, 1e12)}")
    scaled = scaler.fit_transform()
    n = arr.shape[0]
    n_cutoff = np.where(np.abs(scaled) < z_cutoff, True, False).sum()

    assert scaled.shape == arr.shape
    assert n_cutoff/n >= 0.9999


@pytest.mark.parametrize("i", range(100))
def test_p2_nd(i, ndarray):
    scaler = P2Scaler(ndarray, f"test_p2_nd - {np.random.randint(0, 1e12)}")
    scaled = scaler.fit_transform()
    assert scaled.shape == ndarray.shape
    assert np.all(np.abs(scaled) <= z_cutoff)


@pytest.mark.parametrize("i", range(100))
def test_robust_1d(i, arr):
    scaled = SKRobust().fit_transform(arr)
    arr = CNRobust.fit_transform(arr)
    assert np.allclose(scaled, arr, atol=1e-6)


@pytest.mark.parametrize("i", range(100))
def test_robust_nd(i, ndarray):
    scaled = SKRobust().fit_transform(ndarray)
    ndarray = CNRobust.fit_transform(ndarray)
    assert np.allclose(scaled, ndarray, atol=1e-6)


@pytest.mark.parametrize("i", range(100))
def test_standard_1d(i, arr):
    scaled = SKStandard().fit_transform(arr)
    arr = CNStandard.fit_transform(arr, ddof=0)
    assert np.allclose(scaled, arr, atol=1e-6)


@pytest.mark.parametrize("i", range(100))
def test_standard_nd(i, ndarray):
    scaled = SKStandard().fit_transform(ndarray)
    ndarray = CNStandard.fit_transform(ndarray, ddof=0)
    assert np.allclose(scaled, ndarray, atol=1e-6)


@pytest.mark.parametrize("i", range(100))
def test_minmax_1d(i, arr):
    scaled = SKMinMax().fit_transform(arr)
    arr = CNMinMax.fit_transform(arr)
    assert np.allclose(scaled, arr, atol=1e-6)


@pytest.mark.parametrize("i", range(100))
def test_minmax_nd(i, ndarray):
    scaled = SKMinMax().fit_transform(ndarray)
    ndarray = CNMinMax.fit_transform(ndarray)
    assert np.allclose(scaled, ndarray, atol=1e-6)
