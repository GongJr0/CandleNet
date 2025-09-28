import pytest
from sklearn.preprocessing import StandardScaler as SKStandard, RobustScaler as SKRobust, MinMaxScaler as SKMinMax
from CandleNet.scalers import P2Scaler, RobustScaler as CNRobust, StandardScaler as CNStandard, MinMaxScaler as CNMinMax
import numpy as np


@pytest.mark.parametrize("i", range(100))
def test_p2_1d(i, arr):
    scaler = P2Scaler(arr, f"test_p2_1d - {np.random.randint(0, 1e12)}")
    scaled = scaler.fit_transform()
    assert scaled.shape == arr.shape
    assert np.isclose(np.mean(scaled), 0, atol=3e-1)  # tol=0.3 due to small sample size


@pytest.mark.parametrize("i", range(100))
def test_p2_nd(i, ndarray):
    scaler = P2Scaler(ndarray, f"test_p2_nd - {np.random.randint(0, 1e12)}")
    scaled = scaler.fit_transform()
    assert scaled.shape == ndarray.shape
    assert np.all(np.isclose(np.mean(scaled, axis=0), 0, atol=3e-1))  # tol=0.3 due to small sample size


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


@pytest.mark.skip(reason="Sklearn uses ddof=0, CandleNet uses ddof=1")
@pytest.mark.parametrize("i", range(100))
def test_standard_1d(i, arr):
    scaled = SKStandard().fit_transform(arr)
    arr = CNStandard.fit_transform(arr)
    assert np.allclose(scaled, arr, atol=1e-6)


@pytest.mark.skip(reason="Sklearn uses ddof=0, CandleNet uses ddof=1")
@pytest.mark.parametrize("i", range(100))
def test_standard_nd(i, ndarray):
    scaled = SKStandard().fit_transform(ndarray)
    ndarray = CNStandard.fit_transform(ndarray)
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
