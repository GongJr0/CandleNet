import numpy as np
import pandas as pd
from typing import Union, Callable, Any
from functools import reduce
from hashlib import sha256

FRAME = Union[pd.DataFrame, np.ndarray, pd.Series]
SERIES = Union[pd.Series, np.ndarray]


def matrix_minmax(matrix: FRAME, ignore_ones: bool = True) -> FRAME:
    """Min/Max rescale a matrix to [-1, 1]"""
    mat_min, mat_max = None, None
    if ignore_ones:
        mask = np.triu(np.ones_like(matrix, dtype=np.bool))
        triu = np.where(mask, matrix, np.nan)
        mat_min = np.nanmin(triu.flatten())
        mat_max = np.nanmax(triu.flatten())

    else:
        mat_min = np.min(matrix)
        mat_max = np.max(matrix)

    return 2 * ((matrix - mat_min) / (mat_max - mat_min)) - 1


def upper_idx(n, k=1):
    """Return the upper triangular indices of an (n x n) array, excluding the diagonal and k-1 superdiagonals."""
    rows, cols = np.triu_indices(n, k=k)
    return rows, cols


def signed_uniformize(C: FRAME):
    """Uniformize a correlation matrix while preserving the sign of the correlations."""
    C = pd.DataFrame(C)
    if isinstance(C, pd.DataFrame):
        A = C.to_numpy(dtype=float, copy=True)
    else:
        A = np.array(C, dtype=float, copy=True)

    n = A.shape[0]
    idx = upper_idx(n)

    # Fisher Transform
    eps = 1e-12
    R = np.clip(A, -1 + eps, 1 - eps)
    Z = np.arctanh(R)

    # Magnitude
    M = np.abs(Z)

    # ECDF
    m_upper = M[idx]
    m_ut = m_upper[np.isfinite(m_upper)]
    xs = np.sort(m_ut)
    ranks = np.searchsorted(xs, M[idx], side="right")
    U = (ranks - 0.5) / len(xs)

    # Sign Restoration
    S = np.zeros_like(A, dtype=float)
    S[idx] = np.sign(R[idx]) * U
    S = S + S.T
    np.fill_diagonal(S, 1.0)

    return pd.DataFrame(S, index=C.index, columns=C.columns)


def uptri_vals(A: FRAME) -> np.ndarray:
    """Return the upper triangular values of a square matrix, excluding the diagonal."""
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy(dtype=float, copy=True)
    n = A.shape[0]
    idx = upper_idx(n)
    return A[idx]


def uptri_abs_var(S):
    A = S.to_numpy()
    iu = np.triu_indices_from(A, k=1)
    v = np.abs(A[iu])
    return float(np.var(v)) if v.size else 0.0


def matrix_describe(A: FRAME) -> pd.Series:
    values = uptri_vals(A)
    return pd.Series(values).describe()


def pipe(*funcs) -> Callable[[Any], Any]:
    """Compose multiple single-argument functions into a single callable."""
    return lambda x: reduce(lambda x, f: f(x), funcs, x)


def _hash_str(s: str) -> str:
    """Return the SHA-256 hash of the input string."""
    return sha256(s.lower().strip().encode("utf-8")).hexdigest()


def str_encode(s: str) -> int:
    """Deterministically map a string to a 64-bit integer via SHA-256."""
    hash_hex = _hash_str(s)
    hash_int = int(hash_hex, 16)
    return hash_int & ((1 << 64) - 1)
