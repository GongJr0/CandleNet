from __future__ import annotations
from typing import Iterable, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch


def _valid_lag(l) -> bool:  # noqa: E741
    """True if finite, positive, and (near-)integer with 1e-6 precision tolerance.
    (lower tol may be too strict for float32)"""
    try:
        lf = float(l)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(lf) or lf <= 0:
        return False
    # Avoid float precision issues (e.g. 1.0000000000000002)
    return abs(lf - round(lf)) < 1e-6


def _as_1d_float(a) -> np.ndarray:
    """Accept pd.Series/np.ndarray/list → 1D float np.array; drop NaNs at the end."""
    if isinstance(a, pd.Series):
        arr = a.to_numpy()
    elif isinstance(a, np.ndarray):
        arr = a
    else:
        arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array/Series, got shape {arr.shape}")
    return arr.astype(float, copy=False)


def _gather_lags(
    series: np.ndarray, lags: Iterable[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return values at given positive lags (t-l) aligned to the *last* observation t=len(series)-1.
    Output shape: (K, 1) where K=len(valid lags present)."""
    lags = sorted(int(l) for l in lags if _valid_lag(l))  # noqa: E741
    K = len(lags)

    vals = np.empty((K, 1), dtype=float)
    t = len(series) - 1
    keep: list = []
    for i, lag in enumerate(lags):
        idx = t - lag
        if idx < 0 or np.isnan(series[idx]):
            continue  # skip missing/nonexistent
        vals[len(keep), 0] = series[idx]
        keep.append(lag)

    return vals[: len(keep)], np.asarray(keep, dtype=np.int64)


def format_sparse_lag_sample(
    y: pd.Series | np.ndarray | list[float],
    selected_lags: Iterable[int],
    *,
    ticker_id: int,
    sector_id: Optional[int] = None,
    K_pad: Optional[
        int
    ] = None,  # pad to this K (if None, no pad here; collate will pad)
    pad_value: float = 0.0,
    pad_idx: int = 0,  # reserve 0 in lag embedding as PAD
    vals_dtype: torch.dtype = torch.float32,
    idx_dtype: torch.dtype = torch.long,
) -> Dict[str, Any]:
    """
    Produce a single-sample dict ready for the variable-length collate:
      {
        'vals': (K, F_val) torch.float,
        'lag_ids': (K,) torch.long,
        'ticker_id': int,
        'sector_id': Optional[int],
        'pad_mask': (K,) torch.bool   # only if K_pad provided here
      }
    """
    assert isinstance(
        vals_dtype, torch.dtype
    ), f"[vals_dtype] Expected torch.dtype, got {type(vals_dtype)}"
    assert isinstance(
        idx_dtype, torch.dtype
    ), f"[idx_dtype] Expected torch.dtype, got {type(idx_dtype)}"

    series = _as_1d_float(y)
    vals_np, lag_ids_np = _gather_lags(series, selected_lags)  # (K,1), (K,)
    K = vals_np.shape[0]

    # If nothing valid, return a single PAD row; downstream mask will ignore it.
    if K == 0:
        K_eff = 1 if K_pad is None else max(1, K_pad)
        vals = torch.full((K_eff, 1), pad_value, dtype=vals_dtype)
        lag_ids = torch.full((K_eff,), pad_idx, dtype=idx_dtype)
        pad_mask = torch.ones((K_eff,), dtype=torch.bool)
        out = {
            "vals": vals,
            "lag_ids": lag_ids,
            "ticker_id": int(ticker_id),
            "sector_id": int(sector_id) if sector_id is not None else None,
        }
        if K_pad is not None:
            out["pad_mask"] = pad_mask
        return out

    # Convert to torch
    vals = torch.as_tensor(vals_np, dtype=vals_dtype)  # (K,1)
    lag_ids = torch.as_tensor(lag_ids_np, dtype=idx_dtype)  # (K,)

    # Optional padding
    if K_pad is not None:
        if K_pad < K:
            # Left-align, keep the first K_pad
            vals = vals[:K_pad]
            lag_ids = lag_ids[:K_pad]
            K = K_pad
        else:
            pad_rows = K_pad - K
            if pad_rows > 0:
                vals = torch.cat(
                    [
                        vals,
                        torch.full(
                            (pad_rows, vals.shape[1]), pad_value, dtype=vals_dtype
                        ),
                    ],
                    dim=0,
                )
                lag_ids = torch.cat(
                    [lag_ids, torch.full((pad_rows,), pad_idx, dtype=idx_dtype)],
                    dim=0,
                )
        pad_mask = torch.zeros((K_pad,), dtype=torch.bool)
        if K_pad > K:
            pad_mask[K:] = True

    # Dtype checks
    if vals.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"vals must be float32/64, got {vals.dtype}")
    if lag_ids.dtype != torch.long:
        raise TypeError(f"lag_ids must be int64/long, got {lag_ids.dtype}")
    if torch.isnan(vals).any() or torch.isinf(vals).any():
        raise ValueError("vals contain NaN/Inf after formatting")

    out = {
        "vals": vals,  # (K or K_pad, 1)
        "lag_ids": lag_ids,  # (K or K_pad,)
        "ticker_id": int(ticker_id),
        "sector_id": int(sector_id) if sector_id is not None else None,
    }
    if K_pad is not None:
        out["pad_mask"] = pad_mask
    return out
