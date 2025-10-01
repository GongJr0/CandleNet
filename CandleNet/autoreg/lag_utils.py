from __future__ import annotations
from typing import Generator, Optional, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm  # type: ignore
from statsmodels.stats.multitest import multipletests  # type: ignore
from scipy.stats import norm  # type: ignore
from CandleNet.utils import SERIES
from math import sqrt


def get_formatted_arr(arr: SERIES) -> pd.DataFrame:
    assert arr.ndim == 1, "Input must be 1-dimensional"
    if isinstance(arr, pd.Series):
        df = pd.DataFrame(arr.values, columns=["T"])

    elif isinstance(arr, np.ndarray):
        df = pd.DataFrame(arr, columns=["T"])

    else:
        df = pd.DataFrame(arr, columns=["T"])

    return df


def get_lags(arr: SERIES, n_lags: int) -> pd.DataFrame:
    """Generate a DataFrame of lagged versions of the input series. NaNs are not dropped."""
    df = get_formatted_arr(arr)

    for i in range(1, n_lags + 1):
        df[f"T-{i}"] = df["T"].shift(i)

    return df


def get_lag(arr: SERIES, lags: list[int]) -> pd.DataFrame:
    """Generate a DataFrame of lagged versions of the input series. NaNs are not dropped."""
    df = get_formatted_arr(arr)

    for i in lags:
        df[f"T-{i}"] = df["T"].shift(i)

    return df


def _prep_xy(y: SERIES, k: int):
    y = pd.Series(y).astype(float)
    x = y.shift(k)
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    # de-mean to avoid intercept sensitivity in tiny samples
    df = df - df.mean()
    return df["y"].to_numpy(), df["x"].to_numpy()


def _auto_nw_bandwidth(n: int) -> int:
    # Andrews(1991)/Newey-West style small-sample bandwidth; keep it tiny for daily data
    # 4 * (n/100)^(2/9), at least 1
    bw = int(round(4.0 * (max(n, 2) / 100.0) ** (2.0 / 9.0)))
    return max(bw, 1)


def lag_significance_hac(y, max_lag=20, bandwidth="auto", fdr=False, alpha=0.05):
    """
    Returns a DataFrame with beta, HAC t and p-values for y_t ~ y_{t-k}.
    """

    rows = []
    y = pd.Series(y).astype(float)
    for k in range(1, max_lag + 1):
        yy, xx = _prep_xy(y, k)
        n = len(yy)
        if n < 10:
            rows.append((k, np.nan, np.nan, np.nan, n))
            continue
        X = sm.add_constant(xx, has_constant="add")
        model = sm.OLS(yy, X)
        if bandwidth == "auto":
            nlags = _auto_nw_bandwidth(n)
        else:
            nlags = int(bandwidth)
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": nlags})
        beta = res.params[1]
        tval = res.tvalues[1]
        pval = res.pvalues[1]
        rows.append((k, beta, tval, pval, n))

    out = pd.DataFrame(rows, columns=["lag", "beta", "t", "p", "n"])

    if fdr:
        # Benjamini–Hochberg on the p's (ignore NaNs)
        p = out["p"].to_numpy()
        idx = np.where(~np.isnan(p))[0]
        if idx.size:
            p_sorted_idx = idx[np.argsort(p[idx])]
            m = idx.size
            ranks = np.arange(1, m + 1)
            thresh = (ranks / m) * alpha
            p_sorted = p[p_sorted_idx]
            keep_upto = np.where(p_sorted <= thresh)[0]
            reject = np.zeros_like(p, dtype=bool)
            if keep_upto.size:
                reject[p_sorted_idx[: keep_upto.max() + 1]] = True
            out["reject_fdr"] = reject
        else:
            out["reject_fdr"] = False
    return out


def _infer_block_len(n: int) -> int:
    """Heuristic: l = max(5, floor(n**(1/3)))."""
    return max(5, int(np.floor(n ** (1 / 3))))


def cbb_indices(
    n: int,
    B: int,
    block_len: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Generator[np.ndarray, None, None]:
    """
    Yield bootstrap index arrays for Circular Block Bootstrap.

    Parameters
    ----------
    n : int
        Sample length (time dimension).
    B : int
        Number of bootstrap replicates.
    block_len : int, optional
        Block length l. If None, uses a simple heuristic.
    rng : np.random.Generator, optional
        Numpy RNG. If None, uses default Generator.

    Yields
    ------
    idx : np.ndarray of shape (n,)
        Indices into the original series for one bootstrap replicate.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if B <= 0:
        raise ValueError("B must be positive.")

    if block_len is None:
        block_len = _infer_block_len(n)
    if block_len <= 0:
        raise ValueError("block_len must be positive.")

    # For very small n, cap block length
    l = min(block_len, n)  # noqa: E741

    rng = np.random.default_rng() if rng is None else rng

    # Precompute starting points for blocks (0...n-1).
    # CBB wraps modulo n, so every block is [s, s+1, ..., s+l-1] mod n.
    n_blocks = int(np.ceil(n / l))

    for _ in range(B):
        starts = rng.integers(low=0, high=n, size=n_blocks, endpoint=False)
        # Build blocks via broadcasting, then wrap mod n
        offsets = np.arange(l)[None, :]
        blocks = (starts[:, None] + offsets) % n  # shape: (n_blocks, l)
        idx = blocks.reshape(-1)[:n]  # trim to exact length n
        yield idx


def cbb_sample(
    X: np.ndarray,
    B: int,
    block_len: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    return_indices: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Generate B CBB resamples of X (joint resampling across columns if 2D).

    Parameters
    ----------
    X : np.ndarray
        Time-series array. Shape (n,) or (n, p). If 1D, treated as (n, 1).
    B : int
        Number of bootstrap replicates.
    block_len : int, optional
        Block length l. If None, uses heuristic.
    rng : np.random.Generator, optional
        RNG for reproducibility.
    return_indices : bool
        If True, also return the index matrix used for resampling.

    Returns
    -------
    Xb : np.ndarray
        Bootstrap samples. Shape:
            - (B, n) if input was 1D
            - (B, n, p) if input was 2D
    Ib : np.ndarray or None
        Indices used, shape (B, n), if return_indices=True else None.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X2 = X[:, None]
        squeeze = True
    elif X.ndim == 2:
        X2 = X
        squeeze = False
    else:
        raise ValueError("X must be 1D or 2D (time[, features]).")

    n = X2.shape[0]
    p = X2.shape[1]

    Ib = np.empty((B, n), dtype=np.int64)
    Xb = np.empty((B, n, p), dtype=X2.dtype)

    gen = cbb_indices(n=n, B=B, block_len=block_len, rng=rng)
    for b, idx in enumerate(gen):
        Ib[b] = idx
        Xb[b] = X2[idx, :]

    if squeeze:
        Xb = Xb[:, :, 0]  # -> (B, n)

    return (Xb, Ib) if return_indices else (Xb, None)


def _wilson_interval(
    successes: np.ndarray, trials: np.ndarray, conf: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized Wilson score interval for binomial proportion.
    Returns (lo, hi) arrays.
    """
    successes = successes.astype(float)
    trials = trials.astype(float)
    p_hat = np.divide(
        successes, trials, out=np.zeros_like(successes, dtype=float), where=trials > 0
    )

    # z for two-sided
    z = float(norm.ppf(0.5 + conf / 2.0))
    denom = 1.0 + (z * z) / trials
    center = p_hat + (z * z) / (2.0 * trials)
    spread = z * np.sqrt(
        np.divide(
            p_hat * (1.0 - p_hat),
            trials,
            out=np.zeros_like(trials, dtype=float),
            where=trials > 0,
        )
        + (z * z) / (4.0 * trials * trials)
    )
    lo = (center - spread) / denom
    hi = (center + spread) / denom
    # clamp to [0,1]
    lo = np.clip(lo, 0.0, 1.0)
    hi = np.clip(hi, 0.0, 1.0)
    return lo, hi


def bootstrapped_significance(
    y: pd.Series,
    max_lag: int = 20,
    B: int = 200,
    block_len: Optional[int] = None,
    bandwidth: str = "auto",
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
    # optional knobs:
    use_fdr_end: bool = True,
    min_freq: float = 0.25,
    early_stop: bool = True,
    b_min: int = 100,
    check_every: int = 25,
    conf: float = 0.95,
) -> pd.DataFrame:
    """
    Stability selection of significant lags using Circular Block Bootstrap (CBB).
    Returns per-lag selection frequency across bootstraps, plus base-sample p's.

    Interpretation: 'freq' = Pr[lag significant on a CBB resample at level alpha].
    This captures stability, not the null distribution of the statistic.
    """

    rng = np.random.default_rng() if rng is None else rng

    # Base sample HAC results (for reporting + optional end-of-pipeline FDR)
    base = lag_significance_hac(
        y.to_numpy(dtype=float).ravel(),
        max_lag=max_lag,
        bandwidth=bandwidth,
        fdr=False,  # get raw p's; apply FDR later if desired
        alpha=alpha,
    )
    # base: DataFrame with columns ['lag', 't', 'p'] (assumed)
    base = base.set_index("lag").reindex(range(1, max_lag + 1))

    # Bootstrap resamples
    y_arr = y.to_numpy(dtype=float).ravel()
    Xb, _ = cbb_sample(y_arr, B=B, block_len=block_len, rng=rng)  # shape (B, n)

    # Early stopping state per lag
    decided = np.zeros(max_lag, dtype=bool)
    decision_b = np.full(max_lag, -1, dtype=int)  # bootstrap round when decided
    stable_flag = np.zeros(max_lag, dtype=bool)  # True if freq >= min_freq

    # Tally selection frequency and top-rank counts
    select_counts = np.zeros(max_lag, dtype=int)
    top_rank_counts = np.zeros(max_lag, dtype=int)

    for b in range(1, B + 1):
        if decided.all() and early_stop:
            break

        res_b = (
            lag_significance_hac(
                Xb[b - 1],
                max_lag=max_lag,
                bandwidth=bandwidth,
                fdr=False,
                alpha=alpha,
            )
            .set_index("lag")
            .reindex(range(1, max_lag + 1))
        )

        # Only update counts for undecided lags
        mask = ~decided
        if mask.any():
            p_vals = res_b["p"].to_numpy()[mask]
            sel = (p_vals < alpha).astype(int)
            select_counts[mask] += sel

        # rank-based winner (for diagnostics only)
        top = int(res_b["p"].argmin()) + 1
        top_rank_counts[top - 1] += 1

        # Early stopping check
        if early_stop and b >= b_min and (b % check_every == 0):
            # trials so far for undecided lags is b; for decided we keep their decision point
            trials = np.where(decided, np.maximum(decision_b, 1), b).astype(float)
            lo, hi = _wilson_interval(select_counts.astype(float), trials, conf=conf)

            # Decide lags whose CI is entirely above or below the threshold
            newly_stable = (~decided) & (lo >= min_freq)
            newly_unstable = (~decided) & (hi < min_freq)

            # Record decisions
            if newly_stable.any():
                decided[newly_stable] = True
                stable_flag[newly_stable] = True
                decision_b[newly_stable] = b
            if newly_unstable.any():
                decided[newly_unstable] = True
                stable_flag[newly_unstable] = False
                decision_b[newly_unstable] = b

    # finalize trials per lag
    final_trials = np.where(decision_b > 0, decision_b, min(B, len(Xb))).astype(float)
    freq = select_counts / np.maximum(final_trials, 1.0)
    top_freq = top_rank_counts / B

    out = pd.DataFrame(
        {
            "freq": freq,
            "top_freq": top_freq,
            "p_base": base["p"].to_numpy(),
            "t_base": base["t"].to_numpy(),
            "decided": decided,
            "trials": final_trials,
        },
        index=pd.Index(range(1, max_lag + 1), name="lag"),
    )

    # Optional: end-of-pipeline FDR on base sample p's (for reporting)
    if use_fdr_end:
        reject, p_fdr, _, _ = multipletests(
            out["p_base"].to_numpy(), alpha=alpha, method="fdr_bh"
        )
        out["p_fdr"] = p_fdr
        out["reject_base_fdr"] = reject
    else:
        out["p_fdr"] = np.nan
        out["reject_base_fdr"] = False

    out["stable"] = out["freq"] >= min_freq
    if use_fdr_end:
        out["selected"] = out["stable"] & out["reject_base_fdr"]
    else:
        out["selected"] = out["stable"] & (out["p_base"] < alpha)

    return out.sort_values(
        ["selected", "freq", "top_freq"], ascending=[False, False, False]
    )
