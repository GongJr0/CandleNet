from CandleNet import lag_config
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests  # type: ignore[import-untyped]
from .engine import _ols_hac_beta_t_vectorized
from CandleNet import LagConfig
from typing import Any, Generator, Literal, Optional, Tuple


def _is_int_like(x: Any) -> bool:
    if not hasattr(x, "__int__"):
        return False

    return x - int(x) <= 1e-8


def _resolve_lag_cfg(params: LagConfig, n: int) -> dict:
    """
    Resolve a LagConfig mapping into concrete numeric parameters used for lag testing and bootstrapping.

    This converts potentially symbolic or "auto" entries in `params` into integer values appropriate for
    a series of length `n`, applying sensible bounds and heuristics where needed.

    Parameters:
        params (LagConfig): Configuration mapping containing keys:
            - "maxLag": maximum lag to consider (may be numeric or "auto"-like value).
            - "hacBandwidth": Newey–West/HAC bandwidth or "auto".
            - "blockLen": circular block bootstrap block length or "auto".
            - "bootstrapSamples": number of bootstrap replicates or "auto".
            - "maxLagsSelected": cap on number of selected lags or "auto".
            - "minBootstrapSamples", "minLagsSelected": minimums used when resolving "auto".
        n (int): Length of the time series; used to clamp and derive data-dependent defaults.

    Returns:
        dict: A mapping with integer-valued keys:
            - "max_lag": selected max lag (clamped to at least 1 and at most n-2).
            - "bandwidth": resolved HAC bandwidth as an int.
            - "block_len": resolved block length for CBB as an int.
            - "B": number of bootstrap replicates as an int.
            - "max_selected": maximum number of lags to retain as an int (>= 0).
    """
    max_lag = int(params["maxLag"])
    max_lag = max(1, min(max_lag, n - 2))

    # bandwidth
    bw = params["hacBandwidth"]
    if isinstance(bw, str) and bw == "auto":
        bw = _auto_nw_bandwidth(n)

    # block length
    bl = params["blockLen"]
    if isinstance(bl, str) and bl == "auto":
        bl = _auto_block_len(n)

    # bootstrap samples
    B = params["bootstrapSamples"]
    if isinstance(B, str) and B == "auto":
        # heuristic: proportional to tested lags, capped
        B = max(params["minBootstrapSamples"], min(300, 20 * max_lag))

    # max lags selected
    msel_cfg = params["maxLagsSelected"]
    if isinstance(msel_cfg, str) and msel_cfg == "auto":
        msel = max(params["minLagsSelected"], min(5, max_lag))
    else:
        msel = msel_cfg

    assert _is_int_like(msel) and msel >= 0, (
        f"Unsupported maxLagsSelected: {msel_cfg}. "
        f"Must be a non-negative integer or 'auto'."
    )

    return {
        "max_lag": max_lag,
        "bandwidth": int(bw),
        "block_len": int(bl),
        "B": int(B),
        "max_selected": int(msel),
    }


def _auto_nw_bandwidth(n: int) -> int:
    # Andrews(1991)/Newey-West style small-sample bandwidth; keep it tiny for daily data
    # 4 * (n/100)^(2/9), at least 1
    """
    Compute a small-sample Newey–West (Andrews-style) bandwidth heuristic.

    Parameters:
        n (int): Sample size (number of observations) used to derive the bandwidth.

    Returns:
        bw (int): Bandwidth for Newey–West/HAC estimation, computed as max(1, round(4 * (n/100)^(2/9))).
    """
    bw = round(4.0 * (max(n, 2) / 100.0) ** (2.0 / 9.0))
    return max(bw, 1)


def _infer_block_len(n: int) -> int:
    """Heuristic: l = max(5, floor(n**(1/3)))."""
    return max(5, int(np.floor(n ** (1 / 3))))


def _auto_block_len(n: int) -> int:
    """
    Select an automatic block length for block bootstrap based on the series length.

    Parameters:
        n (int): Number of observations in the time series.

    Returns:
        block_len (int): Recommended block length, equal to the greater of 5 and the floor of the cube root of `n`.
    """
    return max(5, int(n ** (1 / 3)))


def cbb_indices(
    n: int,
    B: int,
    block_len: int | Literal["auto"] = "auto",
    rng: Optional[np.random.Generator] = None,
) -> Generator[np.ndarray, None, None]:
    """
    Yield index arrays for circular block bootstrap resampling of a length-n series.

    Parameters:
        n (int): Length of the original time series.
        B (int): Number of bootstrap replicates to generate.
        block_len (int or "auto"): Block length to use for the circular blocks. If "auto",
        a heuristic based on `n` is used.
        rng (np.random.Generator, optional): Random number generator to sample block starting positions. If omitted,
        a default Generator is created.

    Yields:
        np.ndarray: 1-D integer array of length `n` containing indices into the original series
        for one bootstrap replicate.

    Raises:
        ValueError: If `n`, `B`, or resolved `block_len` is not positive.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if B <= 0:
        raise ValueError("B must be positive.")

    if block_len == "auto":
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
    block_len: int | Literal["auto"] = "auto",
    rng: Optional[np.random.Generator] = None,
    return_indices: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Generate B circular block bootstrap (CBB) resamples of a time series, preserving joint structure across columns.

    Parameters:
        X (np.ndarray): Time-series data of shape (n,) or (n, p). 1D inputs are treated as (n, 1).
        B (int): Number of bootstrap replicates to generate.
        block_len (int or "auto"): Block length for the CBB. If "auto", a heuristic block length is chosen.
        rng (np.random.Generator, optional): Random number generator to control reproducibility. If None,
        the default global RNG is used.
        return_indices (bool): If True, also return the index matrix used for resampling.

    Returns:
        Xb (np.ndarray): Bootstrap samples. Shape is (B, n) for 1D inputs or (B, n, p) for 2D inputs.
        Ib (np.ndarray or None): Integer index matrix of shape (B, n) used to form each resample when
        return_indices is True; otherwise None.
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

    if B <= 0:
        raise ValueError("B must be positive.")

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


def vectorized_select_lags(y) -> list[int]:

    params = lag_config()
    _r = _resolve_lag_cfg(params, len(y))

    max_lag = _r["max_lag"]
    bandwidth = _r["bandwidth"]
    B = _r["B"]
    block_len = _r["block_len"]
    max_selected = _r["max_selected"]

    alpha = params["sigLevel"]
    require_stability = params["requireStability"]
    min_freq = params["stabilityFreq"]
    conf = params["stabilityConfidence"]
    early_stop = params["earlyStop"]
    b_min = params["minBootstrapSamples"]
    check_every = params["stabilityCheckEvery"]
    use_fdr_end = params["selectionMethod"] == "fdrAdjusted"
    min_selected = params["minLagsSelected"]
    rank_topn = params["rankTopN"]
    rng = np.random.default_rng(params["randomSeed"])

    yv = np.asarray(y, dtype=np.float64).ravel()
    n = yv.size
    if n < 10:
        return []

    L = _auto_nw_bandwidth(n) if bandwidth == "auto" else int(bandwidth)

    # base-sample
    _, t0, _ = _ols_hac_beta_t_vectorized(yv, max_lag, L)
    p0 = 2.0 * norm.sf(np.abs(t0))
    p0 = np.where(np.isfinite(p0), p0, 1.0)

    # bootstrap
    Xb, _ = cbb_sample(yv, B=B, block_len=block_len, rng=rng)  # (B,n)
    decided = np.zeros(max_lag, bool)
    decision_b = np.full(max_lag, -1, int)
    stable_flag = np.zeros(max_lag, bool)
    select_counts = np.zeros(max_lag, int)
    top_counts = np.zeros(max_lag, int)

    for b in range(1, B + 1):
        if early_stop and decided.all():
            break
        _, t_b, _ = _ols_hac_beta_t_vectorized(
            Xb[b - 1].astype(np.float64, copy=False), max_lag, L
        )
        p_b = 2.0 * norm.sf(np.abs(t_b))

        m = ~decided
        if m.any():
            sel = (p_b[m] < alpha).astype(np.int64)
            select_counts[m] += sel

        if np.isfinite(p_b).any():
            top = int(np.nanargmin(p_b))
            top_counts[top] += 1

        if early_stop and b >= b_min and (b % check_every == 0):
            trials = np.where(decided, np.maximum(decision_b, 1), b).astype(np.float64)
            lo, hi = _wilson_interval(
                select_counts.astype(np.float64), trials, conf=conf
            )
            newly_stable = (~decided) & (lo >= min_freq)
            newly_unstable = (~decided) & (hi < min_freq)
            if newly_stable.any():
                decided[newly_stable] = True
                stable_flag[newly_stable] = True
                decision_b[newly_stable] = b
            if newly_unstable.any():
                decided[newly_unstable] = True
                stable_flag[newly_unstable] = False
                decision_b[newly_unstable] = b

    trials = np.where(decision_b > 0, decision_b, min(B, len(Xb))).astype(np.float64)
    freq = select_counts / np.maximum(trials, 1.0)
    top_freq = top_counts / B

    # Final masks (mirror original)
    if use_fdr_end:
        reject, *_ = multipletests(p0, alpha=alpha, method="fdr_bh")
    else:
        reject = p0 < alpha

    stable = (freq >= min_freq) if require_stability else np.ones_like(freq, bool)
    selected_mask = stable & reject if use_fdr_end else stable & (p0 < alpha)

    # Fallback & caps
    idx = np.arange(1, max_lag + 1)
    selected_idx = idx[selected_mask]

    if selected_idx.size == 0:
        take = max(min_selected, (rank_topn or 0))
        order = np.lexsort((-freq, p0))  # primary p asc, tie by freq desc
        selected_idx = idx[order][:take]

    if selected_idx.size < min_selected:
        need = min_selected - selected_idx.size
        mask_rem = np.ones(max_lag, bool)
        mask_rem[selected_idx - 1] = False
        order = np.lexsort((-freq[mask_rem], p0[mask_rem]))
        add = idx[mask_rem][order][:need]
        selected_idx = np.sort(np.concatenate([selected_idx, add]))

    if selected_idx.size > max_selected:
        order = np.lexsort((-freq[selected_idx - 1], p0[selected_idx - 1]))
        selected_idx = selected_idx[order][:max_selected]

    return np.sort(selected_idx).tolist()
