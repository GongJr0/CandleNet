from __future__ import annotations
from typing import Generator, Optional, Tuple, Literal
import numpy as np
import pandas as pd
import statsmodels.api as sm  # type: ignore
from statsmodels.stats.multitest import multipletests  # type: ignore
from scipy.stats import norm  # type: ignore


from CandleNet.utils import SERIES
from CandleNet import lag_config, LagConfig


def get_formatted_arr(arr: SERIES) -> pd.DataFrame:
    """
    Convert a 1D series or array-like into a pandas DataFrame with a single column named "T".

    Parameters:
        arr (Series | np.ndarray | array-like): One-dimensional input sequence to convert.

    Returns:
        pd.DataFrame: DataFrame of shape (len(arr), 1) with column "T" containing the input values.

    Raises:
        AssertionError: If `arr` is not one-dimensional.
    """
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
    """
    Compute a small-sample Newey–West (Andrews-style) bandwidth heuristic.

    Parameters:
        n (int): Sample size (number of observations) used to derive the bandwidth.

    Returns:
        bw (int): Bandwidth for Newey–West/HAC estimation, computed as max(1, round(4 * (n/100)^(2/9))).
    """
    bw = int(round(4.0 * (max(n, 2) / 100.0) ** (2.0 / 9.0)))
    return max(bw, 1)


def lag_significance_hac(
    y, max_lag=20, bandwidth: int | Literal["auto"] = "auto", fdr=False, alpha=0.05
):
    """
    Compute HAC-robust lag regressions of y_t on y_{t-k} for k = 1...max_lag and report coefficients and
    test statistics.

    Parameters:
        y: array-like
            Univariate time series.
        max_lag (int):
            Maximum lag k to test (tests 1 through max_lag).
        bandwidth (int or "auto"):
            Newey–West/HAC lag parameter (maxlags) to use for covariance estimation; if "auto",
            a data-dependent heuristic is used.
        fdr (bool):
            If True, apply Benjamini–Hochberg false discovery rate correction to the returned p-values and include
            a `reject_fdr` boolean column.
        alpha (float):
            Significance level used only for the FDR procedure.

    Returns:
        pandas.DataFrame:
            One row per tested lag with columns:
            - "lag": the lag k.
            - "beta": estimated coefficient on y_{t-k}.
            - "t": HAC t-statistic for the lag coefficient.
            - "p": two-sided p-value for the lag coefficient.
            - "n": number of observations used for that regression.
            If `fdr` is True, includes an additional boolean column "reject_fdr" indicating FDR rejections.

    Notes:
        For lags where there are fewer than 10 paired observations, the row contains NaNs for beta, t,
        and p while "n" records the sample size.
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
    block_len: int | Literal["auto"] = "auto",
    bandwidth: int | Literal["auto"] = "auto",
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
    Compute stability-selection metrics for lag significance using circular block bootstrap resampling.

    Parameters:
        y (pd.Series): Time series to test for lagged significance.
        max_lag (int): Maximum lag to evaluate (lags 1...max_lag).
        B (int): Number of bootstrap resamples to draw.
        block_len (int | "auto"): Block length for circular block bootstrap or "auto" to derive from sample size.
        bandwidth (int | "auto"): Newey–West/HAC bandwidth to use for HAC-covariance estimation or "auto" to derive
        from sample size.
        alpha (float): Significance level used when counting a lag as selected on each resample.
        rng (np.random.Generator | None): Random number generator; a default RNG is created when None.
        use_fdr_end (bool): If True, apply Benjamini–Hochberg FDR to base-sample p-values for final reporting/selection.
        min_freq (float): Frequency threshold in [0,1] used to mark a lag as "stable" (freq >= min_freq).
        early_stop (bool): If True, allow early stopping of the bootstrap loop when Wilson intervals
        confidently decide lags.
        b_min (int): Minimum number of bootstrap rounds before performing early-stop checks.
        check_every (int): Interval (in bootstrap rounds) at which early-stop checks are performed.
        conf (float): Confidence level for Wilson intervals used in early stopping (e.g., 0.95).

    Returns:
        pd.DataFrame: Indexed by lag (1...max_lag) with columns:
            - freq: Empirical selection frequency = Pr[lag significant on a CBB resample at level alpha].
            - top_freq: Fraction of resamples where the lag had the smallest p-value (diagnostic).
            - p_base: Base-sample p-value from HAC-robust regression y_t ~ y_{t-k}.
            - t_base: Base-sample t-statistic for the lag coefficient.
            - decided: Boolean indicating whether early stopping finalized the lag's decision.
            - trials: Number of bootstrap rounds used to evaluate the lag (may be <= B if decided early).
            - p_fdr: FDR-adjusted p-value for the base sample (NaN if use_fdr_end is False).
            - reject_base_fdr: Boolean indicating FDR rejection on the base sample (False if use_fdr_end is False).
            - stable: Boolean indicating freq >= min_freq.
            - selected: Boolean indicating final selection (stable AND base-sample significance per FDR or
            alpha when FDR disabled).

    Notes:
        - The function reports stability (how often a lag is significant under resampling) rather than providing
        a calibrated null-distribution for test statistics.
        - Early stopping uses Wilson score intervals on selection frequencies to decide stable/unstable
        lags before exhausting B samples.
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


def _auto_block_len(n: int) -> int:
    """
    Select an automatic block length for block bootstrap based on the series length.

    Parameters:
        n (int): Number of observations in the time series.

    Returns:
        block_len (int): Recommended block length, equal to the greater of 5 and the floor of the cube root of `n`.
    """
    return max(5, int(n ** (1 / 3)))


def _resolve_lag_cfg(params: LagConfig, n: int) -> dict:
    # max_lag tested
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
    msel = params["maxLagsSelected"]
    if isinstance(msel, str) and msel == "auto":
        # simple, conservative default
        msel = max(params["minLagsSelected"], min(5, max_lag))

    assert isinstance(msel, int) and msel >= 0

    return {
        "max_lag": max_lag,
        "bandwidth": int(bw),
        "block_len": int(bl),
        "B": int(B),
        "max_selected": int(msel),
    }


def select_lags(y: pd.Series, _debug: bool = False) -> list:
    """
    Select time-series lags deemed significant by a bootstrap stability procedure combined with HAC-robust lag testing.

    Uses configured selection and stability settings to run the bootstrap-based significance procedure,
    optionally applies FDR adjustment and stability requirements,
    enforces configured minimum and maximum selected lags (augmenting or capping as needed),
    and returns the final sorted selection.

    Returns:
        list: Sorted list of selected lag integers in ascending order. Returns an empty list if
        the input series has fewer than 10 observations.
    """
    params = lag_config()
    n = len(y)
    if n < 10:
        return []

    rand_seed = params.get("randomSeed")
    rng = np.random.default_rng(rand_seed)

    r = _resolve_lag_cfg(params, n)
    # run the test
    res = bootstrapped_significance(
        y,
        max_lag=r["max_lag"],
        B=r["B"],
        block_len=r["block_len"],
        bandwidth=(
            params["hacBandwidth"]
            if params["hacBandwidth"] != "auto"
            else r["bandwidth"]
        ),
        alpha=params["sigLevel"],
        use_fdr_end=(params["selectionMethod"] == "fdrAdjusted"),
        min_freq=params["stabilityFreq"] if params["requireStability"] else 0.0,
        early_stop=params["earlyStop"],
        b_min=params["minBootstrapSamples"],
        check_every=params["stabilityCheckEvery"],
        conf=params["stabilityConfidence"],
        rng=rng,
    )

    # --- Build mask(s)
    if params["selectionMethod"] == "fdrAdjusted":
        base_mask = res["reject_base_fdr"].astype(bool)
    else:  # "rawPval"
        base_mask = res["p_base"] < params["sigLevel"]

    mask = base_mask
    if params["requireStability"]:
        mask = mask & res["stable"].astype(bool)

    selected = res[mask]

    if selected.empty:
        if _debug:
            print("No lags selected by criteria.")
        topn = max(params.get("rankTopN", 0), params["minLagsSelected"])
        selected = res.sort_values(by=["p_base", "freq"], ascending=[True, False]).head(
            topn
        )
        return selected.index.sort_values().tolist()

    # Ensure at least minLagsSelected
    if len(selected) < params["minLagsSelected"]:

        need = params["minLagsSelected"] - len(selected)
        if _debug:
            print(
                f"{len(selected)} lags selected by criteria. Augmenting top {need} "
                f"remaining lags by [ASC] p-value & [DESC] freq."
            )
        # sort remaining by p then freq (desc)
        remaining = res[~mask].sort_values(
            by=["p_base", "freq"], ascending=[True, False]
        )
        selected = pd.concat([selected, remaining.head(need)], axis=0)

    # Enforce maxLagsSelected cap
    if len(selected) > r["max_selected"]:
        if _debug:
            print(
                f"{len(selected)} lags selected by criteria. Capping to top {r['max_selected']} "
                f"by [ASC] p-value & [DESC] freq."
            )
        selected = selected.sort_values(
            by=["p_base", "freq"], ascending=[True, False]
        ).head(r["max_selected"])

    # Return sorted lag index (assumes index is the lag)
    return selected.index.sort_values().tolist()
