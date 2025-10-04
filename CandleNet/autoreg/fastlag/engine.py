import numpy as np
from numba import njit  # type: ignore[import-untyped]


@njit(cache=True, fastmath=True)
def _ols_hac_beta_t_vectorized(
    y: np.ndarray,
    max_lag: int,
    L: int,
):

    n = y.shape[0]
    beta = np.full(max_lag, np.nan, dtype=np.float64)
    tval = np.full(max_lag, np.nan, dtype=np.float64)
    nobs = np.zeros(max_lag, dtype=np.int64)

    for k in range(1, max_lag + 1):
        N = n - k
        nobs[k - 1] = N
        if N < 10:
            beta[k - 1] = np.nan
            tval[k - 1] = np.nan

        x = y[:N]
        yy = y[k:n]

        xb, yb = 0.0, 0.0
        for i in range(N):
            xb += x[i]
            yb += yy[i]

        xb /= N
        yb /= N

        Sxx, Sxy = 0.0, 0.0
        for i in range(N):
            dx = x[i] - xb
            dy = yy[i] - yb
            Sxx += dx * dx
            Sxy += dx * dy

        if Sxx <= 0.0:
            beta[k - 1] = np.nan
            tval[k - 1] = np.nan

        b = Sxy / Sxx

        s = np.empty(N, dtype=np.float64)
        for i in range(N):
            dx = x[i] - xb
            dy = yy[i] - yb
            u = dy - b * dx
            s[i] = dx * u

        gamma0 = 0.0
        for i in range(N):
            gamma0 += s[i] * s[i]
        gamma0 /= N
        omega = gamma0

        L_eff = L if L < (N - 1) else (N - 1)
        for l in range(1, L_eff + 1):  # noqa[E741]
            w = 1.0 - l / (L + 1.0)
            g = 0.0
            M = N - l
            for i in range(M):
                g += s[i + l] * s[i]
            g /= N
            omega += 2.0 * w * g

        var_b = (N * omega) / (Sxx * Sxx)
        t = b / np.sqrt(var_b) if var_b > 0.0 else np.nan

        beta[k - 1] = b
        tval[k - 1] = t

    return beta, tval, nobs
