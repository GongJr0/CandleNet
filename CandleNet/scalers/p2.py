from __future__ import annotations
import numpy as np
from typing import Optional
from CandleNet.cache.p2_cache import P2Cache


class P2Scaler:
    """Robust Scaler using P^2 estimated median and IQR through online data."""

    def __init__(self, data: np.ndarray, name: str):
        self.data = data
        self.name = name
        self._last_p: Optional[float] = None
        self._last_med: Optional[float] = None
        self._last_iqr: Optional[float] = None
        self._last_eps: Optional[float] = None

    def _init_from_cache(self, p: float) -> tuple[int, list[float], list[int], float]:
        """Return (N, q[5], n[5], eps). n[0] and n[4] are derived as 0 and N-1 respectively."""
        with P2Cache() as c:
            resp = c.fetch(self.name, p)
        if resp is None:
            raise KeyError("no cached state")
        N: int = int(resp["N"])
        q: list[float] = list(resp["q"])  # q0..q4
        n: list[int] = resp["n"]
        eps: float = float(resp["eps"])

        return N, q, n, eps

    def _bulk_init_from_data(
        self, p: float
    ) -> tuple[int, list[float], list[int], float]:
        """Initialize from the entire self.data in a leakage-safe way (sorted order)."""
        data = np.asarray(self.data, dtype=float).ravel()
        if data.size < 5:
            raise ValueError("Data must have at least 5 samples to initialize P^2.")
        data_sorted = np.sort(data)
        N = int(data_sorted.size)
        # marker heights using *quantiles* (not percentiles)
        q0 = data_sorted[0]
        q4 = data_sorted[-1]
        q1 = float(np.quantile(data_sorted, p / 2.0, method="linear"))  # type: ignore
        q2 = float(np.quantile(data_sorted, p, method="linear"))  # type: ignore
        q3 = float(np.quantile(data_sorted, (1.0 + p) / 2.0, method="linear"))  # type: ignore
        q = [q0, q1, q2, q3, q4]
        # integer marker positions (0-based ranks)
        span = N - 1
        n0 = 0
        n1 = int(round((p / 2.0) * span))
        n2 = int(round(p * span))
        n3 = int(round(((1.0 + p) / 2.0) * span))
        n4 = span
        n = [n0, n1, n2, n3, n4]
        eps = 1e-6
        # persist initial state
        with P2Cache() as c:
            c.insert(
                self.name, p, N, q, [n1, n2, n3], eps
            )  # assumes insert replaces/upserts
        return N, q, n, eps

    @staticmethod
    def _p2_update_one(
        x: float, p: float, N: int, q: list[float], n: list[int]
    ) -> tuple[int, list[float], list[int]]:
        """
        One-step P^2 update using 0-based integer marker positions and float "desired positions".
        Returns updated (N, q, n).
        """
        # locate k interval
        if x < q[0]:
            q[0] = x
            k = 0
        elif x >= q[4]:
            q[4] = x
            k = 3
        else:
            # find k such that q[k] <= x < q[k+1]
            k = 0
            while k < 3 and x >= q[k + 1]:
                k += 1

        # increment sample count
        N += 1

        # shift integer positions for markers above x
        for i in range(k + 1, 5):
            n[i] += 1

        # desired increments are fixed
        dn = [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0]
        # desired positions (0-based) at current N
        base = 0.0
        span = float(N - 1)
        np_des = [
            base + dn[0] * span,
            base + dn[1] * span,
            base + dn[2] * span,
            base + dn[3] * span,
            base + dn[4] * span,
        ]

        # adjust interior markers i=1..3
        for i in (1, 2, 3):
            d = np_des[i] - n[i]
            s = 0
            if d >= 1.0 and (n[i + 1] - n[i]) > 1:
                s = 1
            elif d <= -1.0 and (n[i] - n[i - 1]) > 1:
                s = -1
            if s != 0:
                # parabolic prediction
                num = (n[i] - n[i - 1] + s) * (q[i + 1] - q[i]) / np.where(
                    (n[i + 1] - n[i]) != 0, (n[i + 1] - n[i]), 1e-6
                ) + (n[i + 1] - n[i] - s) * (q[i] - q[i - 1]) / (n[i] - n[i - 1])
                denom: float = n[i + 1] - n[i - 1]
                if denom == 0.0:
                    denom = 1e-6  # avoid div0; should not happen

                qip = q[i] + (s * num) / denom
                # monotonic fallback
                if q[i - 1] < qip < q[i + 1]:
                    q[i] = qip  # type: ignore
                else:
                    q[i] += s * (q[i + s] - q[i]) / (n[i + s] - n[i])
                n[i] += s

        return N, q, n

    def fit(self, p: float) -> None:
        # normalize p
        if not (0.0 < p < 1.0):
            raise ValueError("p must be in (0,1)")
        data = np.asarray(self.data, dtype=float)
        stream = data.ravel()
        total = int(stream.size)

        # try resume from cache
        try:
            N, q, n, eps = self._init_from_cache(p)
            if N < 5:
                # cache is incomplete/corrupt; discard and re-init
                N, q, n, eps = self._bulk_init_from_data(p)
                start = 0
            else:
                # validate: N cannot exceed data length
                if N > total:
                    # shrink N to available data (conservative)
                    N = total
                start = N
        except KeyError:
            # no cache: bulk-init from all available data
            N, q, n, eps = self._bulk_init_from_data(p)
            start = 0  # we already used all data for init; will still stream to be consistent

        # stream remaining samples (if any) using true P^2 updates
        for idx in range(start, total):
            x = float(stream[idx])
            N, q, n = self._p2_update_one(x, p, N, q, n)

        # persist final state
        with P2Cache() as c:
            # store n1...n3 only (0-based n[1:4]); eps persisted for reproducibility
            c.insert(self.name, p, N, q, [n[1], n[2], n[3]], eps)

    def estimates(self, p: float) -> tuple[float, float, float]:
        """
        Return the three interior marker estimates for the given quantile p.

        These correspond to the target quantile p (middle marker q[2]) and its companions at
        p/2 (q[1]) and (1+p)/2 (q[3]) as tracked by the P^2 algorithm.

        Raises KeyError if the persisted state for (self.name, p) is not found,
        instructing the caller to run fit(p) first.
        """
        _, q, _, _ = self._init_from_cache(p)
        return q[1], q[2], q[3]

    def median_IQR(self) -> tuple[float, float]:
        """
        Return the median (p=0.5) and its companion quantiles at 0.25 and 0.75.

        This calls estimates(0.5) and returns the triplet (q1, q2, q3) where q2 is the median.

        Raises KeyError if the persisted state for p=0.5 is not found,
        instructing the caller to run fit(0.5) first.
        """
        est = self.estimates(0.5)
        med = est[1]
        IQR = est[2] - est[0]
        return med, IQR

    def fit_transform(self, p: float = 0.5, clip: Optional[float] = None) -> np.ndarray:
        """
        One-call: fit the P^2 state for quantile p on self.data, then return robust-scaled data.

        Scaling uses median/IQR from the three interior markers for this p:
            z = (x - median) / max(IQR, eps)

        Args:
            p: target quantile to track (default 0.5 for median).
            clip: optional symmetric clip value; if provided, output is clipped to [-clip, clip].

        Returns:
            Scaled numpy array with the same shape as self.data.
        """
        self.fit(p)
        # estimates returns (q1, q2, q3) where q2 is the target quantile (median when p=0.5)
        q1, q2, q3 = self.estimates(p)
        # fetch eps from cache to ensure consistency with persisted state
        try:
            _, _, _, eps = self._init_from_cache(p)
        except KeyError:
            eps = 1e-6

        iqr = max(q3 - q1, float(eps))
        med = float(q2)

        # remember last-used params for fast scale()
        self._last_p = float(p)
        self._last_med = med
        self._last_iqr = iqr
        self._last_eps = float(eps)

        z = (np.asarray(self.data, dtype=float) - med) / iqr
        if clip is not None:
            z = np.clip(z, -float(clip), float(clip))
        return z

    def scale(self, x: float, p: float = 0.5, clip: Optional[float] = None) -> float:
        """
        Scale a single value x using the most recently fitted (or cached) median/IQR for quantile p.

        If the instance has just run fit_transform (or fit) with the same p, cached parameters are used.
        Otherwise, this will read the persisted state for (self.name, p).

        Args:
            x: value to scale
            p: quantile track to use (default 0.5 for median-based robust scaling)
            clip: optional symmetric clip to [-clip, clip]

        Returns:
            The robust-scaled value.
        """
        med: Optional[float] = None
        iqr: Optional[float] = None
        eps: float = 1e-6

        if self._last_p is not None and abs(float(p) - float(self._last_p)) < 1e-12:
            med = self._last_med
            iqr = self._last_iqr
            eps = float(self._last_eps or eps)  # type: ignore[assignment]
        else:
            # Pull parameters from persisted state
            try:
                q1, q2, q3 = self.estimates(p)
                _, _, _, eps_cache = self._init_from_cache(p)
                eps = float(eps_cache)
                med = float(q2)
                iqr = float(q3 - q1)
            except KeyError as e:
                raise RuntimeError(
                    "P2Scaler is not fitted for this (name, p). Call fit(p) first."
                ) from e

        assert iqr is not None
        assert med is not None

        iqr = max(float(iqr), float(eps))
        z = (float(x) - float(med)) / iqr

        if clip is not None:
            z = float(np.clip(z, -float(clip), float(clip)))
        return z
