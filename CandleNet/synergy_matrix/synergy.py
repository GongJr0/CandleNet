from CandleNet.utils import get_lib
import pandas as pd
import numpy as np
import requests
from io import StringIO
import yfinance as yf  # type: ignore[import-untyped]
from CandleNet.cache.synergy_cache import CorrType, YfCache, CorrCache, McapCache
import matplotlib.pyplot as plt
import seaborn as sns
from CandleNet.utils import (
    matrix_minmax,
    uptri_vals,
    FRAME,
    signed_uniformize,
    uptri_abs_var,
)
from typing import cast


def gspc_sector() -> pd.DataFrame:
    try:
        import lxml
    except ImportError:
        yn = input("lxml is required to fetch S&P 500 list. Install it now? (y/n): ")
        if yn.lower() in ["y", "yes"]:
            get_lib("lxml")
        else:
            raise ImportError(
                "lxml is required to fetch S&P 500 list. "
                "Please install it and try again."
            )

    try:
        import html5lib
    except ImportError:
        yn = input(
            "html5lib is required to fetch S&P 500 list. Install it now? (y/n): "
        )
        if yn.lower() in ["y", "yes"]:
            get_lib("html5lib")
        else:
            raise ImportError(
                "html5lib is required to fetch S&P 500 list. "
                "Please install it and try again."
            )

    table_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(table_url, headers=headers)
    if resp.status_code != requests.codes.ok:
        raise ConnectionError(f"Failed to fetch S&P 500 list: {resp.status_code}")
    tables = pd.read_html(StringIO(resp.text))
    df = tables[0].rename(
        columns={
            "GICS Sector": "sec",
            "GICS Sub-Industry": "subsec",
            "Symbol": "symbol",
        }
    )
    df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)
    return df[["symbol", "sec", "subsec"]].set_index("symbol")


INDEX_LIST = ["GSPC"]


class Synergy:
    def __init__(self):
        self.index = "GSPC"

        self._sector_info = None
        self._yf_data = None
        self._tickers = None

    def get_data_yf(self) -> pd.DataFrame:
        with YfCache() as c:
            if (df := c.fetch(self.tickers)) is not None:
                return df

            data = yf.download(
                self.tickers, period="1y", interval="1d", auto_adjust=True
            )
            c.insert(data)
        return data

    def sector_mcaps(self) -> dict[str, float]:
        sectors = self.sectoral_indices()
        sector_mcaps = {}
        for sector in sectors:
            total_mcap = 0.0
            for ticker in sectors[sector]:
                try:
                    with McapCache() as c:
                        if (mcap := c.fetch(ticker)) is not None:
                            total_mcap += mcap
                        else:
                            info = yf.Ticker(ticker).info
                            mcap = info.get("marketCap", 0.0) or 0.0
                            c.insert(ticker, mcap)
                            total_mcap += mcap
                except Exception:
                    continue
            sector_mcaps[sector] = total_mcap
        return sector_mcaps

    def log_return_and_vol_corr(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        price = self.yf_data["Close"]
        log_p: pd.DataFrame = np.log(price)  # type: ignore
        ret = log_p.diff(7).dropna(how="any")
        vol = (log_p.rolling(7).std() * np.sqrt(7)).dropna(how="any")

        return ret.corr(), vol.corr()

    def volume_corr(self) -> pd.DataFrame:
        vol = self.yf_data["Volume"]
        minmax_volume = (vol - vol.min()) / (vol.max() - vol.min())
        return minmax_volume.rolling(7).dropna(how="any").mean().corr()

    def sectoral_indices(self) -> dict[str, list[str]]:
        sectors = self.sector_info["sec"].unique()
        return {
            s: self.sector_info.loc[self.sector_info["sec"] == s].index.tolist()
            for s in sectors
        }

    def _sector_portfolio(self) -> dict[str, dict[str, float]]:
        sectors = self.sectoral_indices()
        sector_portfolio = {}
        for sector in sectors:
            mcap = {}
            total_mcap = 0.0
            for ticker in sectors[sector]:
                try:
                    with McapCache() as c:
                        if (mcap_val := c.fetch(ticker)) is not None:
                            mcap[ticker] = mcap_val
                            total_mcap += mcap_val
                            continue

                        info = yf.Ticker(ticker).info
                        mcap[ticker] = info.get("marketCap", 0.0) or 0.0
                        c.insert(ticker, mcap[ticker])
                        total_mcap += mcap[ticker]
                except Exception:
                    mcap[ticker] = 0

            comp_weight = {
                k: (v / total_mcap if total_mcap > 0 else 0) for k, v in mcap.items()
            }
            sector_portfolio[sector] = comp_weight
        return sector_portfolio

    def _sector_log_returns_and_vol(self) -> dict[str, tuple[pd.Series, pd.Series]]:
        sectors = self.sectoral_indices()
        sector_returns = {}
        price = self.yf_data["Close"]
        log_p: pd.DataFrame = np.log(price)  # type: ignore
        ret = log_p.diff(7).dropna(how="any")

        sector_portfolio = self._sector_portfolio()
        for sector in sectors:
            weights = sector_portfolio[sector]
            valid_tickers = [
                t
                for t in sectors[sector]
                if t in ret.columns and not ret[t].isna().all()
            ]
            if not valid_tickers:
                continue
            weighted_ret: pd.Series = pd.Series(
                sum(ret[t] * weights.get(t, 0) for t in valid_tickers)
            )
            weighted_vol: pd.Series = (
                weighted_ret.rolling(7).std() * np.sqrt(7)
            ).dropna(how="any")

            sector_returns[sector] = weighted_ret, weighted_vol
        return sector_returns

    def _sector_volume(self) -> dict[str, pd.Series]:
        sectors = self.sectoral_indices()
        vol = self.yf_data["Volume"]
        minmax_volume = (vol - vol.min()) / (vol.max() - vol.min())
        rolling_vol = minmax_volume.rolling(7).mean().dropna(how="any")

        sector_volume = {}
        for sector in sectors:
            valid_tickers = [
                t
                for t in sectors[sector]
                if t in rolling_vol.columns and not rolling_vol[t].isna().all()
            ]
            if not valid_tickers:
                continue
            sector_volume[sector] = rolling_vol[valid_tickers].mean(axis=1)
        return sector_volume

    def sector_return_and_vol_corr(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        with CorrCache() as c:
            ret, vol = None, None
            if (ret_resp := c.fetch(self.sectors, CorrType.RETURN)) is not None:
                ret = ret_resp

            if (vol_resp := c.fetch(self.sectors, CorrType.VOLATILITY)) is not None:
                vol = vol_resp

            if isinstance(ret, pd.DataFrame) and isinstance(vol, pd.DataFrame):
                return ret, vol

            sector_returns = self._sector_log_returns_and_vol()
            if isinstance(ret, pd.DataFrame) and (vol is None):
                vol_df = pd.DataFrame({s: sector_returns[s][1] for s in sector_returns})
                c.insert(vol_df.corr(), CorrType.VOLATILITY)
                return ret, vol_df.corr()
            elif isinstance(vol, pd.DataFrame) and (ret is None):
                ret_df = pd.DataFrame({s: sector_returns[s][0] for s in sector_returns})
                c.insert(ret_df.corr(), CorrType.RETURN)
                return ret_df.corr(), vol

            vol_df = pd.DataFrame({s: sector_returns[s][1] for s in sector_returns})
            ret_df = pd.DataFrame({s: sector_returns[s][0] for s in sector_returns})

            ret_corr = ret_df.corr()
            vol_corr = vol_df.corr()

            c.insert(ret_corr, CorrType.RETURN)
            c.insert(vol_corr, CorrType.VOLATILITY)
            return ret_corr, vol_corr

    def sector_vol_corr(self) -> pd.DataFrame:
        with CorrCache() as c:
            if (vol_resp := c.fetch(self.sectors, CorrType.VOLUME)) is not None:
                return vol_resp

            sector_returns = self._sector_volume()
            vol_df = pd.DataFrame(sector_returns)
            vol_corr = vol_df.corr()
            c.insert(vol_corr, CorrType.VOLUME)
            return vol_corr

    @staticmethod
    def _upper_triangle_vals(
        matrix: pd.DataFrame, include_diagonal: bool = False
    ) -> np.ndarray:
        A = matrix.to_numpy()
        k = 0 if include_diagonal else 1
        iu = np.triu_indices_from(A, k=k)
        vals = A[iu]
        vals = vals[np.isfinite(vals)]
        return vals

    def _compute_weights(
        self, risk_aversion: float, w_Q_fixed: float = 0.2, temp: float | None = None
    ) -> np.ndarray:

        S_R, S_V = map(signed_uniformize, self.sector_return_and_vol_corr())
        rho = float(np.clip(risk_aversion, 0.0, 1.0))
        if temp is not None:
            z = (rho - 0.5) / max(temp, 1e-12)
            rho = 1.0 / (1.0 + np.exp(-z))

        R = max(0.0, 1.0 - w_Q_fixed)
        wR_base = (1 - rho) * R
        wV_base = rho * R

        sR, sV = uptri_abs_var(S_R), uptri_abs_var(S_V)
        total = sR + sV

        if total <= 1e-12:
            wR_ad, wV_ad = R * 0.5, R * 0.5
        else:
            wR_ad = (sR / total) * R
            wV_ad = (sV / total) * R

        lamb = float(np.clip(w_Q_fixed, 0.0, 1.0))
        w_R = (1 - lamb) * wR_base + lamb * wR_ad
        w_V = (1 - lamb) * wV_base + lamb * wV_ad
        w_Q = float(np.clip(w_Q_fixed, 0.0, 1.0))

        w = np.clip(np.array([w_R, w_V, w_Q]), 0.0, 1.0)
        s = np.sum(w)
        if s <= 1e-12:
            w = np.array([0.4, 0.4, 0.2], dtype=float)
        else:
            w = w / s

        return w

    def signed_synergy(
        self,
        weights: tuple[float, float, float] | None = None,
        adaptive: bool = True,
        risk_aversion: float = 0.5,
        temperature: float | None = None,
        w_Q_fixed: float = 0.2,
        diag: float = 1.0,
    ) -> pd.DataFrame:

        R, V = self.sector_return_and_vol_corr()
        Q = self.sector_vol_corr()

        assert (
            list(R.index)
            == list(R.columns)
            == list(V.index)
            == list(V.columns)
            == list(Q.index)
            == list(Q.columns)
        )

        SR = signed_uniformize(R)
        SV = signed_uniformize(V)
        SQ = signed_uniformize(Q)

        if weights is None and adaptive:
            wR, wV, wQ = self._compute_weights(risk_aversion, w_Q_fixed, temperature)

        elif weights is not None and adaptive:
            wR, wV, wQ = weights

            vR = np.var(np.abs(uptri_vals(SR)), ddof=1)
            vV = np.var(np.abs(uptri_vals(SV)), ddof=1)
            vQ = np.var(np.abs(uptri_vals(SQ)), ddof=1)
            vv = np.array([vR, vV, vQ])
            if np.sum(vv) > 0:
                wR, wV, wQ = (vv / vv.sum()).tolist()

        elif weights is not None and not adaptive:
            wR, wV, wQ = weights

        else:
            raise ValueError(
                "Either weights must be provided or adaptive must be True."
            )

        S = wR * SR + wV * SV + wQ * SQ
        A = S.to_numpy(dtype=float, copy=True)
        A = 0.5 * (A + A.T)
        A = np.clip(A, -1, 1)
        np.fill_diagonal(A, diag)
        return pd.DataFrame(A, index=R.index, columns=R.columns)

    def sector_synergy(self) -> pd.Series:
        s = self.signed_synergy()
        diag = np.eye(s.shape[0]).astype(bool)
        return s[~diag.astype(bool)].mean()

    def synergy_describe(self) -> dict[str, FRAME]:
        s = self.signed_synergy()
        eye = np.eye(s.shape[0]).astype(bool)
        s[eye] = np.nan

        sec = self.sector_synergy()
        base_desc = s.describe()

        top_3 = sec.nlargest(3)
        bottom_3 = sec.nsmallest(3)

        # Per Sector Quartiles
        index = []
        data: dict[str, list] = {
            "MIN": [],
            "Q1": [],
            "MEDIAN": [],
            "Q3": [],
            "MAX": [],
        }
        for st in s.index:
            index.append(st)
            data["MIN"].append(s[st].min())
            data["Q1"].append(s[st].quantile(0.25))
            data["MEDIAN"].append(s[st].median())
            data["Q3"].append(s[st].quantile(0.75))
            data["MAX"].append(s[st].max())
        qdf = pd.DataFrame(data, index=index)

        return {
            "describe": base_desc,
            "sectors_mean": sec,
            "top_3": top_3,
            "bottom_3": bottom_3,
            "quartiles": qdf,
        }

    def synergy_heatmap(self) -> None:
        s = self.signed_synergy()
        plt.figure(figsize=(10, 8))
        sns.heatmap(s, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Sector Synergy Heatmap")
        plt.tight_layout()
        plt.show(block=False)

    @staticmethod
    def plot_triu_hist(
        matrix: pd.DataFrame,
        bins: int = 30,
        include_diagonal: bool = False,
        clip_bounds: tuple[float, float] | None = (0.0, 1.0),
        thresholds: list[float] | None = None,
        title: str = "Upper-triangle correlation distribution",
    ) -> None:
        vals = Synergy._upper_triangle_vals(matrix, include_diagonal)
        if clip_bounds is not None:
            L, U = clip_bounds
            vals = np.clip(vals, L, U)
        plt.figure(figsize=(9, 5))
        plt.hist(vals, bins=bins, density=True, alpha=0.7)
        if thresholds:
            for t in thresholds:
                plt.axvline(t, linestyle="--")
        plt.xlabel("Correlation")
        plt.ylabel("Density")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)

    @staticmethod
    def plot_triu_kde(
        matrix: pd.DataFrame,
        include_diagonal: bool = False,
        clip_bounds: tuple[float, float] | None = (-1.0, 1.0),
        bandwidth: float | None = None,
        title: str = "Upper-triangle KDE (correlations)",
    ) -> None:
        vals = Synergy._upper_triangle_vals(matrix, include_diagonal)
        if clip_bounds is not None:
            L, U = clip_bounds
            vals = np.clip(vals, L, U)
        x_min, x_max = (vals.min(), vals.max()) if clip_bounds is None else clip_bounds
        xs = np.linspace(x_min, x_max, 512)

        # Try SciPy KDE; fallback to a simple Gaussian smoother if SciPy isn't available
        try:
            from scipy.stats import gaussian_kde

            kde = (
                gaussian_kde(vals, bw_method=bandwidth)
                if bandwidth
                else gaussian_kde(vals)
            )
            ys = kde(xs)
        except Exception:
            # Fallback: manual Gaussian smoothing of a fine histogram (not as good as SciPy but works)
            hist_y, hist_x = np.histogram(
                vals, bins=100, range=(x_min, x_max), density=True
            )
            cx = 0.5 * (hist_x[:-1] + hist_x[1:])
            bw = (
                (0.9 * np.std(vals, ddof=1) * (len(vals) ** (-1 / 5)))
                if bandwidth is None
                else float(bandwidth)
            )
            # Ensure a small positive bandwidth
            bw = max(bw, 1e-3)
            # Gaussian kernel smoothing
            diffs = xs[:, None] - cx[None, :]
            ys = np.exp(-0.5 * (diffs / bw) ** 2).dot(hist_y) / (
                np.sqrt(2 * np.pi) * bw
            )

        plt.figure(figsize=(9, 5))
        plt.plot(xs, ys)
        plt.xlabel("Correlation")
        plt.ylabel("Density")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)

    @staticmethod
    def plot_triu_ecdf(
        matrix: pd.DataFrame,
        include_diagonal: bool = False,
        clip_bounds: tuple[float, float] | None = (-1.0, 1.0),
        title: str = "Upper-triangle ECDF",
    ) -> None:
        vals = Synergy._upper_triangle_vals(matrix, include_diagonal)
        if clip_bounds is not None:
            L, U = clip_bounds
            vals = np.clip(vals, L, U)
        x = np.sort(vals)
        y = np.linspace(0, 1, len(x), endpoint=True)
        plt.figure(figsize=(9, 5))
        plt.plot(x, y)
        plt.xlabel("Correlation")
        plt.ylabel("ECDF")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)

    @property
    def sector_info(self) -> pd.DataFrame:
        if self._sector_info is None:
            self._sector_info = gspc_sector()
        return self._sector_info

    @property
    def yf_data(self) -> pd.DataFrame:
        if self._yf_data is None:
            self._yf_data = self.get_data_yf()
        return self._yf_data

    @property
    def tickers(self) -> list:
        if self._tickers is None:
            self._tickers = self.sector_info.index.to_list()
        return self._tickers

    @property
    def sectors(self) -> list:
        return self.sector_info["sec"].unique().tolist()
