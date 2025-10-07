import pandas as pd
import numpy as np


from typing import Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .ticker_id import TickerID
from CandleNet.ticker.ticker import get_tickers
from CandleNet.autoreg import select_lags, get_lag, get_lag_cols
from CandleNet.utils import demean, get_integer_lag

from CandleNet import lag_config


def _get_lag_cols(df) -> list[str]:
    return [c for c in df.columns if c.startswith("T-")]


def _row_to_sparse_ids_vals(
    row_vals: np.ndarray, lag_ids: np.ndarray, eps: float = 1e-8
):
    m = np.abs(row_vals) > eps
    ids = lag_ids[m].astype(np.int64)  # [K_i]
    vals = row_vals[m].astype(np.float32)  # [K_i]
    return ids, vals


def _pad_batch(list_ids, list_vals):
    B = len(list_ids)
    K = max((len(x) for x in list_ids), default=1)
    lag_ids = torch.full((B, K), 0, dtype=torch.long)
    lag_vals = torch.zeros((B, K), dtype=torch.float32)
    lag_mask = torch.ones((B, K), dtype=torch.bool)  # True = pad

    for b, (ids, vals) in enumerate(zip(list_ids, list_vals)):
        k = len(ids)
        if k:
            lag_ids[b, :k] = torch.from_numpy(ids)
            lag_vals[b, :k] = torch.from_numpy(vals)
            lag_mask[b, :k] = False
    return lag_ids, lag_vals, lag_mask


class LagFormatter:
    def __init__(self):
        self._selected_lags = None
        self._tickers = None
        self._lookup = None

    def _insert_or_fetch(self, ticker: str) -> int:
        with self.lookup as l:  # noqa: E741
            if (res := l.fetch(ticker)) is None:
                l.insert(ticker)
                res = l.fetch(ticker)
            assert isinstance(res, int)
            return res

    def _init_lags(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        tickers = self.tickers
        lags = self.selected_lags

        lag_data = {
            t: get_lag(demean(tickers[t].log_returns), lags[t], include_t0=True)
            for t in tickers
        }
        lag_features = {t: lag_data[t].drop(columns=["T-0"]) for t in tickers}
        lag_targets = {t: lag_data[t]["T-0"].copy().to_frame() for t in tickers}

        for t in tickers:
            tid = self._insert_or_fetch(t)
            lag_data[t]["ticker"] = tid
            lag_targets[t]["ticker"] = tid

        lag_df = pd.concat(lag_data.values())
        target_df = pd.concat(lag_targets.values())

        try:
            max_lag = int(lag_config()["maxLag"])
        except (KeyError, ValueError):
            raise ValueError(
                "maxLag in featureConfig.yaml must be specified as an integer."
            )

        for pad_cols in get_lag_cols(max_lag):
            if pad_cols not in lag_df.columns:
                lag_df[pad_cols] = np.nan

        lag_df = lag_df[sorted(lag_df.columns, key=get_integer_lag)]  # type: ignore[arg-type]
        lag_df = lag_df.fillna(self.PAD)
        target_df = target_df[sorted(target_df.columns, key=get_integer_lag)]  # type: ignore[arg-type]
        return lag_df, target_df

    @staticmethod
    def get_pad_mask(l: int, pos_idx: Iterable[int]):
        mask = np.zeros(l)
        for idx in pos_idx:
            if idx < l:
                mask[idx] = 1
        return mask.astype(np.bool)

    @property
    def PAD(self):
        return 0.0

    @property
    def tickers(self):
        if self._tickers is None:
            self._tickers = get_tickers()
        return self._tickers

    @property
    def selected_lags(self):
        tickers = get_tickers()
        if self._selected_lags is None:
            self._selected_lags = {
                t: select_lags(demean(tickers[t].log_returns), engine="numba")
                for t in tickers
            }

        return self._selected_lags

    @property
    def lookup(self) -> TickerID:
        if self._lookup is None:
            self._lookup = TickerID()
        return self._lookup


class LagDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        y: pd.Series | np.ndarray,
        dense_cols: list[str] | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.lag_cols = _get_lag_cols(df)
        self.lag_ids = np.asarray(
            [get_integer_lag(col) for col in self.lag_cols], dtype=np.int64
        )
        self.dense_cols = dense_cols or []

        self._lags_np = self.df[self.lag_cols].to_numpy(dtype=np.float32)
        self._dense_np = (
            self.df[self.dense_cols].to_numpy(dtype=np.float32)
            if self.dense_cols
            else None
        )
        self._ticker_id = df["ticker"].to_numpy(dtype=np.int64)
        self._y = np.asarray(y, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        vals = self._lags_np[i]
        ids, v = _row_to_sparse_ids_vals(vals, self.lag_ids)
        sample = {
            "ticker_id": self._ticker_id[i],
            "lag_ids": ids,
            "lag_vals": v,
        }
        if self._dense_np is not None:
            sample["X_dense"] = self._dense_np[i]
        sample["y"] = self._y[i][1]
        return sample


def collate_fn(batch: list[dict]) -> dict:
    ls_ids = [b["lag_ids"] for b in batch]
    ls_vals = [b["lag_vals"] for b in batch]
    lag_ids, lag_vals, lag_mask = _pad_batch(ls_ids, ls_vals)
    y = [b["y"] for b in batch]

    out = {
        "lag_ids": lag_ids,  # (B, K)
        "lag_vals": lag_vals,  # (B, K)
        "lag_mask": lag_mask,  # (B, K) True = pad
        "ticker_id": torch.as_tensor([b["ticker_id"] for b in batch], dtype=torch.long),
    }
    if "X_dense" in batch[0]:
        out["X_dense"] = torch.as_tensor(
            [b["X_dense"] for b in batch], dtype=torch.float32
        )
    out["y"] = torch.as_tensor(y, dtype=torch.float32)
    return out


class LagSetEncoder(nn.Module):
    def __init__(self, d_lag: int = 16, pad_idx: int = 0, agg: str = "sum"):
        try:
            max_lag = int(lag_config()["maxLag"])
        except (KeyError, ValueError):
            raise ValueError(
                "maxLag in featureConfig.yaml must be specified as an integer."
            )

        super().__init__()
        self.emb = nn.Embedding(max_lag + 1, d_lag, padding_idx=pad_idx)  # id 0 is PAD
        self.agg = agg

    def forward(self, lag_ids, lag_vals, lag_mask):
        E = self.emb(lag_ids)  # [B,K,d]
        W = E * lag_vals.unsqueeze(-1)  # weight by values
        W = W.masked_fill(lag_mask.unsqueeze(-1), 0.0)
        S = W.sum(dim=1)  # [B,d]
        if self.agg == "mean":
            denom = (~lag_mask).sum(dim=1).clamp_min(1).unsqueeze(-1)
            S = S / denom
        return S


class CandleNetHead(nn.Module):
    def __init__(self, num_tickers: int, d_lag=16, d_ticker=32, d_dense=32):
        try:
            max_lag = int(lag_config()["maxLag"])
        except (KeyError, ValueError):
            raise ValueError(
                "maxLag in featureConfig.yaml must be specified as an integer."
            )

        super().__init__()
        self.ticker_emb = nn.Embedding(
            num_tickers + 1, d_ticker
        )  # assume ids >=1, 0 unused here
        self.lag_enc = LagSetEncoder(d_lag=d_lag)
        self.mlp = nn.Sequential(
            nn.Linear(d_dense + d_lag + d_ticker, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, X_dense, lag_ids, lag_vals, lag_mask, ticker_id):
        z = [
            self.lag_enc(lag_ids, lag_vals, lag_mask),  # [B,d_lag]
            self.ticker_emb(ticker_id),  # [B,d_ticker]
        ]
        if X_dense is not None:
            z.insert(0, X_dense)  # [B,d_dense]
        return self.mlp(torch.cat(z, dim=-1)).squeeze(-1)


class CandleNetModel(nn.Module):
    def __init__(
        self,
        num_tickers: int,
        d_dense: int,
        d_lag: int = 16,
        d_ticker: int = 12,
        hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lag_enc = LagSetEncoder(d_lag=d_lag, pad_idx=0, agg="sum")
        self.ticker_emb = nn.Embedding(
            num_tickers + 1, d_ticker
        )  # ids start at 1; keep 0 unused
        layers: list[nn.Module] = []
        in_dim = d_dense + d_lag + d_ticker
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        *,
        lag_ids: torch.Tensor,  # [B, K] Long
        lag_vals: torch.Tensor,  # [B, K] Float
        lag_mask: torch.Tensor,  # [B, K] Bool
        ticker_id: torch.Tensor,  # [B] Long
        X_dense: Optional[torch.Tensor] = None,  # [B, d_dense] Float
    ) -> torch.Tensor:
        z = [
            self.lag_enc(lag_ids, lag_vals, lag_mask),  # [B,d_lag]
            self.ticker_emb(ticker_id),
        ]  # [B,d_ticker]
        if X_dense is not None:
            z.insert(0, X_dense)  # [B,d_dense]
        x = torch.cat(z, dim=-1)
        return self.mlp(x).squeeze(-1)  # [B]
