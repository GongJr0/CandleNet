import yfinance as yf  # type: ignore
import pandas as pd
import numpy as np

from typing import Literal, cast
from .DataAccess.utils import ML_GLOBAL_FLOAT_TYPE


class TickerContainer(yf.Ticker):  # type: ignore
    def __init__(self, ticker: str, data: pd.DataFrame):
        super().__init__(ticker)
        self.ticker = ticker
        self.data = data

    def market_propensity_to_trade(self) -> float:
        span = np.ceil(np.sqrt(len(self.data)))
        ewma_volume = self.data['Volume'].ewm(span=span).mean().mean()
        min_volume = self.data['Volume'].quantile(0.05)
        max_volume = self.data['Volume'].quantile(0.95)

        normalized_propensity = (ewma_volume - min_volume) / (max_volume - min_volume)
        clipped = np.clip(normalized_propensity, 0, 1)
        return ML_GLOBAL_FLOAT_TYPE(clipped)

    def rsi_ewma(self) -> float:
        span = np.ceil(np.sqrt(len(self.data)))
        gain: pd.Series = cast(pd.Series, self.data['Close'].diff().clip(lower=0).ewm(span=span).mean().mean())
        loss: pd.Series = cast(pd.Series, -self.data['Close'].diff().clip(upper=0).ewm(span=span).mean().mean())
        rs = (gain / loss).iloc[0]
        rsi = 100 - (100 / (1 + rs))
        return ML_GLOBAL_FLOAT_TYPE(rsi)

    def crossover_signal_count(self, sig_type: Literal['buy', 'sell', 'all']) -> float:
        lma_span = np.ceil(np.sqrt(len(self.data)))
        sma_span = np.ceil(len(self.data)**(1/3))

        lma = self.data['Close'].ewm(span=lma_span).mean()
        sma = self.data['Close'].ewm(span=sma_span).mean()
        if sig_type == 'buy':
            crossover = np.logical_and(
                lma > sma,
                lma.shift(1) <= sma.shift(1)
            ).sum().iloc[0]

        elif sig_type == 'sell':
            crossover = np.logical_and(
                lma < sma,
                lma.shift(1) >= sma.shift(1)
            ).sum().iloc[0]

        else:
            crossover = np.logical_or(
                    np.logical_and(lma > sma, lma.shift(1) <= sma.shift(1)),
                    np.logical_and(lma < sma, lma.shift(1) >= sma.shift(1))
                ).sum().iloc[0]

        return ML_GLOBAL_FLOAT_TYPE(crossover)

    def log_volatility(self) -> float:
        span = np.ceil(np.sqrt(len(self.data)))
        log_returns: pd.Series = cast(pd.Series, np.log(self.data['Close'] / self.data['Close'].shift(1))).dropna()
        volatility = cast(pd.Series, log_returns.ewm(span=span).std().mean()).iloc[0]
        return ML_GLOBAL_FLOAT_TYPE(volatility)

    def comparative_volatility(self) -> float:
        span = np.ceil(np.sqrt(len(self.data)))
        log_returns = cast(pd.Series, np.log(self.data['Close'] / self.data['Close'].shift(1))).dropna()
        volatility = log_returns.ewm(span=span).std()
        comparative_volatility = cast(pd.Series, volatility.pct_change().ewm(span=span/2).mean().mean()).iloc[0]  # Check recent percent change by EWMA
        return ML_GLOBAL_FLOAT_TYPE(comparative_volatility)

    def directional_persistence(self) -> float:
        span = int(np.ceil(np.sqrt(len(self.data))))
        directionality = cast(pd.Series, np.sign(self.data['Close'].diff())).rolling(window=span).sum()
        persistence = directionality.abs().mean() / span
        return ML_GLOBAL_FLOAT_TYPE(persistence)
    