from typing import TypedDict
from typing import Optional
import pandas as pd


class TickerData(TypedDict):
    ticker: str
    price: pd.Series
    volume: pd.Series
    hilo: pd.DataFrame
    sentiment: Optional[float]
