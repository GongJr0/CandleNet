from io import StringIO
import requests
from typing import Optional
from .ticker_data import TickerData
from ..cache.ticker_cache import TickerCache
from ..sentiment import get_sentiment_obj
from ..utils import get_lib
import yfinance as yf  # type: ignore[import-untyped]
import pandas as pd
import numpy as np


class Ticker:
    def __init__(self,
                 symbol: str,
                 _from_bulk_download: bool = False,
                raw_data: Optional[pd.DataFrame] = None,
                 _from_cache: Optional[TickerData] = None):
        self.symbol = symbol
        self._ticker = None
        self.s_polarity = get_sentiment_obj()

        if _from_cache:
            self.data = _from_cache
            self.raw_data = None
            self.news: dict | None = None
            return


        if not _from_bulk_download:
            with TickerCache() as c:
                if (resp := c[symbol]) is not None:
                    self.raw_data = None
                    self.news = None
                    price = resp['price']
                    volume = resp['volume']
                    hilo = resp['hilo']
                    sentiment = resp['sentiment']

                else:
                    self.raw_data = yf.download(symbol, period='1y', interval='1d', auto_adjust=True)
                    self.news = self.ticker.news
                    price = pd.Series(self.raw_data['Close'].values.flatten(), index=self.raw_data.index)
                    volume = pd.Series(self.raw_data['Volume'].values.flatten(), index=self.raw_data.index)
                    hilo = self.raw_data[['High', 'Low']]
                    sentiment = self.polarity()

                    c[symbol] = {
                        'ticker': symbol,
                        'price': price,
                        'volume': volume,
                        'hilo': hilo,
                        'sentiment': sentiment
                    }
        else:
            assert raw_data is not None, "raw_data must be provided when _from_bulk_download is True."
            self.raw_data = raw_data
            self.news = self.ticker.news

            cls: np.ndarray = raw_data['Close'].to_numpy()
            vol: np.ndarray = raw_data['Volume'].to_numpy()

            price = pd.Series(cls.flatten(), index=raw_data.index)
            volume = pd.Series(vol.flatten(), index=raw_data.index)
            hilo = raw_data[['High', 'Low']]
            sentiment = self.polarity()

            with TickerCache() as c:
                c[symbol] = {
                    'ticker': symbol,
                    'price': price,
                    'volume': volume,
                    'hilo': hilo,
                    'sentiment': sentiment
                }


        self.data = TickerData(
            ticker=symbol,
            price=price,
            volume=volume,
            hilo=hilo,
            sentiment=sentiment,
        )



    def process_news(self) -> tuple[list[str], list[str], list[str]]:
        assert self.news is not None
        titles, summaries, dates = [], [], []
        for a in self.news:
            # adapt keys if needed
            titles.append(a.get("title", "") or a.get("content", {}).get("title", ""))
            summaries.append(a.get("summary", "") or a.get("content", {}).get("summary", ""))
            dates.append(
                a.get("providerPublishTime")
                or a.get("date")
                or a.get("content", {}).get("pubDate", "")
            )
        return titles, summaries, dates

    def get_sentiment(self, news_dump: tuple[list[str], list[str], list[str]]) -> float:
        titles, summaries, dates = news_dump

        title_s = self.s_polarity.pipe_sentiment(titles)
        summary_s = self.s_polarity.pipe_sentiment(summaries)
        s = (2 / 3) * title_s + (1 / 3) * summary_s

        d_ser: pd.Series = pd.to_datetime(pd.Series(dates), errors="coerce", utc=True)

        nans: pd.Series = d_ser.isna()
        na_proportion: float = float(nans.mean())

        if na_proportion > 0.5:
            d_ser = pd.to_datetime(pd.Series(dates), unit="s", errors="coerce", utc=True)

        # finalize as DatetimeIndex
        d: pd.DatetimeIndex = pd.DatetimeIndex(d_ser)

        m = ~d.isna()
        d = d[m]
        s = s[m]

        ser = pd.Series(s, index=d).sort_index()

        # EWMA with 7-day half-life → per pandas: halflife in days if index is datetime
        ewma = ser.ewm(halflife=pd.Timedelta(7, 'days'), times=ser.index).mean()
        return float(ewma.iloc[-1])

    def polarity(self) -> float | None:
        if not self.news:
            return None
        news_dump = self.process_news()
        return self.get_sentiment(news_dump)

    @property
    def price(self) -> pd.Series:
        return self.data['price']

    @property
    def volume(self) -> pd.Series:
        return self.data['volume']

    @property
    def hilo(self) -> pd.DataFrame:
        return self.data['hilo']

    @property
    def sentiment(self) -> Optional[float]:
        return self.data['sentiment']

    @property
    def ticker(self) -> yf.Ticker:
        if self._ticker is None:
            self._ticker = yf.Ticker(self.symbol)
        return self._ticker




def _get_gspc() -> list:
    try:
        import lxml
    except ImportError:
        yn = input("lxml is required to fetch S&P 500 list. Install it now? (y/n): ")
        if yn.lower() in ['y', 'yes']:
            get_lib("lxml")
        else:
            raise ImportError("lxml is required to fetch S&P 500 list. Please install it and try again.")

    try:
        import html5lib
    except ImportError:
        yn = input("html5lib is required to fetch S&P 500 list. Install it now? (y/n): ")
        if yn.lower() in ['y', 'yes']:
            get_lib("html5lib")
        else:
            raise ImportError("html5lib is required to fetch S&P 500 list. Please install it and try again.")

    table_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(table_url, headers=headers)
    if resp.status_code != requests.codes.ok:
        raise ConnectionError(f"Failed to fetch S&P 500 list: {resp.status_code}")
    tables = pd.read_html(StringIO(resp.text))
    symbols = tables[0]['Symbol'].tolist()

    symbols = [s.replace(".", "-") for s in symbols]
    return symbols


def gspc_tickers() -> dict[str, Ticker]:
    symbols = _get_gspc()
    data = yf.download(symbols, period="1y", interval="1d", auto_adjust=True)
    data.columns = data.columns.swaplevel(0, 1)

    tickers = {}
    for symbol in symbols:
        if symbol in data.columns.get_level_values(0):
            df = data[symbol].dropna(how='all')
            tickers[symbol] = Ticker(symbol, _from_bulk_download=True, raw_data=df)

    return tickers


def gspc_from_cache() -> dict[str, Ticker]:
    """Only returns tickers of SP500 that are cached and not expired.
    This function exists avoid API rate limits of yfinance which are approx. 1500-2000 requests per hour.
    By definition, this requires two queries per ticker:
        1. Function Level: Check if the ticker is in cache.
        2. In Ticker.__init__: Hit the cache for raw data to construct the Ticker object.
    """
    symbols = _get_gspc()
    tickers = {}
    with TickerCache() as c:
        for symbol in symbols:
            if (resp := c[symbol]) is not None:
                tickers[symbol] = Ticker(symbol, _from_cache=resp)

    return tickers
