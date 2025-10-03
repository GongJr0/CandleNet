from pytest import fixture
from CandleNet.logger import LogType, OriginType, CallerType
from CandleNet.ticker.ticker import get_tickers_from_cache
from CandleNet.cache.ticker_cache import TickerCache
from CandleNet.ticker.ticker import Ticker
from CandleNet.synergy_matrix import Synergy
import numpy as np
import random
import string
import os



@fixture(scope="function")
def msg():
    charset = string.ascii_letters + string.digits + string.punctuation + " "
    length = random.randint(20, 140)
    return "".join(random.choices(charset, k=length))


@fixture
def origin():
    return random.choice([OriginType.SYSTEM, OriginType.USER])


@fixture
def log_type():
    return random.choice(
        [LogType.EVENT, LogType.WARNING, LogType.ERROR, LogType.STATUS]
    )

@fixture
def caller():
    return random.choice([CallerType.TICKER, CallerType.CACHE, CallerType.UTILS,
                          CallerType.AUTOREG, CallerType.SYNERGY, CallerType.SCALERS,
                          CallerType.SENTIMENT])

@fixture(scope="session")
def sp500_symbols():
    # If gspc_from_cache returns a dict[symbol -> Ticker], adapt:
    """
    Get the S&P 500 ticker symbols from the local ticker cache.
    
    If the cache returns a mapping of symbol -> Ticker, returns the mapping's keys as a list; otherwise returns the cached sequence as-is.
    
    Returns:
        symbols (list[str]): List of ticker symbol strings.
    """
    cached = get_tickers_from_cache()
    return list(cached) if isinstance(cached, dict) else cached  # robust


@fixture(scope="function")
def ticker(sp500_symbols):
    symbol = random.choice(sp500_symbols)
    with TickerCache() as c:
        resp = c[symbol]
        assert resp is not None, f"{symbol} not present in cache"
        return Ticker(symbol, _from_cache=resp)


@fixture
def s():
    return Synergy()


@fixture
def arr():
    mag = random.randint(1, 6)
    size = random.randint(50, 100)
    size = size**2  # len(arr) == len(ndarray) with same rng
    return np.random.randn(size, 1) * (10**mag)

@fixture
def ndarray():
    mag = random.randint(1, 6)
    size = random.randint(50, 100)
    return np.random.randn(size, size) * (10**mag)


# Optional seed for reproducibility
@fixture(scope="session", autouse=True)
def _seed_random():
    random.seed(0)


@fixture(scope="session", autouse=True)
def _seed_numpy():
    np.random.seed(0)


@fixture(scope="session", autouse=False)
def _gspc_to_txt():
    """
    Ensure a local ./data directory exists and write the current list of S&P 500 tickers to ./data/gspc.txt, one ticker per line.
    
    This overwrites any existing ./data/gspc.txt file.
    """
    os.makedirs("./data/", exist_ok=True)
    ticker_list = get_tickers_from_cache()

    with open("./data/gspc.txt", "w") as f:
        f.write("\n".join(ticker_list))
