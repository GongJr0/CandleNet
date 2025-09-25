from pytest import fixture
from CandleNet import LogType, OriginType
from CandleNet import gspc_from_cache
from CandleNet.cache.ticker_cache import TickerCache
from CandleNet.ticker.ticker import Ticker
import random
import string
import os
import json


@fixture(scope="function")
def msg():
    charset = string.ascii_letters + string.digits + string.punctuation + ' '
    length = random.randint(20, 140)
    return ''.join(random.choices(charset, k=length))


@fixture
def origin():
    return random.choice([OriginType.SYSTEM, OriginType.USER])


@fixture
def log_type():
    return random.choice([LogType.EVENT, LogType.WARNING, LogType.ERROR, LogType.STATUS])


@fixture(scope="session")
def sp500_symbols():
    # If gspc_from_cache returns a dict[symbol -> Ticker], adapt:
    cached = gspc_from_cache()
    return list(cached) if isinstance(cached, dict) else cached  # robust

@fixture(scope="function")
def ticker(sp500_symbols):
    symbol = random.choice(sp500_symbols)
    with TickerCache() as c:
        resp = c[symbol]
        assert resp is not None, f"{symbol} not present in cache"
        return Ticker(symbol, _from_cache=resp)


# Optional seed for reproducibility
@fixture(scope="session", autouse=False)
def _seed_random():
    random.seed(0)


@fixture(scope="session", autouse=False)
def _gspc_to_txt():
    os.makedirs("./data/", exist_ok=True)
    ticker_list = gspc_from_cache()

    with open("./data/gspc.txt", "w") as f:
        f.write("\n".join(ticker_list))

