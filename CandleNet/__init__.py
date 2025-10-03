from .logger.logger import Logger
from .logger.logger_types import LogType, OriginType
from .cache.base_cache import BaseCache
from .cache.ticker_cache import TickerCache, TickerCodec
from ._config import config, LagConfig, FeaturePool

from typing import cast


def lag_config() -> LagConfig:
    return cast(LagConfig, config.lagSelection)


def feature_pool() -> FeaturePool:
    return cast(FeaturePool, config.featurePool)
