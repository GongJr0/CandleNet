from .logger.logger import Logger
from .logger.logger_types import LogType, OriginType
from .cache.base_cache import BaseCache
from .cache.ticker_cache import TickerCache, TickerCodec
from ._config import config, LagConfig, FeaturePool

from typing import cast


def lag_config() -> LagConfig:
    """
    Access the application's lag selection configuration.
    
    Returns:
        LagConfig: The lag selection configuration from the global config.
    """
    return cast(LagConfig, config.lagSelection)


def feature_pool() -> FeaturePool:
    """
    Get the current feature pool configuration.
    
    Returns:
        FeaturePool: The configured feature pool.
    """
    return cast(FeaturePool, config.featurePool)
