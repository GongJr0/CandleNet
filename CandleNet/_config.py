import yaml
import pathlib
from dataclasses import dataclass
from typing import TypedDict, Union, Literal, Iterable, Mapping, Optional, Any
from types import MappingProxyType

_SUPPORTED_INDEX = [
    "GSPC",
    # "DJI",
    # "IXIC",
    # NDX
]


def valid_index(x: Iterable[str]) -> tuple[str, ...]:
    return tuple(idx for idx in x if idx in _SUPPORTED_INDEX)


_confPath = pathlib.Path("./featureConfig.yaml")
try:
    _text = _confPath.read_text()
    _conf = yaml.safe_load(_text) or {}
except (FileNotFoundError, yaml.YAMLError):
    _conf = {}


class LagConfig(TypedDict):
    minLagsSelected: int
    maxLagsSelected: Union[int, Literal["auto"], None]
    maxLag: int
    selectionMethod: Literal["fdrAdjusted", "rawPval"]
    sigLevel: float

    requireStability: bool
    stabilityFreq: float
    stabilityConfidence: float
    stabilityCheckEvery: int

    bootstrapSamples: int
    minBootstrapSamples: int
    earlyStop: bool
    blockLen: Union[int, Literal["auto"]]

    rankTopN: int

    hacBandwidth: Union[int, Literal["auto"]]

    randomSeed: Optional[int]


class FeaturePool(TypedDict):
    autoregressive: bool
    sentiment: bool
    sectoral: bool
    technical: bool


@dataclass(frozen=True)
class Config:
    index: tuple[str, ...]
    lagSelection: Mapping[str, Any]
    featurePool: Mapping[str, Any]


lag_defaults: LagConfig = {
    "minLagsSelected": 2,
    "maxLagsSelected": "auto",
    "maxLag": 20,
    "selectionMethod": "fdrAdjusted",
    "sigLevel": 0.05,
    "requireStability": True,
    "stabilityFreq": 0.25,
    "stabilityConfidence": 0.95,
    "stabilityCheckEvery": 25,
    "bootstrapSamples": 500,
    "minBootstrapSamples": 100,
    "earlyStop": True,
    "blockLen": "auto",
    "rankTopN": 5,
    "hacBandwidth": "auto",
    "randomSeed": None,
}


feature_defaults: FeaturePool = {
    "autoregressive": True,
    "sentiment": True,
    "sectoral": True,
    "technical": True,
}

lag_conf: dict = _conf.get("lagSelection", {})
lag_dict: dict = {
    **lag_defaults,
    **lag_conf,
}
feature_dict: dict = {
    **feature_defaults,
    **_conf.get("featurePool", {}),
}

_index: tuple[str, ...] = valid_index(
    [k for k, v in _conf.get("index", {}).items() if v] or _SUPPORTED_INDEX
)
_lag = MappingProxyType(lag_dict)
_feature = MappingProxyType(feature_dict)


config: Config = Config(
    index=_index,
    lagSelection=_lag,
    featurePool=_feature,
)
