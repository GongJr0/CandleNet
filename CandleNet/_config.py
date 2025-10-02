import yaml, pathlib
from dataclasses import dataclass
from typing import TypedDict, Union, Literal, Iterable
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
_conf = yaml.safe_load(_confPath.read_text())


class LagConfig(TypedDict):
    minLags: int
    maxLags: Union[int, Literal["auto"], None]
    selectionMethod: Literal["auto"]  # One method as of now
    sigLevel: float


class FeaturePool(TypedDict):
    autoregressive: bool
    sentiment: bool
    sectoral: bool
    technical: bool


@dataclass(frozen=True)
class Config:
    index: tuple[str, ...]
    lagSelection: LagConfig
    featurePool: FeaturePool


lag_defaults: LagConfig = {
    "minLags": 2, "maxLags": "auto", "selectionMethod": "auto", "sigLevel": 0.05
}


feature_defaults: FeaturePool = {
    "autoregressive": True, "sentiment": True, "sectoral": True, "technical": True
}

lag_dict: LagConfig = {
    **lag_defaults,
    **_conf.get("lagSelection", {}),
}
feature_dict: FeaturePool = {
    **feature_defaults,
    **_conf.get("featurePool", {}),
}

_index: tuple[str, ...] = valid_index(
    [k for k, v in _conf["index"].items() if v] or _SUPPORTED_INDEX
)
_lag = MappingProxyType(lag_dict)
_feature = MappingProxyType(feature_dict)


config: Config = Config(
    index=_index,
    lagSelection=_lag,  # type: ignore
    featurePool=_feature,  # type: ignore
)
