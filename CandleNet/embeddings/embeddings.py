from CandleNet.utils import str_encode
from CandleNet.synergy_matrix import gspc_sector

from typing import Optional


class EmbedUtils:
    def __init__(self) -> None:
        self.data = gspc_sector()

        self._ticker_encoding: Optional[dict[str, int]] = None
        self._sector_encoding: Optional[dict[str, int]] = None

    @staticmethod
    def encoded(s: str):
        return str_encode(s)

    @property
    def ticker_encoding(self) -> dict[str, int]:
        if self._ticker_encoding is None:
            symbols = self.data["symbol"].tolist()
            self._ticker_encoding = {sym: str_encode(sym) for sym in symbols}
        return self._ticker_encoding

    @property
    def sector_encoding(self) -> dict[str, int]:
        if self._sector_encoding is None:
            sectors = self.data["sec"].unique().tolist()
            self._sector_encoding = {sec: str_encode(sec) for sec in sectors}
        return self._sector_encoding
