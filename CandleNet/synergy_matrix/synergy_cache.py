import numpy as np
import pandas as pd
from hashlib import sha256
import json
from CandleNet.cache import BaseCache
from CandleNet.cache.codec import Codec
from typing import Sequence, Optional


class SynergyCodec:
    @staticmethod
    def enc_matrix(matrix: pd.DataFrame) -> tuple[bytes, bytes]:
        names = np.asarray(matrix.columns.sort_values().astype(str).to_numpy(), dtype="U16")
        grid = np.asarray(matrix.values, dtype=np.float32, order='C')
        grid_enc = Codec.enc_numpy(grid)
        names_enc = Codec.enc_numpy(names)

        return grid_enc, names_enc

    @staticmethod
    def dec_matrix(grid: bytes, names: bytes) -> pd.DataFrame:
        grid_dec = Codec.dec_numpy(grid).astype(np.float32, copy=False)
        names_dec = Codec.dec_numpy(names).astype(str)

        return pd.DataFrame(grid_dec, columns=names_dec, index=names_dec)

    @staticmethod
    def enc_raw_names(names: Sequence[str]) -> bytes:
        names_list = sorted(list(map(str, names)))
        names_raw = json.dumps(names_list, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        return names_raw


class SynergyCache(BaseCache):
    def __init__(self):
        super().__init__(86400 * 14)  # 2 weeks
        self._table_name = 'GSPC_synergy_matrix'

    def insert(self, mat: pd.DataFrame, version: int = 1):
        self.check_con()

        enc_grid, enc_names = SynergyCodec.enc_matrix(mat)
        names_raw = SynergyCodec.enc_raw_names(mat.columns)

        matrix_id = sha256(names_raw).digest()
        matrix = enc_grid
        tickers = enc_names
        created_at = self.ts_now_iso()
        ttl_epoch = self.ts_now_epoch() + self.TTL

        query = f"""
                INSERT OR REPLACE INTO {self.TABLE_NAME} (matrix_id, version, matrix, tickers, created_at, ttl_epoch)
                VALUES (?, ?, ?, ?, ?, ?);"""
        self.con.execute(query, (matrix_id, version, matrix, tickers, created_at, ttl_epoch))


    def fetch(self, matrix_tickers: Sequence[str], version_spec: Optional[float] = None) -> pd.DataFrame | None:
        self.check_con()

        names_raw = SynergyCodec.enc_raw_names(matrix_tickers)
        matrix_id = sha256(names_raw).digest()

        cursor = self.con.cursor()
        if version_spec is not None:
            query = f"""SELECT matrix, tickers, ttl_epoch
                      FROM {self.TABLE_NAME} WHERE matrix_id = ? AND version = ?;"""

            cursor.execute(query, (matrix_id, version_spec))
        else:
            query = f"""SELECT matrix, tickers, ttl_epoch
                    FROM {self.TABLE_NAME}
                    WHERE matrix_id = ?
                    ORDER BY version DESC
                    LIMIT 1;"""
            cursor.execute(query, (matrix_id,))
        resp = cursor.fetchone()

        if resp is None:
            return None

        matrix_blob, tickers_blob, ttl_epoch = resp
        if ttl_epoch < self.ts_now_epoch():
            self.delete(matrix_id)
            return None

        mat = SynergyCodec.dec_matrix(matrix_blob, tickers_blob)
        return mat

    def delete(self, matrix_id: bytes) -> None:
        self.check_con()
        query = f"DELETE FROM {self.TABLE_NAME} WHERE matrix_id = ?;"
        self.con.execute(query, (matrix_id,))
        return

    def clear(self) -> None:
        self.check_con()
        query = f"DELETE FROM {self.TABLE_NAME};"
        self.con.execute(query)
        return

    @property
    def TABLE_SCHEMA(self) -> dict:
        return {
            'matrix_id': 'BLOB PRIMARY KEY',
            'version': 'INTEGER NOT NULL',
            'matrix': 'BLOB NOT NULL',
            'tickers': 'BLOB NOT NULL',
            'created_at': 'TEXT NOT NULL',
            'ttl_epoch': 'INTEGER NOT NULL'
        }

    @property
    def TABLE_NAME(self) -> str:
        return self._table_name
