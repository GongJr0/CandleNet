from . import BaseCache
from ..logger import CallerType, LogType, OriginType


class P2Cache(BaseCache):
    def __init__(self):
        super().__init__(0)  # No expiration for P2 cache

    def insert(
        self,
        data_name: str,
        p: float,
        N: int,
        q: list[float],
        n: list[int],
        eps: float = 1e-6,
    ):
        con = self.check_con()

        cols = "data_name, p, N, q0, q1, q2, q3, q4, n1, n2, n3, eps, updated_at"
        placeholders = ",".join(["?"] * len(cols.split(", ")))
        query = (
            f"INSERT OR REPLACE INTO {self.TABLE_NAME} ({cols}) VALUES ({placeholders})"
        )
        values = (data_name, round(p, 2), N, *q, *n, eps, self.ts_now_iso())

        cur = con.cursor()
        cur.execute(query, values)
        self._log(
            LogType.EVENT,
            OriginType.USER,
            CallerType.CACHE,
            f"Upserted P2 state for '{data_name}' at p={p}",
        )

    def fetch(self, data_name: str, p: float):
        con = self.check_con()

        cur = con.cursor()
        query = f"SELECT * FROM {self.TABLE_NAME} WHERE data_name = ? AND p = ?"
        cur.execute(query, (data_name, p))
        resp = cur.fetchone()

        if resp is None:
            self._log(
                LogType.EVENT,
                OriginType.USER,
                CallerType.CACHE,
                f"No P2 state found for '{data_name}' at p={p}",
            )
            return None

        N = resp[2]
        q = resp[3:8]
        n = [0, *resp[8:11], N - 1]
        eps = resp[11]

        return {"N": N, "q": list(q), "n": list(n), "eps": eps}

    def delete(self, data_name: str):
        con = self.check_con()

        cur = con.cursor()
        query = f"DELETE FROM {self.TABLE_NAME} WHERE data_name = ?"
        cur.execute(query, (data_name,))
        con.commit()
        self._log(
            LogType.EVENT,
            OriginType.SYSTEM,
            CallerType.CACHE,
            f"Deleted P2 state for '{data_name}'",
        )

    def clear(self):
        con = self.check_con()

        cur = con.cursor()
        query = f"DELETE FROM {self.TABLE_NAME}"
        cur.execute(query)
        con.commit()
        self._log(
            LogType.EVENT,
            OriginType.SYSTEM,
            CallerType.CACHE,
            "Cleared all P2 state entries",
        )

    @property
    def TABLE_NAME(self) -> str:
        return "p2"

    @property
    def TABLE_SCHEMA(self) -> dict[str, str]:
        return {
            # identity
            "data_name": "TEXT NOT NULL",
            "p": "REAL NOT NULL",  # quantile in (0,1) | as text to avoid float precision issues
            # state
            "N": "INTEGER NOT NULL",  # total samples seen
            "q0": "REAL NOT NULL",
            "q1": "REAL NOT NULL",
            "q2": "REAL NOT NULL",
            "q3": "REAL NOT NULL",
            "q4": "REAL NOT NULL",
            "n1": "INTEGER NOT NULL",
            "n2": "INTEGER NOT NULL",
            "n3": "INTEGER NOT NULL",
            # meta
            "eps": "REAL NOT NULL DEFAULT 1e-06",
            "updated_at": "TEXT NOT NULL",  # ISO-8601 UTC
        }

    @property
    def TABLE_CONSTRAINTS(self) -> list[str]:
        return ["PRIMARY KEY (data_name, p)"]
