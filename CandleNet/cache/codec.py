import io
import pickle
import pandas as pd
import numpy as np
import sys


class Codec:
    @staticmethod
    def enc_arrow(df: pd.DataFrame) -> bytes:
        try:
            import pyarrow as pa
            import pyarrow.ipc as ipc
        except ImportError:
            install = input(
                "pyarrow is required for caching. Do you want to install it now? (y/n): "
            )
            if install.lower() == "y" or install.lower() == "yes":
                import subprocess

                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "pyarrow"]
                )
                import pyarrow as pa
                import pyarrow.ipc as ipc
            else:
                raise ImportError("Missing dependency: pyarrow")

        table = pa.Table.from_pandas(df)
        sink = io.BytesIO()
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)

        return sink.getvalue()

    @staticmethod
    def dec_arrow(blob: bytes) -> pd.DataFrame:
        try:
            import pyarrow as pa
            import pyarrow.ipc as ipc
        except ImportError:
            install = input(
                "pyarrow is required for caching. Do you want to install it now? (y/n): "
            )
            if install.lower() in {"y", "yes"}:
                import subprocess

                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "pyarrow"]
                )
                import pyarrow as pa
                import pyarrow.ipc as ipc
            else:
                raise ImportError("Missing dependency: pyarrow")

        source = io.BytesIO(blob)
        with ipc.open_stream(source) as reader:
            table = reader.read_all()  # pa.Table
            return table.to_pandas()  # pd.DataFrame

    @staticmethod
    def enc_pickle(obj) -> bytes:
        return pickle.dumps(obj)

    @staticmethod
    def dec_pickle(blob: bytes):
        return pickle.loads(blob)

    @staticmethod
    def enc_numpy(arr: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        np.save(buffer, arr, allow_pickle=False)
        return buffer.getvalue()

    @staticmethod
    def dec_numpy(blob: bytes) -> np.ndarray:
        buffer = io.BytesIO(blob)
        buffer.seek(0)
        return np.load(buffer, allow_pickle=False)
