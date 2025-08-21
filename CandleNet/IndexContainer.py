import pandas as pd
from functools import wraps
from dataclasses import dataclass

from .DataAccess.utils import INDEX_TYPE, VALID_INDEX, PERIOD_TYPE, DATE_TYPE, INTERVAL_TYPE
from .DataAccess.utils import returns_dataframe, cast_global_ml
from .DataAccess.TickerAccessor import TickerAccessor
from .TickerContainer import TickerContainer


def match_combined_params(_COMBINE_INDEX):
    @wraps(_COMBINE_INDEX)
    def wrapper(*args):
        assert len(set(arg.period for arg in args)) == 1, "Period mismatch in combined indices."
        assert len(set(arg.interval for arg in args)) == 1, "Interval mismatch in combined indices."
        assert len(set(arg.start for arg in args)) == 1, "Start date mismatch in combined indices."
        assert len(set(arg.end for arg in args)) == 1, "End date mismatch in combined indices."
        return _COMBINE_INDEX(*args)
    return wrapper


@dataclass
class IndexMetaData:
    name: str
    n_components: int
    period: PERIOD_TYPE
    interval: INTERVAL_TYPE
    start: DATE_TYPE = None
    end: DATE_TYPE = None


class IndexContainer:
    def __init__(self,
               index: INDEX_TYPE,
               interval: INTERVAL_TYPE,
               period: PERIOD_TYPE = None,
               start: DATE_TYPE = None,
               end: DATE_TYPE = None
               ) -> None:

        """Collection of ticker that define an index (e.g. S&P 500, NASDAQ)
        Tickers are separated into independent components, each wrapped to provide easy-access to conventional technical analysis methods.
        Joining tickers into a single frame intentionally omits ticker names to avoid providing categorical identifiers (creates bias) to ML models during training."""

        assert index in VALID_INDEX, f"Invalid index: {index}. Valid indices are: {VALID_INDEX}"

        self._data: pd.DataFrame = TickerAccessor.get_index(index, period=period, interval=interval, start=start, end=end)
        self.components = self.decompose()

        # Metadata
        self._metadata = IndexMetaData(
            name=index,
            n_components=len(self._data.columns),
            period=period,
            interval=interval,
            start=start,
            end=end
        )

    def decompose(self) -> dict[str, TickerContainer]:

        def get_data(name) -> pd.DataFrame:
            """
            Decomposes the index data into its components.
            """
            ticker_df: pd.DataFrame = self._data[name]
            return ticker_df

        names = list(map(
            lambda x: x[0],
            self._data.columns
        ))
        return {
            name: TickerContainer(name, get_data(name)) for name in names
        }

    @returns_dataframe
    def drop_names(self, df: pd.DataFrame) -> pd.DataFrame:
        no_name: pd.DataFrame = pd.concat(
            [val.data for val in self.components.values()],

        )
        return no_name

    @property
    def nameless_data(self) -> pd.DataFrame:
        no_name: pd.DataFrame = self.drop_names(self._data)
        return no_name

    @property
    @returns_dataframe
    def pass_data(self) -> pd.DataFrame:
        return self._data

    @property
    def get_meta(self) -> IndexMetaData:
        return self._metadata


class _CombinedIndex(IndexContainer):
    """
    A container for multiple indices, allowing for combined operations.
    USE COMBINE_INDEX() TO CREATE AN INSTANCE.
    """

    def __init__(self, *args: IndexContainer) -> None:
        assert all(isinstance(arg, IndexContainer) for arg in args), "All arguments must be IndexContainer instances."

        """Combines multiple IndexContainer instances into a single DataFrame."""

        self._data = pd.concat(
            [arg.pass_data for arg in args],
            join='outer',
            axis=1
        )
        self.components = self.decompose()

        combined_n_components = sum(arg.get_meta.n_components for arg in args)
        combined_names = f"CombinedIndex<{', '.join(arg.get_meta.name for arg in args)}>"

        self._metadata = IndexMetaData(
            name=combined_names,
            n_components=combined_n_components,
            period=args[0].get_meta.period,
            interval=args[0].get_meta.interval,
            start=args[0].get_meta.start,
            end=args[0].get_meta.end
        )


@match_combined_params
def combine_index(*args: IndexContainer) -> _CombinedIndex:
    """
    Combines multiple IndexContainer instances into a single CombinedIndex.
    """
    return _CombinedIndex(*args)
