from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple, Union

from pandas import DataFrame


class DataTable(ABC):
    @abstractmethod
    def insert(self, entities: Iterable[Dict]):
        """Adds entries to the table."""

    @abstractmethod
    def __len__(self):
        """Returns the number of rows."""

    @abstractmethod
    def __getitem__(self, column: Union[str, List[str]]) -> Iterator[Any]:
        """Returns an iterator over the values of a column."""

    @abstractmethod
    def __contains__(self, column: str) -> bool:
        """Returns if a column exists in the table."""

    @abstractmethod
    def filter(self, filters: Dict[str, Any], select: Union[None, str, List[str]]) -> Iterator[Any]:
        """Returns a view that consists of rows chosen according to kwargs.

        Args:
            filters: dictionary mapping from column name to value.
                (e.g., {"color": "blue"} will return rows with value "blue" in the "color" column.)
            select: when select is specified, only selected column(s) are returned.
                (projection in Mongo DB.)

        Returns:
            an iterator over the column values or dictionaries containing the selected
            column names and values.
        """

    @abstractmethod
    def update_or_insert(self, entries: Iterable[Dict], on: str, choose: Tuple[Callable, str]):
        """Update the table by grouping first on a key and choosing one from each group according to min/max of some column.

        Typically we would group with a unique identifier and choose the record with the latest
        timestamp. In this case, we would use `on="unique_id"` and `choose=(max, "timestamp")`.

        Args:
            entries: iterable of dictionaries (e.g., can be obtained by df.to_dict("records")).
            on: column name to group by.
            choose: a (callable, column name) tuple. The callable is applied to each value in the column
                specified by the column name and decides which entry is chosen from each group.
        """

    @abstractmethod
    def commit(self):
        """Updates the underlying storage with the local changes. May be a no-op if update is instantaneous."""

    @abstractmethod
    def to_dataframe(self) -> DataFrame:
        """Convert to pandas DataFrame."""
