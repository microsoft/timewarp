from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple, Union

import pandas
from pandas import DataFrame, Series

from .base import DataTable

from utilities.common import StrPath


def get_nested_single(data: Union[DataFrame, Series], column: str, sep: str = ".") -> Series:
    for key in column.split(sep):
        if isinstance(data, DataFrame):
            data = data[key]
        else:
            data = data.apply(lambda v: v[key])
    return data.rename(column)


def get_nested(
    data: Union[DataFrame, Series], column: Union[str, List[str]], sep: str = "."
) -> Union[DataFrame, Series]:
    """Returns a series or dataframe holding the values specified by a nested column key
    (e.g., field.subfield.subsubfield) from a dataframe containing dictionaries.

    Args:
        data: source dataframe or series.
        column: (a list of) nested column key joined by separators.
        sep: the separator (default: '.')

    Returns:
        a dataframe if column is a list, otherwise a series.
    """
    if isinstance(column, str):
        return get_nested_single(data, column, sep)
    concat = pandas.concat((get_nested_single(data, ind, sep) for ind in column), axis=1)
    return concat


@dataclass
class PandasDataTable(DataTable):
    output_path: StrPath
    _df: DataFrame = field(default_factory=DataFrame)

    def insert(self, entities: Union[Iterable[Dict], DataFrame]):
        self._df = pandas.concat((self._df, DataFrame(entities)))

    def __len__(self):
        return len(self._df)

    def __getitem__(self, column: Union[str, List[str]]) -> Iterator[Any]:
        return self.filter({}, select=column)

    def __contains__(self, column: str) -> bool:
        return column in self._df

    def filter(
        self, filters: Dict[str, Any], select: Union[None, str, List[str]] = None
    ) -> Iterator[Any]:
        filtered = (
            self._df
            if len(filters) == 0
            else self._df[
                reduce(
                    (lambda a, b: a & b),
                    [get_nested(self._df, key) == val for key, val in filters.items()],
                )
            ]
        )
        if select is not None:
            filtered = get_nested(filtered, select)
        if isinstance(select, str):
            yield from filtered
        else:
            yield from filtered.to_dict("records")

    def update_or_insert(
        self, entries: Union[Iterable[Dict], DataFrame], on: str, choose: Tuple[Callable, str]
    ):
        if not isinstance(entries, DataFrame):
            entries = DataFrame(entries)
        op, column = choose
        assert callable(op)
        concat = pandas.concat([self._df, entries])
        idx = concat.groupby([on])[column].transform(op) == concat[column]
        self._df = concat[idx]

    def commit(self):
        self._df.to_pickle(self.output_path)

    def to_dataframe(self) -> DataFrame:
        return self._df
