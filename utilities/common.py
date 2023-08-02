from contextlib import contextmanager
import importlib
import os
import sys
import glob
from typing import Any, Dict, Iterable, Iterator, List, Sequence, TypeVar, Union

T = TypeVar("T")

StrPath = Union[str, os.PathLike]


def glob_only(path: StrPath, recursive: bool = True) -> str:
    """
    Expand `path` using `glob`, erroring if none or more than one expansions are found.

    Note that this is a no-op if `path` is a path without glob-patterns.
    """
    return unique_item(glob.glob(str(path), recursive=recursive))


def find_files(dirpath: StrPath, name: str) -> Iterator[str]:
    """
    Find files with a given name in subdirectories.

    Args:
        dirpath: path to root.
        name: name of the file to look for.
    """
    for dirname, _, files in os.walk(dirpath):  # type: ignore
        if name in files:
            yield os.path.join(dirname, name)


def unique_item(iterable: Iterable[T]) -> T:
    """returns the content of a sequence containing a single item."""
    lst = list(iterable)
    assert len(lst) == 1, f"Tried to call unique_item, but {lst} contains {len(lst)} items."
    return lst[0]


def approximately_equal_partition(lst: Sequence[T], n: int) -> List[Sequence[T]]:
    """
    Returns an order-preserving approximately equal sized partition of a sequence.

    Args:
        lst: a sequence of objects to be partitioned.
        n: the size of the partition.

    Returns:
        length n list of parts.
    """
    delta = len(lst) / n
    return [lst[round(delta * i) : round(delta * (i + 1))] for i in range(n)]


class Returns(object):
    """
    Represents a callable which always returns the same `value`.

    Usage:
        >>> f = Returns(0)
        >>> f()
        0
        >>> f(1)
        0
        >>> f(1, 2, 3, x=3)
        0
    """

    def __init__(self, value):
        self.value = value

    def __call__(self, *args, **kwargs):
        return self.value


def get_nested_item(d: Dict[str, Any], index: str, sep: str = ".") -> Any:
    """Returns a value from a nested dictionary using keys like field.subfiled.subsubfield.

    Args:
        d: a nested dictionary.
        index: nested keys joined by separators.
        sep: the separator (default: '.').

    Returns:
        the value stored at the nested index.
    """
    for key in index.split(sep):
        d = d[key]
    return d


def extend_list_if(cond: bool, lst1: List[T], lst2: List[T]) -> List[T]:
    """Returns the concatenation of two lists if the condition is true, otherwise returns
    only the first list.

    Args:
        cond: the condition.
        lst1: the first list.
        lst2: the second list.

    Returns:
        a possibly concatenated list.
    """
    return lst1 + (lst2 if cond else [])


@contextmanager
def change_working_directory(path: StrPath):
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def maypluralize(num: int, item: str) -> str:
    return item + "s" if num > 1 else item


def import_path(path, module_name=None):
    if module_name is None:
        module_name = os.path.basename(path).replace("-", "_")
        module_name = module_name.partition(".py")[0]
    spec = importlib.util.spec_from_loader(
        module_name, importlib.machinery.SourceFileLoader(module_name, path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module
