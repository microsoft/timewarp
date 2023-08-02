import os
from functools import wraps
from typing import Callable, Tuple, Type, TypeVar, Union

import pytest

ReturnType = TypeVar("ReturnType")


def xfail_in_merge_queue(
    raises: Union[Type[BaseException], Tuple[Type[BaseException], ...]], reason: str
) -> Callable:
    """Decorates a pytest test function so that it xfails in GitHub merge queue for known exception types.

    Args:
        raises: an exception type or a tuple of exception types to xfail on.
        reason: reason for xfail.
    """

    def wrapper(f: Callable[..., ReturnType]) -> Callable[..., ReturnType]:
        @wraps(f)
        def wrapped(*args, **kwargs):
            source_branch = os.environ.get("BUILD_SOURCEBRANCH", "")
            try:
                return f(*args, **kwargs)
            except raises:
                if source_branch.startswith("refs/heads/gh-readonly-queue"):
                    pytest.xfail(reason)
                else:
                    raise

        return wrapped

    return wrapper
