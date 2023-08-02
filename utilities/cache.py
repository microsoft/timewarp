from collections import defaultdict, namedtuple
from dataclasses import dataclass
from functools import singledispatch
import copy
from typing import Any, Callable, Dict, Optional, Set, Tuple


@dataclass
class NullaryClosure:
    """
    Represents what is effectively a `partial` which does not accept further
    positional or keyword arguments beyond those provided upon construction.

    See `Cache` for how to use this to cache the results of calls to `func`.

    Attributes:
        func: a callable which will be fed the `args` and `keywords`.
        args: a `tuple` representing the positional arguments.
        keywords: a `dict` representing the keyword arguments.

    Example:
        >>> from utilities.cache import NullaryClosure

        >>> # Example function
        ... def f(x, y=1):
        ...     return x + y

        >>> # Construct a `NullaryClosure`.
        ... f_cacheable = NullaryClosure(f, (1,), dict(y=2))

        >>> f_cacheable()  # (✓) Works like a regular nullary function.
        3

        >>> # More convenient constructor for `NullaryClosure`.
            ff_cacheable = NullaryClosure.create(f, 1, y=2)

        >>> f_cacheable()  # (✓) Works like a regular nullary function.
        3

        >>> f_cacheable(1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        TypeError: __call__() takes 1 positional argument but 2 were given

        >>> # Note that the alternative constructor `cacheable` won't work
        ... # if `f` is also the name of a keyword argument.
        ... def g(x, f=1):
        ...    return x + f

        >>> NullaryClosure.create(f, 1, f=2)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        TypeError: NullaryClosure.create() got multiple values for argument 'f'
    """

    func: Callable
    args: Tuple
    keywords: Dict

    def __call__(self):
        return self.func(*self.args, **self.keywords)

    @classmethod
    def create(cls, f, *args, **kwargs):
        """
        Return an instance of `NullaryClosure` wrapping `f`, `args` and `kwargs`.

        Note that if `kwargs` also contains the key `f`, calling this function
        will result in a `TypeError` complaining about multiple values for argument `f`.
        In such a case, one instead has to call `NullaryClosure` directly.
        """
        return cls(f, args, kwargs)


CacheInfo = namedtuple("CacheInfo", ["hits", "misses"])


# NOTE: Copy-paste from `functools.@lru_cache` implementation.
# https://github.com/python/cpython/blob/ccbc31ecf3a08ef626be9bbb099f0ce801142fc8/Lib/functools.py#L440-L485
class _HashedSeq(list):
    """This class guarantees that hash() will be called no more than once
    per element.  This is important because the lru_cache() will hash
    the key multiple times on a cache miss.

    """

    __slots__ = "hashvalue"

    def __init__(self, tup, hash=hash):
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue


# NOTE: Copy-paste from `functools.@lru_cache` implementation.
# https://github.com/python/cpython/blob/ccbc31ecf3a08ef626be9bbb099f0ce801142fc8/Lib/functools.py#L440-L485
# with minor alterations: we now include `f` in the key and removed unnused code relating to "fasttypes".
def _make_key(
    f,
    args,
    kwds,
    typed,
    kwd_mark=(object(),),
    tuple=tuple,
    type=type,
):
    """Make a cache key from optionally function, typed positional and keyword arguments

    The key is constructed in a way that is flat as possible rather than
    as a nested structure that would take more memory.
    """
    # All of code below relies on kwds preserving the order input by the user.
    # Formerly, we sorted() the kwds before looping.  The new way is *much*
    # faster; however, it means that f(x=1, y=2) will now be treated as a
    # distinct call from f(y=2, x=1) which will be cached separately.
    key = (f, *args)
    if kwds:
        key += kwd_mark
        for item in kwds.items():
            key += item
    if typed:
        key += tuple(type(v) for v in args)
        if kwds:
            key += tuple(type(v) for v in kwds.values())
    return _HashedSeq(key)


@singledispatch
def get_function_identifier(f: Any):
    """
    Return the identifier for `f`. The identifier depends on the type of `f`.
    """
    return f


@get_function_identifier.register
def _(f: NullaryClosure):
    return f.func


def make_key(
    f: NullaryClosure,
    argument_transforms: Dict[int, Callable] = dict(),
    keyword_transforms: Dict[str, Callable] = dict(),
):
    args = list(copy.copy(f.args))
    for (i, transform) in argument_transforms.items():
        if i < len(args):
            args[i] = transform(args[i])

    kwargs = copy.copy(f.keywords)
    for (k, transform) in keyword_transforms.items():
        if k in kwargs:
            kwargs[k] = transform(kwargs[k])

    return _make_key(f.func, tuple(args), kwargs, False)


class Cache(object):
    """
    A simple implementation of a cache requiring manual specification
    of what is and what is not considered `cacheable`.

    This is meant to provide an alternative to `@lru_cache` where
    - we have direct control of the lifetime of the cache, and
    - transformations can be applied to arguments and keyword arguments before hashing.

    Attributes:
        cacheable: A `Set` representing the keys considered cacheable.
        argument_transforms: A "function" -> "argument index" -> "callable" lookup,
            where "callable" is used to transform the value provided for "keyword argument".
            This is useful way of altering how certain keyword arguments are compared.
        keyword_transforms: A "function" -> "keyword argument" -> "callable" lookup,
            where "callable" is used to transform the value provided for "keyword argument".
            This is useful way of altering how certain keyword arguments are compared.

    Example:
        >>> from utilities.cache import Cache, cacheable

        >>> # An example method we'd like to cache.
        ... def f(x, y=1):
        ...     return x + y

        >>> # Make a cache and specify that `f` is cacheable.
        ... cache = Cache(cacheable={f,})

        >>> # Wrap it in a `NullaryClosure` so it can be called like `f()`.
        ... f_partial = NullaryClosure.create(f, 1, y=2)

        >>> cache.load_or_produce(f_partial)  # result is cached
        3

        >>> cache.load_or_produce(f_partial)  # cached result is returned
        3

        >>> # And we can inspect the `CacheInfo` for `f` to verify the behavior.
        ... cache.cache_info(f)
        CacheInfo(hits=1, misses=1)

        >>> # Under the hood we use `hash` for `f`, the arguments, and the keyword arguments,
        ... # which means that one needs to be mindful of how `hash` works.
        ... # For example, a `list` is not hashable, so we if we instead have:
        ... def g(x, y=1):
        ...     return x[0] + y

        >>> g_partial = NullaryClosure.create(g, [1], y=2)

        >>> # Then things start breaking.
        ... cache = Cache(cacheable={g,})

        >>> cache.load_or_produce(g_partial)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        TypeError: ...

        >>> # It's of course a contrived example, but in this particular case we
        ... # can fix this by caching based on `x[0]` instead of `x`.
        ... # Note that this is ok here because we have intricate knowledge of how `g`
        ... # is implemented. In general, this should be used with care.
        ... cache = Cache(cacheable={g,}, argument_transforms={g: {0: lambda x: x[0],},})

        >>> cache.load_or_produce(g_partial)
        3

        >>> cache.load_or_produce(g_partial)
        3

        >>> cache.cache_info(g)
        CacheInfo(hits=1, misses=1)
    """

    def __init__(
        self,
        cacheable: Optional[Set[Callable]] = None,
        argument_transforms: Optional[Dict[Any, Dict[int, Callable]]] = None,
        keyword_transforms: Optional[Dict[Any, Dict[str, Callable]]] = None,
    ):
        self.cacheable = cacheable if cacheable is not None else set()
        self.argument_transforms = (
            argument_transforms if argument_transforms is not None else dict()
        )
        self.keyword_transforms = keyword_transforms if keyword_transforms is not None else dict()
        self._lookup: Dict = dict()

        self._hits: Dict = defaultdict(lambda: 0)
        self._misses: Dict = defaultdict(lambda: 0)

    def empty_like(self):
        "Return new and empty instance of `Cache` similar to this one."
        return Cache(
            cacheable=copy.copy(self.cacheable),
            argument_transforms=copy.copy(self.argument_transforms),
            keyword_transforms=copy.copy(self.keyword_transforms),
        )

    def should_cache(self, f: Callable):
        "Return `True` if `f` is considered cacheable. Return `False` otherwise."
        return get_function_identifier(f) in self.cacheable

    def load_or_produce(self, f: NullaryClosure):
        """
        Return result of `f()`, regardless of whether `f` is considered cacheable.

        If `f` is present in the cache, the cached value will be returned.
        Otherwise, `f()` is computed. The result is stored in the cache
        if `f` is considered cacheable.
        """
        # We'll need the identifier for `f` a couple of times, so let's just get it once.
        func_identifier = get_function_identifier(f)
        # Construct the key for `f`, possibly using transformations of the keyword arguments.
        key = make_key(
            f,
            self.argument_transforms.get(func_identifier, dict()),
            self.keyword_transforms.get(func_identifier, dict()),
        )

        # If we have a hit, return early.
        if key in self._lookup:
            self._hits[func_identifier] += 1
            return self._lookup[key]

        # Otherwise, compute.
        result = f()

        # Only consider it a cache-miss if we actually mean to cache this.
        if self.should_cache(func_identifier):
            self._misses[func_identifier] += 1
            self._lookup[key] = result

        return result

    def cache_info(self, f):
        "Return some cache statistics."
        return CacheInfo(self._hits[f], self._misses[f])
