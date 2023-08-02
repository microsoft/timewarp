from functools import partial
import time
from weakref import WeakKeyDictionary

from cachetools import TTLCache
from cachetools.func import _UnboundTTLCache  # type: ignore


class SelfTTLCaches:
    """Use this class as the cache argument to `cachetools.cachedmethod` decorator
    so that the cache is per-instance, but reference is weak so that it doesn't
    prevent garbage collection.
    """

    def __init__(self, maxsize=128, ttl=10, timer=time.monotonic):
        if maxsize is None:
            self.cache_cls = partial(_UnboundTTLCache, ttl, timer)
        else:
            self.cache_cls = partial(TTLCache, 128, ttl, timer)
        self.selfs = WeakKeyDictionary()

    def __call__(self, obj):
        cache = self.selfs.get(obj)
        if cache is not None:
            return cache
        return self.selfs.setdefault(obj, self.cache_cls())
