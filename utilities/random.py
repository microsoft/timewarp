from datetime import datetime
import random
import string
from typing import Any, Callable, Dict, Iterator, List, Type, Union
import warnings

from typing_extensions import get_origin

_registry: Dict[Type, Callable] = {}


def register(target_class):
    def inner(f):
        global _registry
        if target_class in _registry:
            warnings.warn(
                f"Overriding the existing random generator ({_registry[target_class]})"
                f" for {target_class} with {f}..."
            )
        _registry[target_class] = f
        return f

    return inner


def generate(cls, *args, **kwargs):
    """Generates a random instance of a given class.

    Args:
        cls: object class to be generated. Use register() decorator
            to register a generator.
        *args, **kwargs: additional arguments for the generator.
    """
    if cls in _registry:
        generator = _registry[cls]
        return generator(*args, **kwargs)

    # trivial types
    if cls is None or cls is type(None):  # noqa
        return None
    if cls is Any:
        typ = random.choice([int, float, str])
        return generate(typ)

    # generics
    origin = get_origin(cls)

    if origin is list:
        length = generate(int, 1, 10)
        typ = cls.__args__[0]
        return [generate(typ) for _ in range(length)]
    if origin is tuple:
        return tuple(generate(typ) for typ in cls.__args__)
    if origin is dict:
        length = generate(int, 1, 5)
        return dict(tuple(generate(typ) for typ in cls.__args__) for _ in range(length))
    if origin is Union:
        typ = random.choice(cls.__args__)
        return generate(typ)

    annotations = getattr(cls.__init__, "__annotations__", {})
    if len(annotations) == 0:
        raise ValueError(f"Don't know how to generate {cls}")

    # __init__ method is annotated (dataclass etc)
    return cls(**{name: generate(typ) for name, typ in annotations.items() if name != "return"})


@register(int)
def random_int(a: int = -5, b: int = 5):
    return random.randint(a, b)


@register(float)
def random_float(a: float = 0, b: float = 1.0):
    return random.random() * (b - a) + a


@register(str)
def random_str(length: int = 6):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


def random_hexdigits(nchar: int = 6) -> str:
    """
    Utility to generate nchar digit random hexadecimal string
    """
    return "".join(random.choice(string.hexdigits) for _ in range(nchar))


@register(datetime)
def random_datetime():
    return datetime.utcfromtimestamp(random_int(0, 1678094398))


# from https://en.wikipedia.org/wiki/Full_cycle
def cycle(seed: int, sample_size: int, increment: int) -> Iterator[int]:
    nb = seed
    for i in range(sample_size):
        nb = (nb + increment) % sample_size
        yield nb


def full_cycle_random_alpha_numeric_sequence(length: int = 5) -> Iterator[str]:
    """Generates a sequence of pseudo-random alpha-numeric characters that has
    a guaranteed period of 36 ** length (full cycle).
    """
    chars = string.digits + string.ascii_uppercase
    D = len(chars)
    N = D**length

    for nb in cycle(random.randint(0, N), N, increment=211):
        seq: List[int] = []
        while nb:
            seq.insert(0, nb % D)
            nb //= D
        yield "".join([chars[m] for m in seq])
