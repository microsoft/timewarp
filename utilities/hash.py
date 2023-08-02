from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Dict, Optional, Set, Union

from monty.json import MontyEncoder, MSONable


def compute_hash(
    d: Union[Dict[str, Any], MSONable], exclude_fields: Optional[Set[str]] = None
) -> str:
    """Computes sha1 hash of a dictionary or a MSONable object.

    Args:
        d: input object to be hashed.
        exclude_fields: (optional) set of field names to be excluded from hash computation.

    Returns:
        the sha1 hash value.
    """
    exclude_fields = exclude_fields or set()
    if not isinstance(d, dict):
        d = d.as_dict()
    d = {key: value for key, value in d.items() if key not in exclude_fields}
    return sha1(MontyEncoder(sort_keys=True).encode(d).encode("utf-8")).hexdigest()


@dataclass(frozen=True, unsafe_hash=False)
class Hashable(MSONable):
    # NOTE: if we want to use Hashable class instances as keys to a dictionary
    # or use in sets, we need to define self.__hash__ and duplicate definition
    # in all classes that inherit Hashable and are dataclasses

    def _as_dict(self) -> Dict[str, Any]:
        # make class-specific changes to output from MSONable.as_dict here
        # before hashing the result in Hashable.as_dict

        return super().as_dict()

    def as_dict(self) -> Dict[str, Any]:
        # get all attributes that we want to serialize, except @instance_hash
        out = self._as_dict()

        out["@instance_hash"] = self.compute_hash(out)

        return out

    def compute_hash(self, d: Optional[Dict[str, Any]] = None) -> str:
        if not d:
            d = self._as_dict()
        return compute_hash(d)
