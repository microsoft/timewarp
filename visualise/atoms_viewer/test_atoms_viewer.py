import sys
from typing import Any, Dict

import pytest as pytest

from . import _dict_to_style

# Skip tests in CI if IPython module is not present
pytestmark = pytest.mark.skipif("IPython" not in sys.modules, reason="requires the IPython library")


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ({"a": 3}, "a:3"),
        ({"width": "400px", "height": "300px"}, "width:400px;height:300px"),
    ],
)
def test_dict_to_style(test_input: Dict[str, Any], expected: str) -> None:
    assert _dict_to_style(test_input) == expected
