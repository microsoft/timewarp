from typing import List

import pytest

try:
    from pymatgen.core import Lattice, Structure
except ModuleNotFoundError:
    pass

try:
    from amlt.config import AMLTConfig, SkuInfo

    from utilities.amlt_config_utils import get_amlt_config
except ModuleNotFoundError:
    pass

from utilities import amlt_utils, mock_amlt_utils, mongo_utils
from utilities.amlt_utils import amlt_in_env, amlt_project


def get_h2() -> "Structure":
    # mp-632172
    a = 2.228394
    b = 1.573944
    return Structure(
        Lattice(matrix=[[-a, a, b], [a, -a, b], [a, a, -b]]),
        ["H", "H"],
        [(0.0000, -0.0000, 2.7728), [0.0000, -0.0000, 0.3751]],
    )


@pytest.fixture(scope="function")
def structure() -> "Structure":
    return get_h2()


@pytest.fixture(scope="function")
def structures() -> List["Structure"]:
    s = get_h2()
    s.perturb(0.1)
    return [s, get_h2()]


@pytest.fixture
def amlt_config() -> "AMLTConfig":
    with amlt_project("test"):
        return get_amlt_config(
            target_name="ci-cluster",
            environment_name="utilities-test",
            sku=SkuInfo.from_string("C1"),
            only_upload_projects=["utilities"],
        )


@pytest.fixture
def mock_mongo_db(mocker, mongodb):
    """
    patches get_mongo_client to return a local mongodb instance provided through pytest-mongo
    """
    mocker.patch.object(
        mongo_utils,
        "get_collection",
        lambda database_name, collection_name: mongodb[collection_name],
    )
    return mongodb


@pytest.fixture
def use_mock_amlt_remote() -> bool:
    """default value for use_mock_amlt_remote."""
    return True


@pytest.fixture
def mock_amlt_remote(mocker, use_mock_amlt_remote: bool):
    """
    patches amlt_utils to use mocked versions of call_in_config and map_in_config
    """
    if use_mock_amlt_remote:
        mocker.patch.object(amlt_utils, "amlt_project", mock_amlt_utils.amlt_project)
        mocker.patch.object(amlt_utils, "call_in_config", mock_amlt_utils.call_in_config)
        mocker.patch.object(amlt_utils, "map_in_config", mock_amlt_utils.map_in_config)
    else:
        if not amlt_in_env:
            pytest.skip("requires amlt")
