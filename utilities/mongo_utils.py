from contextlib import contextmanager
from functools import lru_cache
import json
import subprocess
from warnings import warn

from pymongo import MongoClient
from pymongo.collection import Collection
from azure.core.exceptions import ClientAuthenticationError

from utilities.aml_utils import NotRunningInAMLError, get_secret_from_workspace
from utilities.globals import (
    DEFAULT_AZURE_COSMOS_DB_ACCOUNT,
    DEFAULT_AZURE_KEYVAULT_NAME,
    DEFAULT_AZURE_RESOURCE_GROUP,
    DEFAULT_AZURE_SUBSCRIPTION_NAME,
)


TEMP_DATABASE_NAME = "test_db"

# global variable that controls the use of production vs. non-production (_test) Mongo DB
_use_production_database = False

# names taken from /projects folder to be consistent
# append to this list if a new project is created
ALL_PROJECTS = (
    "common",  # prefix for a database shared between projects
    "user",  # prefix for a database for a specific named user
    "coarsegraining",
    "compcat",
    "dft",
    "drugstruct",
    "dynami",
    "materials",
    "mlpotential",
    "oneqmc",
    "retrosynthesis",
    "sampling",
    "simulation",
    "timewarp",
    "toponet",
    "vasp",
)


@contextmanager
def use_production_database(enabled: bool = True):
    """Context manager that enables the use of production Mongo database.

    This is not thread safe.
    """
    global _use_production_database
    original_state = _use_production_database
    try:
        _use_production_database = enabled
        yield
    finally:
        _use_production_database = original_state


def production_database_is_used() -> bool:
    """Returns if production Mongo database is currently enabled."""
    return _use_production_database


def _get_cosmos_db_connection_string_from_azure_cli(
    subscription_name: str, resource_group: str, account_name: str
) -> str:
    return json.loads(
        subprocess.check_output(
            f"az cosmosdb keys list -n {account_name} -g {resource_group} "
            f" --subscription '{subscription_name}' --type connection-strings",
            shell=True,
        )
    )["connectionStrings"][0]["connectionString"]


def _get_cosmos_db_connection_string_from_keyvault(keyvault_name: str, account_name: str) -> str:
    import azure.keyvault.secrets
    from utilities.authentication import get_azure_creds

    secret_client = azure.keyvault.secrets.SecretClient(
        vault_url=f"https://{keyvault_name}.vault.azure.net",
        credential=get_azure_creds(),
    )
    return secret_client.get_secret(f"{account_name}-connection-string").value


def _get_cosmos_db_connection_string_from_workspace(account_name: str) -> str:
    return get_secret_from_workspace(f"{account_name}-connection-string")


def _get_cosmos_db_connection_string(
    subscription_name: str, resource_group: str, account_name: str, keyvault_name: str
) -> str:

    try:
        return _get_cosmos_db_connection_string_from_workspace(account_name)
    except NotRunningInAMLError:
        pass

    try:
        return _get_cosmos_db_connection_string_from_keyvault(keyvault_name, account_name)
    except (ModuleNotFoundError, ClientAuthenticationError):
        pass

    return _get_cosmos_db_connection_string_from_azure_cli(
        subscription_name, resource_group, account_name
    )


@lru_cache()
def get_mongo_client(
    subscription_name: str = DEFAULT_AZURE_SUBSCRIPTION_NAME,
    resource_group: str = DEFAULT_AZURE_RESOURCE_GROUP,
    account_name: str = DEFAULT_AZURE_COSMOS_DB_ACCOUNT,
    keyvault_name: str = DEFAULT_AZURE_KEYVAULT_NAME,
) -> MongoClient:
    """Gets a Mongo DB client for a Cosmos DB hosted in Azure.

    Args:
        subscription_name: name of the subscription (usually 'Molecular Dynamics')
        resource_group: name of the resource group (usually 'shared_infrastructure')
        account_name: database account name (e.g., 'msrmdscore')
        keyvault_name: name of the Azure key vault that stores the connection string.

    As a prerequisite, please install Azure CLI (https://docs.microsoft.com/en-us/cli/azure/)
    and sign in to azure by running `az login` in the terminal.

    Please see pymongo documentation to learn how to use the DB client:
    https://pymongo.readthedocs.io/en/stable/tutorial.html
    """
    return MongoClient(
        _get_cosmos_db_connection_string(
            subscription_name, resource_group, account_name, keyvault_name
        )
    )


@lru_cache()
def _check_database_name(database_name: str):
    """Checks database name follows naming conventions. If database already exists, check is skipped."""

    # Gets default msrmdscore Mongo DB client
    client = get_mongo_client()

    # check to see what databases already exist
    database_names = client.list_database_names()
    if database_name not in database_names:
        if not database_name.startswith(ALL_PROJECTS):
            warn(
                "Please consult docs/databases.md before creating a new database and follow naming conventions."
            )


def get_collection(database_name: str, collection_name: str) -> Collection:
    """Gets (possibly non-production) Mongo DB collection from database name and collection name."""

    # Gets default msrmdscore Mongo DB client
    client = get_mongo_client()

    # Use "_test" suffix if we are not inside `with use_production_database(): ...` block,
    # unless we are already using a temporary database for unit tests.
    # See docs/databases.md for more information on conventions
    if not _use_production_database and database_name != TEMP_DATABASE_NAME:
        database_name = f"{database_name}_test"

    # will warn, not raise, if naming conventions not followed
    _check_database_name(database_name)

    # if doesn't already exist, will create on insert of data
    database = client[database_name]

    # document collection. If doesn't already exist, will create on insert of data
    return database[collection_name]
