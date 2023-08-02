from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple, Union

from pandas import DataFrame
from pymongo import MongoClient

from .base import DataTable
from utilities.common import get_nested_item
from utilities.mongo_utils import get_mongo_client


@dataclass(frozen=True)
class MongoDataTable(DataTable):
    mongo_client: MongoClient
    database_name: str
    collection_name: str

    @staticmethod
    def create(
        subscription_name: str,
        resource_group: str,
        account_name: str,
        database_name: str,
        collection_name: str,
    ):
        mongo_client = get_mongo_client(subscription_name, resource_group, account_name)
        return MongoDataTable(mongo_client, database_name, collection_name)

    @property
    def collection(self):
        return self.mongo_client[self.database_name][self.collection_name]

    def insert(self, entities: Iterable[Dict]):
        self.collection.insert_many(entities)

    def __len__(self):
        return self.collection.count_documents({})

    def __getitem__(self, column: Union[str, List[str]]) -> Iterator[Any]:
        return self.filter({}, select=column)

    def __contains__(self, column: str) -> bool:
        return self.collection.find_one({column: {"$exists": True}}) is not None

    def filter(
        self, filters: Dict[str, Any], select: Union[None, str, List[str]] = None
    ) -> Iterator[Any]:
        projection = (
            {}
            if select is None
            else {select: True}
            if isinstance(select, str)
            else {s: True for s in select}
        )
        projection.update({"_id": False})  # do not include _id
        filtered = self.collection.find(filters, projection=projection)
        if isinstance(select, str):
            return (get_nested_item(e, select) for e in filtered)
        elif isinstance(select, list):
            return ({col: get_nested_item(e, col) for col in select} for e in filtered)
        else:
            # return a list of dicts
            return filtered

    def update_or_insert(self, entities: Iterable[Dict], on: str, choose: Tuple[Callable, str]):
        op, column = choose
        for entity in entities:
            existing_values = list(self.filter({on: entity[on]}, select=column))
            if len(existing_values) == 0:
                # The summary doesn't exist in the DB. We need to insert.
                self.insert([entity])
                continue
            assert len(existing_values) == 1
            existing_value = existing_values[0]
            if op([entity[column], existing_value]) == entity[column]:
                # Need to update
                self.collection.update_one({on: entity[on]}, {"$set": entity})

    def commit(self):
        # Nothing to do.
        pass

    def to_dataframe(self) -> DataFrame:
        return DataFrame(self.filter({}))
