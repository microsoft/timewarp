# DataTable

DataTable is an abstraction that allows you to transparently interact with pandas DataFrame and Mongo DB. It is useful for example, when you have written a application that uses pandas and want to extend it to interact with Mongo DB while keeping the capability to deal with pandas DataFrame for debugging and testing.

## Quick start

The imports:
```python
from utilities.tables.pandas import PandasDataTable
from utilities.tables.mongo import MongoDataTable
```

To start from an empty table using pandas as the backend,
```python
table = PandasDataTable("output.pck")
```
(`output.pck` is the destination the content is eventually written to.)

To start from an empty or existing data hosted in our shared Mongo DB account, 
```python
table = MongoDataTable.create(
    subscription_name="Molecular Dynamics",
    resource_group="shared_infrastructure",
    account_name="msrmdscore",
    database_name="<your-database-name>",
    collection_name="<your-collection-name>",
)
```

The rest of this document does not distinguish if we are using `PandasDataTable` or `MongoDataTable` because they should work the same way.

## Examples

You can load a pickled dataframe using
```python
table.insert(pandas.read_pickle("input.pck").to_dict("records"))
```

Otherwise, `table.insert()` method accepts any sequence of dictionaries.

You can look for rows (documents) using
```python
table.filter({"dataset": "T1-large"})
table.filter({"config.learning_rate": 0.001})
```
etc. Note that `table.filter()` method returns a generator.

You can select which column values to return like
```python
table.filter({"dataset": "T1-large"}, select="min_validation_loss")
```

When there is no condition, we can use the brackets as a short hand like
```python
table["min_validation_loss"] # the same as table.filter({}, select="min_validation_loss")
```

You can update (or insert) a row using a unique id and a rule to apply to decide which document to choose. For example, if the rule is to choose the document with the largest value of `created_at` column within the same value of `unique_id` column,
```python
table.update_or_insert(new_documents, on="unique_id", choose=(max, "created_at"))
```
where `new_documents` is a list of dictionaries with the structure like
```python
new_documents = [
    ...
    {
        "unique_id": 1234,
        "created_at": datetime(2022, 4, 21, 15, 44, 8, 322430),
        ...
    },
    ...
]
```

The above command will insert the new document if the document with the same `unique_id` value (e.g., 1234) does not exist yet.

You can save the changes explicitly (at the moment this only matters for pandas; all updates for Mongo are instantaneous) by
```python
table.commit()
```

You can convert DataTable into pandas DataFrame by
```python
table.to_dataframe()
```
