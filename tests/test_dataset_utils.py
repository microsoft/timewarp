import os
from pathlib import Path
import pytest
from typing import Optional

from timewarp.utils.dataset_utils import get_dataset
from utilities import unique_item

TESTDATA_DIR = Path(os.path.dirname(__file__)) / ".." / "testdata"


def test_data_exist():
    dataset = get_dataset(
        dataset_name="test", data_dir=TESTDATA_DIR, cache_dir=Path(".data"), step_width=1
    )[0]
    assert dataset.raw_dataset.pdb_names == ("2olx",)


def test_data_does_not_exist():
    with pytest.raises(
        RuntimeError,
        match=f"Could not find TrajectoryIterableDataset data in {TESTDATA_DIR / 'foo'}.",
    ):
        get_dataset(
            dataset_name="test",
            data_dir=TESTDATA_DIR / "foo",
            cache_dir=Path(".data"),
            step_width=1,
        )


@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize("shuffle", [None, False, True])
@pytest.mark.parametrize("lmdb", [False, True])
def test_can_load_dataset(split: str, shuffle: Optional[bool], lmdb: bool):
    dataset_name = "T1"
    dataset = unique_item(
        get_dataset(
            dataset_name,
            cache_dir=Path(".data"),
            step_width=1000,
            splits=[split],
            shuffle=shuffle,
            lmdb=lmdb,
        )
    )
    if not lmdb:
        if shuffle is None:
            assert dataset.shuffle == (split == "train")
        else:
            assert dataset.shuffle == shuffle
