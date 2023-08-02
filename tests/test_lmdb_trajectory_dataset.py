from argparse import Namespace
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from timewarp.preprocess_lmdb import check_files, preprocess, get_output_data_dir
from timewarp.utils.dataset_utils import get_dataset
from timewarp.utils.config_utils import load_config
from timewarp.model_constructor import model_constructor
from timewarp.utils.loss_utils import get_log_likelihood
from timewarp.dataloader import moldyn_dense_collate_fn

from utilities.lmdb_utils import lmdb_read_metadata


@lru_cache
def get_model():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "gaussian_baseline.yaml"
    config = load_config(config_path)
    return model_constructor(config.model_config)


def eval_dataset(model, dataset):
    """Compute a hash over a dataset using a model. Run this function inside torch.no_grad()."""
    for batch in DataLoader(dataset, batch_size=1024, collate_fn=moldyn_dense_collate_fn):
        yield from [float(ll) for ll in get_log_likelihood(model, batch)]


def compare_datasets(dataset_ref, dataset):
    model = get_model()
    # torch.no_grad() should be outside the generators to prevent a race condition.
    with torch.no_grad():
        for loss_ref, loss in tqdm(
            zip(eval_dataset(model, dataset_ref), eval_dataset(model, dataset)),
            desc=f"Comparing datasets {dataset_ref.data_dir} and {dataset.data_dir}",
        ):
            assert loss_ref == loss


def check_datasets(datasets_ref, datasets):
    for dataset_ref, dataset in zip(datasets_ref, datasets):
        compare_datasets(dataset_ref, dataset)
        assert isinstance(dataset.num_atoms, list)
        assert len(dataset.num_atoms) == len(dataset)
        assert set(dataset.pdb_indices.keys()) == set(p.name for p in dataset_ref)
        assert sorted(sum(dataset.pdb_indices.values(), [])) == list(range(len(dataset)))


@pytest.mark.parametrize("step_width", [1000])
@pytest.mark.parametrize("dataset_name", ["T1"])
def test_load_lmdb_trajectory_dataset(dataset_name, step_width):
    datasets_ref = get_dataset(
        dataset_name, cache_dir=".data", step_width=step_width, lmdb=False, shuffle=False
    )
    datasets = get_dataset(
        dataset_name, cache_dir=".data", step_width=step_width, lmdb=True, shuffle=False
    )
    check_datasets(datasets_ref, datasets)


@pytest.mark.parametrize("step_width", [1])
@pytest.mark.parametrize("dataset_name", ["T1"])
def test_preprocess_dataset(dataset_name, step_width):
    with TemporaryDirectory() as tmpdir:
        args = Namespace(
            dataset=dataset_name,
            step_width=step_width,
            data_dir=None,
            cache_dir=".data",
            output_root_dir=tmpdir,
            splits=["train", "val", "test"],
            upload=False,
            num_partitions=2,
            num_workers=1,
            only_pdb_names=[],
            equal_data_spacing=False,
        )
        preprocess(args)
        check_files(args)
        datasets_ref = get_dataset(
            dataset_name, cache_dir=".data", step_width=step_width, lmdb=False, shuffle=False
        )
        datasets = get_dataset(
            dataset_name,
            data_dir=get_output_data_dir(args),
            step_width=step_width,
            lmdb=True,
            shuffle=False,
        )
        check_datasets(datasets_ref, datasets)


@pytest.mark.parametrize("step_width", [1])
@pytest.mark.parametrize("dataset_name", ["T1"])
def test_preprocess_dataset_in_two_steps(dataset_name, step_width):
    datasets_ref = get_dataset(
        dataset_name, cache_dir=".data", step_width=step_width, lmdb=False, shuffle=False
    )
    with TemporaryDirectory() as tmpdir:
        # Take the first half of each partition
        pdb_names = sum(
            (dataset.pdb_names_for_worker(i, 4) for dataset in datasets_ref for i in (0, 2)),
            (),
        )
        args1 = Namespace(
            dataset=dataset_name,
            step_width=step_width,
            data_dir=None,
            cache_dir=".data",
            output_root_dir=tmpdir,
            splits=["train", "val", "test"],
            upload=False,
            num_partitions=2,
            num_workers=1,
            only_pdb_names=pdb_names,
            equal_data_spacing=False,
        )
        preprocess(args1)
        data_dir = get_output_data_dir(args1)
        with pytest.raises(
            RuntimeError, match=f"Could not find LmdbTrajectoryDataset data in {data_dir}"
        ):
            # This fails because the datasets are not complete.
            get_dataset(
                dataset_name,
                data_dir=data_dir,
                step_width=step_width,
                lmdb=True,
                shuffle=False,
            )
        for split, dataset in zip(["train", "val", "test"], datasets_ref):
            db_path = data_dir / split / "data.0000.lmdb"
            pdb_indices = lmdb_read_metadata(db_path, "pdb_indices")
            assert set(pdb_indices.keys()).issubset(dataset.pdb_names_for_worker(0, 2))

        args2 = Namespace(
            dataset=dataset_name,
            step_width=step_width,
            data_dir=None,
            cache_dir=".data",
            output_root_dir=tmpdir,
            splits=["train", "val", "test"],
            upload=False,
            num_partitions=2,
            num_workers=1,
            # Take all
            only_pdb_names=[],
            equal_data_spacing=False,
        )
        preprocess(args2)
        datasets = get_dataset(
            dataset_name,
            data_dir=data_dir,
            step_width=step_width,
            lmdb=True,
            shuffle=False,
        )
        check_datasets(datasets_ref, datasets)


@pytest.mark.parametrize("step_width", [1])
@pytest.mark.parametrize("dataset_name", ["T1"])
def test_preprocess_dataset_in_two_steps_wrong_num_partitions(dataset_name, step_width):
    datasets_ref = get_dataset(
        dataset_name, cache_dir=".data", step_width=step_width, lmdb=False, shuffle=False
    )
    with TemporaryDirectory() as tmpdir:
        pdb_names = sum((dataset.pdb_names_for_worker(0, 2) for dataset in datasets_ref), ())
        args1 = Namespace(
            dataset=dataset_name,
            step_width=step_width,
            data_dir=None,
            cache_dir=".data",
            output_root_dir=tmpdir,
            splits=["train", "val", "test"],
            upload=False,
            num_partitions=1,
            num_workers=1,
            # Take the first half of each split
            only_pdb_names=pdb_names,
            equal_data_spacing=False,
        )
        preprocess(args1)

        args2 = Namespace(
            dataset=dataset_name,
            step_width=step_width,
            data_dir=None,
            cache_dir=".data",
            output_root_dir=tmpdir,
            splits=["train", "val", "test"],
            upload=False,
            num_partitions=2,
            num_workers=1,
            # Take all
            only_pdb_names=[],
            equal_data_spacing=False,
        )
        with pytest.raises(
            RuntimeError, match="num_partitions==2 was specified, but found 1 lmdb files"
        ):
            preprocess(args2)


def get_exclude_pdb_names():
    t1b = get_dataset("T1B", cache_dir=".data", step_width=1000)
    t1_large = get_dataset("T1-large", cache_dir=".data", step_width=1000)
    all_t1b_names = set(sum((dataset.raw_dataset.pdb_names for dataset in t1b), []))
    all_t1large_names = set(sum((dataset.raw_dataset.pdb_names for dataset in t1_large), []))
    return all_t1large_names - all_t1b_names


def test_t1_large_like_t1b_dataset():
    t1_large = get_dataset("T1-large", cache_dir=".data", step_width=1000)
    t1_large_like_t1b = get_dataset("T1-large-like-T1B", cache_dir=".data", step_width=1000)
    exclude_pdb_names = get_exclude_pdb_names()

    assert exclude_pdb_names == {
        "3zy1",
        "4g13",
        "6cf4",
        "6cg3",
        "6udw",
        "6uf8",
        "6ufu",
        "3c3h",
        "5vsg",
        "6av8",
        "6phj",
        "6phq",
        "2ag3",
        "3sgn",
        "5v63",
        "6php",
        "6sbw",
    }
    for dataset_ref, dataset in zip(t1_large, t1_large_like_t1b):
        assert (
            set(dataset.raw_dataset.pdb_names)
            == set(dataset_ref.raw_dataset.pdb_names) - exclude_pdb_names
        )
