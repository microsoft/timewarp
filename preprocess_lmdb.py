import argparse
from dataclasses import dataclass, field
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from timewarp.datasets import TrajectoryIterableDataset
from timewarp.datasets.lmdb_dataset import init_all_metadata
from timewarp.utils.dataset_utils import get_dataset, get_dataset_metadata, get_container_blob_path
from utilities.lmdb_utils import (
    check_and_init_metadata,
    ensure_metadata,
    get_length,
    list_db_paths,
    lmdb_get,
    lmdb_open,
    lmdb_read_metadata,
    write_data_points_to_lmdb,
)


def get_datasets(args, **kwargs):
    return get_dataset(
        args.dataset,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        step_width=args.step_width,
        splits=args.splits,
        shuffle=False,
        equal_data_spacing=args.equal_data_spacing,
        **kwargs,
    )


def get_output_lmdb_datasets(args, **kwargs):
    return get_dataset(
        args.dataset,
        data_dir=get_output_data_dir(args),
        step_width=args.step_width,
        splits=args.splits,
        shuffle=False,
        lmdb=True,
        equal_data_spacing=args.equal_data_spacing,
        **kwargs,
    )


# This is a workaround for the issue that we cannot pass a lambda or a generator to starmap.
@dataclass(frozen=True)
class DataPointAndMetadataWriter:
    dataset: TrajectoryIterableDataset
    num_partitions: int
    only_pdb_names: Sequence[str] = field(default_factory=list)

    def __call__(self, partition_id: int, db_path: str) -> int:
        pdb_names = self.dataset.pdb_names_for_worker(partition_id, self.num_partitions)
        if len(self.only_pdb_names) > 0:
            pdb_names = sorted(set(pdb_names) & set(self.only_pdb_names))

        if os.path.isfile(db_path):
            print(f"Ensuring db {db_path} has all metadata...")
            ensure_metadata(db_path, metadata=init_all_metadata(self.dataset.step_width))

            # Check pdb_names in existing dataset
            pdb_indices = lmdb_read_metadata(db_path, "pdb_indices")

            # Only keep the names not already in the database
            pdb_names = sorted(set(pdb_names) - set(pdb_indices.keys()))

        if len(pdb_names) == 0:
            # nothing to do
            return 0

        print(f"Need to preprocess trajectory data for {len(pdb_names)} proteins")

        return write_data_points_to_lmdb(
            db_path,
            self.dataset.make_iterator_for_pdb_names(pdb_names),
            pid=partition_id,
            metadata=init_all_metadata(self.dataset.step_width),
        )


def get_output_data_dir(args):
    metadata = get_dataset_metadata(args.dataset)
    cb = get_container_blob_path(
        root_blob_path=metadata.root_blob_path,
        step_width=args.step_width,
        lmdb=True,
    )
    return Path(args.output_root_dir or args.cache_dir) / cb.container_name / cb.blob_path


def preprocess(args):
    datasets = get_datasets(args, lmdb=False, download=True)
    output_data_dir = get_output_data_dir(args)
    for split, dataset in zip(args.splits, datasets):
        print(f"Processing {split} split...")
        num_existing_files = len(list_db_paths(Path(output_data_dir) / split))
        if num_existing_files > 0 and num_existing_files != args.num_partitions:
            raise RuntimeError(
                f"num_partitions=={args.num_partitions} was specified, but found {num_existing_files} lmdb files in {Path(output_data_dir) / split}"
            )
        db_paths = [
            output_data_dir / split / f"data.{pid:04d}.lmdb" for pid in range(args.num_partitions)
        ]

        os.makedirs(os.path.dirname(db_paths[0]), exist_ok=True)

        writer = DataPointAndMetadataWriter(dataset, args.num_partitions, args.only_pdb_names)
        if args.num_workers > 1:
            pool = mp.Pool(args.num_workers)
            num_points = pool.starmap(writer, enumerate(db_paths))
        else:
            num_points = [writer(i, db_path) for i, db_path in enumerate(db_paths)]
        print(f"Written {num_points} data points to {db_paths}")


def check_files(args):
    output_data_dir = get_output_data_dir(args)
    datasets = get_datasets(args, lmdb=False, download=True)
    for split, dataset in zip(args.splits, datasets):
        total_length = 0
        pdb_names = []
        for db_path in list_db_paths(Path(output_data_dir) / split):
            with lmdb_open(db_path) as db:
                length = get_length(db)
                print(f"Checking {db_path}: length={length}...")
                new_metadata = check_and_init_metadata(
                    db, init_all_metadata(args.step_width), length, return_all=False, verbose=True
                )
                total_length += length
                if len(new_metadata) > 0:
                    print(f"missing metadata: {[m.name for m in new_metadata]}")
                with db.begin() as txn:
                    pdb_names.extend(lmdb_get(txn, "pdb_indices", {}).keys())
        expected_pdb_names = dataset.pdb_names_for_worker(0, 1)
        print(
            f"total length for {split} is {total_length} (number of pdb_names is {len(pdb_names)} compared to {len(expected_pdb_names)})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, type=str, help="Name of the dataset to preprocess"
    )
    parser.add_argument(
        "--step_width",
        required=True,
        type=int,
        help="How large of a time-step to predict into the future",
    )
    parser.add_argument(
        "--data_dir", default=None, help="Where the dataset is located if using blobfuse."
    )
    parser.add_argument(
        "--cache_dir", default=".data", help="Where the local cache of the dataset is stored."
    )
    parser.add_argument(
        "--output_root_dir",
        default=None,
        help="(optional) Where to write the preprocessed dataset to. If unspecified, it is inferred from --data_dir and --cache_dir.",
    )
    parser.add_argument("--splits", type=str, nargs="+", help="Splits to be preprocessed.")
    parser.add_argument("--upload", action="store_true", help="Upload the dataset to blob storage.")
    parser.add_argument(
        "--num_partitions", type=int, default=1, help="Number of .lmdb files to produce."
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers to process the data."
    )
    parser.add_argument(
        "--only_pdb_names",
        type=str,
        default=[],
        nargs="*",
        help="Protein names to be processed.",
    )
    parser.add_argument(
        "--check-files", action="store_true", help="Check raw LMDB files stored under --data_dir"
    )
    parser.add_argument(
        "--equal_data_spacing",
        action="store_true",
        help="If specified, conditioning-target data pairs will be equally spaced.",
    )
    args = parser.parse_args()

    if args.check_files:
        return check_files(args)

    # Check if the dataset already exists
    try:
        datasets = get_output_lmdb_datasets(args, download=False)
    except RuntimeError:
        preprocess(args)
        datasets = get_output_lmdb_datasets(args, download=False)

    if args.upload:
        for dataset in datasets:
            dataset.upload_all(
                "*.lmdb", cache_dir=args.output_root_dir or args.cache_dir, overwrite=True
            )


if __name__ == "__main__":
    main()
