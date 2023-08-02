"""
This measures the time required to copy individual files from blob storage into the machine the script runs on.
The script outputs a png file with a plot that shows time and speed for varying file size.
"""

import argparse
import datetime
import logging
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from rich.logging import RichHandler

from utilities.blob_storage import BlobClient, download_stream

LOG = logging.getLogger(__name__)


def main():
    random.seed(0)
    sns.set_style("whitegrid")

    parser = argparse.ArgumentParser(
        description=(
            "Download all files from a blob storage root into a temporary directory, "
            "and benchmark the copy speed."
        )
    )
    parser.add_argument("--subscription-name", type=str, default="Molecular Dynamics")
    parser.add_argument("--storage-account-name", type=str, default="feynman0storage")
    parser.add_argument("--container-name", type=str, default="dm21data")
    parser.add_argument(
        "--blob-root",
        type=str,
        default="processed_split/w4-11",
        help="Root directory from which all data will be copied.",
    )

    parser.add_argument("--output-file", type=str, default="blobstorage-timing.png")
    parser.add_argument(
        "--exec-location",
        "-l",
        type=str,
        required=True,
        help="Geo location of the node running this script, mentioned in the output plot.",
    )
    parser.add_argument(
        "--max-num-files",
        type=int,
        default=100,
        help="If the `blob-root` contains more files than this, they will be subsampled.",
    )

    args = parser.parse_args()
    today = datetime.datetime.now().isoformat()[:10]
    tmp_dir = tempfile.TemporaryDirectory().name

    blob = BlobClient(
        args.subscription_name,
        args.storage_account_name,
        args.container_name,
        args.blob_root,
        tmp_dir,
    )

    df = run_benchmark(blob, max_num_files=args.max_num_files)
    df["file_size"] /= 1e6  # Convert to megabytes.
    df["speed"] = df["file_size"] / df["download_time"]

    df.sort_values("file_size", inplace=True)
    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))

    df.plot.scatter("file_size", "download_time", ax=ax[0])
    ax[0].set(xlabel="File size [MB]", ylabel="Time [s]")

    df.plot.scatter("file_size", "speed", ax=ax[1])
    ax[1].set(xlabel="File size [MB]", ylabel="Average speed [MB/s]")

    fig.suptitle(
        f"Blob Storage Copy Times (single files from {args.storage_account_name} to {args.exec_location} on {today})"
    )

    fig.tight_layout()
    fig.savefig(args.output_file, dpi=300)

    LOG.info(f"Output written to {args.output_file}.")


@dataclass
class TimingRecord:
    file_size: float  # In bytes.
    download_time: float  # In seconds.


def run_benchmark(blob: BlobClient, max_num_files: Optional[int] = None) -> pd.DataFrame:
    """"""
    Path(blob.local_dir).mkdir()
    files = list(blob.list_files())
    random.shuffle(files)
    if max_num_files is not None:
        file_subset = files[:max_num_files]
        LOG.info(f"Downloading {len(file_subset)} out of {len(files)} files for this benchmark.")
    else:
        file_subset = files
        LOG.info(f"Downloading {len(files)} files for this benchmark.")

    records = []
    for file in file_subset:
        start_time = time.time()
        blob_name = blob.make_full_blob_name(file)
        with download_stream(blob.container_client, blob_name, max_concurrency=4) as stream:
            with open(Path(blob.local_dir) / file, "wb+") as outfile:
                stream.readinto(outfile)
        end_time = time.time()
        file_size = (Path(blob.local_dir) / file).stat().st_size  # In bytes.
        records.append(TimingRecord(file_size=file_size, download_time=end_time - start_time))

    return pd.DataFrame(records)


if __name__ == "__main__":
    FORMAT = "%(message)s"
    logging.basicConfig(
        level=logging.ERROR, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    LOG.setLevel(logging.INFO)
    main()
