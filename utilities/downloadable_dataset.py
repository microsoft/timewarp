import os
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Optional, Union, TypeVar, List, Any

# from utilities.blob_storage import BlobClient
from utilities.common import StrPath
# from utilities.globals import DEFAULT_AZURE_STORAGE_ACCOUNT, DEFAULT_AZURE_SUBSCRIPTION_NAME


class DatasetNotFoundError(Exception):
    def __init__(self, cls, data_dir: StrPath):
        super().__init__(
            f"Could not find {cls.__qualname__} data in {data_dir}. Try download=True."
        )
        self.cls = cls.__qualname__
        self.data_dir = data_dir


class DatasetCacheInvalid(Exception):
    def __init__(self, cls, data_dir: StrPath):
        super().__init__(
            f"{cls.__qualname__} data in {data_dir} is invalid/outdated, please delete cache."
        )
        self.cls = cls.__qualname__
        self.data_dir = data_dir


DatasetStatsType = TypeVar("DatasetStatsType")


@dataclass(frozen=True)
class DatasetStats:
    """Dataset metadata to verify the local cache is valid"""

    num_files: Optional[int] = field(default=None)
    num_points: Optional[int] = field(default=None)


@dataclass(frozen=True)
class DownloadableDatasetMixin:
    data_dir: StrPath
    downloader: Any = field(default=None)


@dataclass(frozen=True)
class ContainerBlobPath:
    container_name: str
    blob_path: str

    @staticmethod
    def from_string(container_blob_path_str: str):
        """
        Creates an instance from a string formatted as "[container name]:/path/to/blob"
        """
        return ContainerBlobPath(*container_blob_path_str.split(":/"))

    @staticmethod
    def from_local_dir(local_dir: StrPath, cache_dir: StrPath):
        container_name, *blob_path_parts = Path(local_dir).relative_to(cache_dir).parts
        return ContainerBlobPath(container_name, "/".join(blob_path_parts))


def create_blob_client(data_dir: StrPath, container_blob_path: ContainerBlobPath):
    pass
    # return BlobClient(
    #     DEFAULT_AZURE_SUBSCRIPTION_NAME,
    #     DEFAULT_AZURE_STORAGE_ACCOUNT,
    #     container_blob_path.container_name,
    #     container_blob_path.blob_path,
    #     local_dir=str(data_dir),
    # )


class DownloadableDataset(DownloadableDatasetMixin, Generic[DatasetStatsType]):
    def __init__(self, data_dir: StrPath, downloader: Optional[Any] = None, **_kwargs):
        super().__init__(data_dir=data_dir, downloader=downloader)

    @staticmethod
    @abstractmethod
    def check_dir(data_dir: StrPath, expected_dataset_stats: DatasetStatsType):
        """
        Checks if data_dir contains the files corresponding to expected number of data points, etc.
        """

    @classmethod
    def possibly_download_dataset(
        cls,
        data_dir: StrPath,
        container_blob_path: Union[ContainerBlobPath, str],
        expected_dataset_stats: DatasetStatsType,
        download: bool = True,
        **kwargs,
    ):
        """
        Creates an instance of a downloadable dataset.

        Args:
            cls: class of dataset to be constructed
            data_dir: where to store the data locally
            container_blob_path: ContainerBlobPath or string formatted as "[container name]:/path/to/blob"
            download: allow download if dataset does not exist (default: True)
            **kwargs can be provided for additional arguments (e.g., step_width)
        """
        if os.path.isdir(data_dir) and cls.check_dir(data_dir, expected_dataset_stats):
            return cls(data_dir=data_dir, downloader=None, **kwargs)
        if not download:
            raise DatasetNotFoundError(cls, data_dir)
        print(
            f"I: Could not find {cls.__qualname__} data in {data_dir}. Creating a downloader for {container_blob_path}"
        )
        downloader = create_blob_client(
            data_dir,
            container_blob_path=ContainerBlobPath.from_string(container_blob_path)
            if isinstance(container_blob_path, str)
            else container_blob_path,
        )
        return cls(data_dir=data_dir, downloader=downloader, **kwargs)

    def download_all(self, overwrite=False):
        """
        Download all files from blob storage.

        Args:
            overwrite: (optional) overwrite any existing local files.
        """
        assert self.downloader is not None
        self.downloader.download_all(overwrite=overwrite)

    def upload_all(self, pattern: str, cache_dir: StrPath, overwrite: bool = False):
        """
        Upload all files in data_dir that matches the glob pattern.

        The destination is inferred from the provided cache_dir. That is, we assume that
            data_dir = cache_dir / [container_name] / [blob_path]

        Args:
            pattern: glob pattern of files to be uploaded.
            cache_dir: dataset cache root directory.
            overwrite: (optional) overwrite any existing files.
        """
        uploader = create_blob_client(
            self.data_dir, ContainerBlobPath.from_local_dir(self.data_dir, cache_dir)
        )
        for fn in Path(uploader.local_dir).glob(pattern):
            if os.path.isfile(fn):
                uploader.upload_file(str(fn), overwrite=overwrite)


class BlobFileMatcher:
    """A remote blob with the ability to automatically match files and easily refer to these
    matches by prespecified names.

    Args:
        client (:class:`utilities.downloadable_dataset.BlobClient`): Client.
        path (str): Path to the blob relative to the client.
        queries(list[dict]): A list of dictionaries with two keys: `"name"` and `"match"`. Every
            element in this list is a query for one or more files. More specifically, queries
            refer to files by matching the _end_ of a file path. In every dictionary, the key
            `"match"` refers to the _end_ of a file path, e.g. `".ckpt"`, and `"name"` refers
            to how you want to refer to all files satisfying this condition. You will use the values
            for `"name"` in the methods of this class. For example, see
            :meth:`.BlobFileMatcher.local_path` and :meth:`.BloblFileMatcher.local_paths`.

    Attributes:
        client (:class:`utilities.downloadable_dataset.BlobClient`): Client.
        cache (str): Path to the local cache. This is where files will be stored locally.
        path (str): Path to the blob relative to the client.
        matches (dict): For every query, the remote paths satisfying the query's condition.
    """

    def __init__(self, client: Any, path: str, queries: List[dict]) -> None:
        self.client = client
        self.cache = client.local_dir
        self.path = path
        self.matches: dict = {query["name"]: [] for query in queries}

        for f in client.list_files(path):
            # Check if there a match with any of the queries.
            match = None
            for query in queries:
                if f.lower().endswith(query["match"].lower()):
                    match = query

            if match is None:
                # No match was found. Skip this file.
                continue
            else:
                # Match was found. Record the path.
                self.matches[match["name"]].append(f)

    def local_paths(self, name: str) -> List[str]:
        """Get all local paths corresponding to a query.

        Args:
            name (str): Name of the query.

        Returns:
            list[str]: List of local paths.
        """
        local_paths = []

        for f in self.matches[name]:
            remote_path = os.path.join(self.path, f)
            local_path = os.path.join(self.cache, remote_path)

            # If the local path does not yet exist, download the file.
            if not os.path.isfile(local_path):
                self.client.download_file(remote_path)

            local_paths.append(local_path)

        return local_paths

    def local_path(self, name: str) -> str:
        """Get the unique local path corresponding to a query.

        Args:
            name (str): Name of the query.

        Returns:
            str: Unique local path corresponding to the query.
        """
        local_paths = self.local_paths(name)
        if len(local_paths) != 1:
            raise AssertionError(
                f"There are {len(local_paths)} local paths associated to" f' "{name}".'
            )
        return local_paths[0]
