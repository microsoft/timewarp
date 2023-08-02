import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

try:
    from amlt.config import AMLTConfig
    from amlt.helpers.parallel import RateLimit
except ModuleNotFoundError:
    pass

from utilities.amlt_utils import (
    AmuletJobInfo,
    AmuletRemoteFuture,
    AmuletRemoteResult,
    _amlt_remote_exp_name_from_function,
    maybe_format_random_string,
)
from utilities.common import unique_item
from utilities.globals import DEFAULT_AZURE_STORAGE_ACCOUNT

ResultType = TypeVar("ResultType")

_current_project_name = "mock-project"
_current_experiment_name = "mock-experiment"
_current_job_name = "mock-job"


def current_project_name() -> str:
    return _current_project_name


def current_experiment_name() -> str:
    return _current_experiment_name


def current_job_name() -> str:
    return _current_job_name


@contextmanager
def amlt_project(project_name: str) -> Generator[None, None, None]:
    global _current_project_name
    prev_project_name = _current_project_name
    try:
        _current_project_name = project_name
        yield
    finally:
        _current_project_name = prev_project_name


@contextmanager
def amlt_experiment_job(experiment_name: str, job_name: str) -> Generator[None, None, None]:
    global _current_experiment_name, _current_job_name
    prev_experiment_name = _current_experiment_name
    prev_job_name = _current_job_name
    try:
        _current_experiment_name = experiment_name
        _current_job_name = job_name
        yield
    finally:
        _current_job_name = prev_job_name
        _current_experiment_name = prev_experiment_name


@dataclass(frozen=True)
class MockJobModel:
    name: str
    submit_args: Dict[str, Any]


@dataclass(frozen=True)
class MockJobsClient:
    configs: List[MockJobModel]

    @property
    def names(self) -> List[str]:
        return [config.name for config in self.configs]

    @property
    def config(self) -> MockJobModel:
        return unique_item(self.configs)


@dataclass(frozen=True)
class MockJobPortalInfo:
    job_name: str
    is_terminated: bool = True


class MockAmuletRemoteFuture(AmuletRemoteFuture[ResultType]):
    def __init__(
        self,
        experiment_name: str,
        configs: List[MockJobModel],
        source_functions: List["partial[ResultType]"],
        description: Optional[str] = None,
    ):
        super().__init__(
            project_name=current_project_name(),
            storage_account_name=DEFAULT_AZURE_STORAGE_ACCOUNT,
            container_name="container",
            registry_name="projects",
            experiment_name=experiment_name,
            _job_names=tuple([maybe_format_random_string(config.name) for config in configs]),
        )
        self.description = description or "mock amulet experiment."
        self.configs = configs
        self.source_functions = source_functions

    @property
    def jobs_client(self) -> MockJobsClient:
        return MockJobsClient(self.configs)

    def source_function_for_job(self, job_name: str) -> "partial[ResultType]":
        return self.source_functions[self.job_names.index(job_name)]

    @property
    def job_names(self) -> List[str]:
        return list(self._job_names)

    @property
    def statuses(self) -> List[str]:
        return ["pass" for _ in self.job_names]

    @property
    def portal_infos(self) -> List[MockJobPortalInfo]:
        return [MockJobPortalInfo(job_name) for job_name in self.job_names]

    @property
    def portal_urls(self) -> List[str]:
        return [f"https://ml.azure.com/runs/{job_name}" for job_name in self.job_names]

    @overload
    async def async_get_unique(
        self, polling_interval: float = 30, raise_on_remote_error: Literal[True] = True
    ) -> Tuple[ResultType, AmuletJobInfo]:
        ...

    @overload
    async def async_get_unique(
        self, polling_interval: float = 30, raise_on_remote_error: Literal[False] = False
    ) -> Tuple[Optional[ResultType], AmuletJobInfo]:
        ...

    def async_get_unique(
        self, polling_interval: float = 0.001, raise_on_remote_error: bool = True
    ) -> Coroutine[Any, Any, Tuple[Optional[ResultType], AmuletJobInfo]]:
        return super().async_get_unique(polling_interval=polling_interval)

    def get(
        self,
        job_names: Union[None, str, List[str]] = None,
        raise_on_remote_error: bool = True,
        polling_interval: float = 30,
    ) -> List[AmuletRemoteResult]:
        if isinstance(job_names, str):
            job_names = [job_names]
        job_names = job_names or self.job_names
        results = []
        for job_name in job_names:
            with amlt_experiment_job(self.experiment_name, job_name):
                results.append(
                    AmuletRemoteResult(
                        result=self.source_function_for_job(job_name)(),
                        job_info=AmuletJobInfo(
                            project_name=self.project_name,
                            experiment_name=self.experiment_name,
                            job_name=job_name,
                            description=self.description,
                            output_blob_uri="blob-storage://feynman0storage/container/path/to/blob",
                            portal_url=self.portal_urls[self.job_names.index(job_name)],
                        ),
                    )
                )
        return results


def call_in_config(
    config: "AMLTConfig", func: Callable[..., ResultType], *args, **kwargs
) -> MockAmuletRemoteFuture:
    p = partial(func, *args, **kwargs)
    return MockAmuletRemoteFuture(
        experiment_name=config.exp_name or _amlt_remote_exp_name_from_function(func),
        configs=[
            MockJobModel(
                name=f"mock-job-for-{p.func.__qualname__}_{{random_string}}",
                submit_args=config.jobs[0].submit_args,
            )
        ],
        source_functions=[p],
        description=config.description,
    )


def map_in_config(
    config: "AMLTConfig",
    func: Callable[..., ResultType],
    iterable: Iterable,
    *iterables: Iterable,
    job_name_func: Optional[Callable[..., str]] = None,
    rate_limit: Optional["RateLimit"] = None,
    prefer_hyperdrive: bool = False,
) -> MockAmuletRemoteFuture:
    if rate_limit:
        warnings.warn("Mocked Amulet does not support rate limiting.")
    if prefer_hyperdrive:
        warnings.warn("Mocked Amulet does not support hyperdrive.")

    partials = [partial(func, arg, *args) for arg, *args in zip(iterable, *iterables)]
    configs = [
        MockJobModel(
            name=job_name_func(arg, *args) if job_name_func else f"mock-job-for-{p}",
            submit_args=config.jobs[0].submit_args,
        )
        for p, arg, *args in zip(partials, iterable, *iterables)
    ]
    return MockAmuletRemoteFuture(
        experiment_name=config.exp_name or _amlt_remote_exp_name_from_function(func),
        configs=configs,
        source_functions=partials,
        description=config.description,
    )
