from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from typing_extensions import Self

from utilities.amlt_utils import AmuletJobInfo
from utilities.hash import Hashable
from utilities.run_metadata_utils import get_code_git_sha, get_command


@dataclass(frozen=True)
class BaseCreatedBy:
    # name of method that created this structue
    name: str

    # code version
    version: str

    # exact commit in feynman repo
    commit_id: str = field(default_factory=str)  # default = ''

    # job commands that led to creation of this document
    command: List[str] = field(default_factory=list)  # default = []

    # any calculation-specific parameter files (eg from DFT)
    input_files: Optional[Dict[str, str]] = None

    # amulet job info
    amulet_project_name: str = ""
    amulet_experiment_name: str = ""
    amulet_job_name: str = ""
    amulet_description: str = ""
    amulet_output_blob_uri: str = ""

    # optional info about the checkpoint used
    model_checkpoint_info: Dict[str, Any] = field(default_factory=dict)

    # miscellaneous (human readable) comments
    comment: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_current_environment(
        cls,
        name: str,
        version: str,
        input_files: Optional[Dict[str, str]] = None,
        model_checkpoint_info: Optional[Dict[str, Any]] = None,
        comment: Optional[Dict[str, Any]] = None,
    ) -> Self:
        job_info = AmuletJobInfo.from_current_environment()
        return cls(
            name=name,
            version=version,
            commit_id=get_code_git_sha(),
            command=get_command(),
            input_files=input_files or {},
            amulet_project_name=job_info.project_name,
            amulet_experiment_name=job_info.experiment_name,
            amulet_job_name=job_info.job_name,
            amulet_description=job_info.description,
            amulet_output_blob_uri=job_info.output_blob_uri,
            model_checkpoint_info=model_checkpoint_info or {},
            comment=comment or {},
        )


class DefaultCreatedBy(BaseCreatedBy, Hashable):
    ...
