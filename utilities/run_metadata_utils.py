import os
import sys
from typing import List, Optional

import git
from git import InvalidGitRepositoryError, GitCommandNotFound

from utilities.amlt_utils import AmuletJobInfo


def get_code_git_sha() -> str:
    """Retrives the code SHA from the source directory or environment
    variable set by Amulet.
    """
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.commit.hexsha
    except InvalidGitRepositoryError:
        pass

    # assume we are running with Amulet
    return os.environ.get("AMLT_CODE_GIT_SHA", "")


def get_amulet_job_info() -> AmuletJobInfo:
    """Retrieves job info from environment variables set by Amulet."""
    return AmuletJobInfo.from_current_environment()


def get_command() -> List[str]:
    """Returns a list of commands including the currently running script."""
    return [" ".join([sys.executable] + sys.argv)]


def add_git_suffix_maybe(name: str, repo_path: Optional[os.PathLike] = None) -> str:
    """
    Add Git related information as a suffix in `name` if a repo is found.

    Args:
        name: string to which the suffix will be added.
        repo_path: path of the repo. If not provided, `git.Repo` will be called with
            `search_parent_directories=True`.
    """
    try:
        # Extract the commit hash.
        if repo_path is None:
            repo = git.Repo(search_parent_directories=True)
        else:
            repo = git.Repo(repo_path)

        # Add output from `git describe --always --dirty`
        name += "_" + repo.git.describe(always=True, dirty=True)
    except (InvalidGitRepositoryError, GitCommandNotFound):
        print("Unable to add Git information to run name; leaving as is.")

    return name
