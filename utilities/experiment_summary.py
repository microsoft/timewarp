import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from operator import itemgetter
import os
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
from typing_extensions import Literal, get_args

try:
    from amlt.globals import DEFAULT_OUTPUT_DIR
except ImportError:
    pass


import pandas
from pandas import DataFrame
import plotly.express as px
import plotly.graph_objects as go

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
    ScalarEvent,
)
import yaml

from utilities import find_files

try:
    from utilities.amlt_utils import (
        get_amlt_status,
        guess_exp_name_job_name,
        checkout_project,
        pull_results,
    )
except ModuleNotFoundError:
    pass


@lru_cache(None)
def load_data(logdir: str) -> EventAccumulator:
    ea = EventAccumulator(logdir)
    ea.Reload()
    print(f"Found tags: {ea.Tags()} in {logdir}")
    return ea


def load_scalars(
    logdir: str, names_to_report: Iterable[str]
) -> Iterator[Tuple[str, List[ScalarEvent]]]:
    ea = load_data(logdir)
    for name in ea.Tags()["scalars"]:
        if name not in names_to_report:
            continue
        values = ea.Scalars(name)
        yield name, values


T = TypeVar("T")


def try_with_default(f: Callable[[], T], default: Callable[[], T]) -> T:
    try:
        return f()
    except Exception as e:
        print(f"Got exception {e}")
        return default()


def try_get_amlt_status(project_name: str, jobdir: str) -> Dict[str, Any]:
    """Returns the status of a job found in a job directory.

    Args:
        project_name: Amulet project name
        jobdir: name of the job directory
            (usually amlt/<experiment name>/<job name>/XX/YY).

    Returns:
        a dictionary containing the job status.
    """
    exp_name, job_name = guess_exp_name_job_name(jobdir)
    return try_with_default(lambda: get_amlt_status(project_name, exp_name, job_name), dict)


def try_get_config(jobdir: str, config_file_name: str = "config.yaml") -> Dict[str, Any]:
    """Returns the config for a job found in a job directory.

    Args:
        jobdir: name of the job directory
            (usually amlt/<experiment name>/<job name>/XX/YY).
        config_file_name: name of the config file to look for in the
            subdirectory of jobdir.

    Returns:
        a dictionary containing the config.
    """

    def get_config() -> Dict[str, Any]:
        config_file = next(find_files(jobdir, config_file_name))
        with open(config_file) as f:
            return yaml.safe_load(f)

    return try_with_default(get_config, dict)


@dataclass(frozen=True)
class ScalarEventWithRelative:
    wall_time: float
    relative_wall_time: float
    step: int
    value: float


@dataclass(frozen=True)
class NamedMetricsSequence:
    name: str
    sequence: List[ScalarEventWithRelative]


ScalarEventWithRelativeKeys = Literal["wall_time", "relative_wall_time", "step", "value"]


def metric_column_name(
    metric_name: str,
    key: ScalarEventWithRelativeKeys = "value",
    metric_direction: Optional[str] = None,
):
    parts = (
        ([metric_direction] if metric_direction is not None else [])
        + [metric_name]
        + ([] if key == "value" else [key])
    )
    return "_".join(parts)


NestedScalarEventWithRelative = Union[ScalarEventWithRelative, List]

NestedScalar = Union[float, int, List]


def flatten_scalar_event_with_relative(
    nested_values: NestedScalarEventWithRelative,
) -> Dict[ScalarEventWithRelativeKeys, NestedScalar]:
    """Turns List[List[...[ScalarEventWithRelative]]] into
    Dict(
        wall_time=List[List[...[float]]],
        relative_wall_time=List[List[...[float]]],
        step=List[List[...[int]]],
        value=List[List[...[float]]]
    )
    """

    def flatten(
        value: NestedScalarEventWithRelative,
        key: ScalarEventWithRelativeKeys,
    ) -> NestedScalar:
        if isinstance(value, ScalarEventWithRelative):
            return getattr(value, key)
        return [flatten(v, key) for v in value]

    return {key: flatten(nested_values, key) for key in get_args(ScalarEventWithRelativeKeys)}


def get_metrics_from_local_tensorboard_files(
    jobdir: str, metric_names: List[str]
) -> List[NamedMetricsSequence]:
    return [
        NamedMetricsSequence(
            name=name,
            sequence=[
                ScalarEventWithRelative(
                    wall_time=s.wall_time,
                    relative_wall_time=s.wall_time - values[0].wall_time,
                    step=s.step,
                    value=s.value,
                )
                for s in values
            ]
            if len(values) > 0
            else [],
        )
        for name, values in load_scalars(jobdir, names_to_report=metric_names)
    ]


def summarize_job(
    jobdir: str, config_file_name: str, metric_names: List[str], project_name: Optional[str] = None
):
    """Summarizes AMLT status, config, and metrics into a single dictionary.

    Args:
        jobdir: name of the job directory
            (usually amlt/<experiment name>/<job name>/XX/YY).
        config_file_name: name of the config file to look for in the
            subdirectory of jobdir.
        metric_names: list of metric names.

    Returns:
        a dictionary containing AMLT status, config and metrics.
    """
    if project_name:
        amlt_status = try_get_amlt_status(project_name, jobdir)
    else:
        amlt_status = {}
    config = try_get_config(jobdir, config_file_name)

    # Load metrics from tensorboard files in jobdir.
    metrics = get_metrics_from_local_tensorboard_files(jobdir, metric_names)

    return {
        "jobdir": jobdir,
        **amlt_status,
        **config,
        **{m.name: m.sequence for m in metrics},
    }


def process_metric_args(metrics: List[str], default_direction="min") -> Tuple[List[str], List[str]]:
    """Unzips a list of "metric_name:metric_direction" into two lists separately containing names and directions."""
    metric_names = []
    metric_directions = []
    for metric in metrics:
        metric_name, metric_direction = (
            tuple(metric.split(":")) if ":" in metric else (metric, default_direction)
        )
        assert metric_direction in ["min", "max"]
        metric_names.append(metric_name)
        metric_directions.append(metric_direction)
    return metric_names, metric_directions


def reduce_and_flatten_metrics(df: DataFrame, metrics: List[str]) -> DataFrame:
    metric_names, metric_directions = process_metric_args(metrics)
    for metric_name, metric_direction in zip(metric_names, metric_directions):
        min_or_max = min if metric_direction == "min" else max
        extremizers: List[ScalarEventWithRelative] = [
            min_or_max(
                r,
                key=lambda e: e.value,
                default=ScalarEventWithRelative(float("nan"), float("nan"), -1, float("nan")),
            )
            for r in df[metric_name]
        ]
        df = df.assign(
            **{
                metric_column_name(metric_name, key, metric_direction): value
                for key, value in flatten_scalar_event_with_relative(extremizers).items()
            }
        )

    # also flatten the raw metrics
    for metric_name in metric_names:
        df = df.assign(
            **{
                metric_column_name(metric_name, key): value
                for key, value in flatten_scalar_event_with_relative(
                    df[metric_name].tolist()
                ).items()
            }
        )

    return df


def summarize(args):
    metric_names, _ = process_metric_args(args.metrics)
    data = []
    # Checkout amulet if both amulet_project_name and amulet_local_path are specified;
    # otherwise use the current directory.
    with checkout_project(
        args.amulet_local_path, args.amulet_project_name
    ) if args.amulet_local_path and args.amulet_project_name else nullcontext():
        # Pull results from Amulet database if project_name and pull_results are specified;
        # otherwise use experiment_dirs.
        for experiment_dir in (
            pull_results(
                project_name=args.amulet_project_name,
                experiments=args.pull_results,
                limit=args.pull_results_limit,
                # assume that the standard 'amlt' output directory is used.
                output_dir=Path(args.amulet_local_path or ".") / DEFAULT_OUTPUT_DIR,
            )
            if args.amulet_project_name and args.pull_results
            else args.experiment_dirs
        ):
            num_jobs = 0
            for dirname, _, files in os.walk(experiment_dir):
                if not any([file.startswith("events.out.tfevents") for file in files]):
                    continue
                data.append(
                    summarize_job(
                        dirname,
                        args.config_file_name,
                        metric_names,
                        project_name=args.amulet_project_name,
                    )
                )
                num_jobs += 1
            print(f"Found {num_jobs} jobs in {experiment_dir}")
    df = DataFrame(data)
    df = reduce_and_flatten_metrics(df, args.metrics)
    df.to_pickle(args.output_pickle_file)

    return df


def plot(args, df):
    # Plot the first metric
    metric_name, metric_direction = map(itemgetter(0), process_metric_args(args.metrics))
    scatter_x = "relative_wall_time" if args.scatter_x == "relative" else args.scatter_x
    fig = px.scatter(
        df,
        x=metric_column_name(metric_name, scatter_x, metric_direction),
        y=metric_column_name(metric_name, "value", metric_direction),
        color=args.scatter_color,
        hover_data=args.scatter_hover_data,  # ["job_name", "sku", "learning_rate", "batch_size"]
    )
    for i, row in df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=row[metric_column_name(metric_name, scatter_x)],
                y=row[metric_name],
                mode="lines",
                name=row.job_name if "job_name" in row else str(i),
                line_color="#0000aa",
                opacity=0.1,
            )
        )

    fig.write_html(args.output_scatter_html)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize experiments into a pandas DataFrame and creates an interactive plot.",
        epilog="""
Example:

# Save the experiment directories into experiments.txt
cat > experiments.txt <<EOF
amlt/lenient-hound
amlt/desired-cattle
EOF

# Invoke experiment_summary.py
# (-- separates optional and positional arguments)
python projects/utilities/experiment_summary.py \\
    --metrics valid_loss:min \\
    --scatter-x relative \\
    --scatter-color sku \\
    --scatter-hover-data job_name sku learning_rate batch_size \\
    -- `< experiments.txt`

""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "experiment_dirs",
        type=str,
        nargs="+",
        help="Name of experiment directories (e.g., amlt/*).",
    )
    parser.add_argument(
        "--config-file-name",
        type=str,
        default="config.yaml",
        help="Name of the config file to look for in each job directory (default: config.yaml).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        help="Names (and optionally directions) of metrics to read from the tensor board (e.g., valid_loss:min"
        " accuracy:max). Note: only the first metric is plotted. To plot other metrics, Invoke"
        " experiment_summary.py again with --load-dataframe option and specify the metric to be "
        " plotted in the --metrics option.",
    )
    parser.add_argument(
        "-o",
        dest="output_pickle_file",
        type=str,
        default="experiment_summary.pck",
        help="Path to the summary file to be output.",
    )
    parser.add_argument(
        "--amulet-project-name",
        type=str,
        help="Amulet project name.",
    )
    parser.add_argument(
        "--amulet-local-path",
        type=str,
        help=(
            "Amulet local path. If this is specified, we make a temporary checkout of the project "
            "at this location. Otherwise, the current directory is assumed to be the local path."
        ),
    )
    parser.add_argument(
        "--load-dataframe",
        type=str,
        help="Path to the summary file to read from (instead of reading data from EXPERIMENT_DIRS).",
    )
    parser.add_argument(
        "--scatter-x",
        type=str,
        choices=["wall_time", "relative", "step"],
        default="step",
        help="X-axis for the scatter plot.",
    )
    parser.add_argument(
        "--scatter-color",
        type=str,
        help="Summary dataframe column name to be used for the color in the scatter plot.",
    )
    parser.add_argument(
        "--scatter-hover-data",
        type=str,
        nargs="*",
        help="Summary dataframe column names to be used for 'hover data' in the scatter plot.",
    )
    parser.add_argument(
        "--output-scatter-html",
        type=str,
        default="scatter.html",
        help="Path to the scatter plot to be output.",
    )
    args = parser.parse_args()

    if args.load_dataframe:
        df = pandas.read_pickle(args.load_dataframe)
    else:
        df = summarize(args)
    plot(args, df)


if __name__ == "__main__":
    main()
