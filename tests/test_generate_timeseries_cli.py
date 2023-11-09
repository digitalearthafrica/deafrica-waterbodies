import os
from pathlib import Path

import fsspec
import pandas as pd
import pytest
from click.testing import CliRunner

from deafrica_waterbodies.cli.generate_timeseries import generate_timeseries

# Test directory.
HERE = Path(__file__).parent.resolve()
TEST_WATERBODY = os.path.join(HERE, "data", "sm9rtw98n.parquet")
TEST_OUTPUT_DIRECTORY = HERE / "test_outputs"


@pytest.fixture
def runner():
    return CliRunner(echo_stdin=True)


def test_generate_timeseries(runner):
    waterbodies_vector_file = TEST_WATERBODY
    use_id = "UID"
    output_directory = TEST_OUTPUT_DIRECTORY
    time_span = "custom"
    temporal_range = "2023-01--P1M"

    args = [
        "--verbose",
        f"--waterbodies-vector-file={waterbodies_vector_file}",
        f"--use-id={use_id}",
        f"--output-directory={output_directory}",
        f"--time-span={time_span}",
        f"--temporal-range={temporal_range}",
        "--not-missing-only",
    ]

    result = runner.invoke(generate_timeseries, args=args, catch_exceptions=True)

    assert result.exit_code == 0

    test_timeseries = pd.read_csv(os.path.join(output_directory, "sm9r/sm9rtw98n.csv"))

    assert len(test_timeseries) == 9
    assert test_timeseries.iloc[3]["pc_wet"] == 49.66442953020135

    # File clean up.
    fs = fsspec.filesystem("file")
    fs.rm(output_directory, recursive=True)
