import math
import os
from pathlib import Path

import fsspec
import geopandas as gpd
import pytest
from click.testing import CliRunner

from deafrica_waterbodies.cli.generate_polygons import generate_polygons

# Test directory.
HERE = Path(__file__).parent.resolve()
TEST_WATERBODY = os.path.join(HERE, "data", "sm9rtw98n.parquet")
TEST_OUTPUT_DIRECTORY = HERE / "data"


@pytest.fixture
def runner():
    return CliRunner(echo_stdin=True)


def test_generate_polygons(runner):
    aoi_vector_file = TEST_WATERBODY
    tile_size_factor = 1
    num_workers = 8
    detection_threshold = 0.1
    extent_threshold = 0.05
    min_valid_observations = 60
    raster_processing_plugin_name = "raster_processing_filtering"
    output_directory = TEST_OUTPUT_DIRECTORY
    min_polygon_size = 4500
    max_polygon_size = math.inf
    timeseries_directory = os.path.join(TEST_OUTPUT_DIRECTORY, "timeseries")
    file_name_prefix = "waterbodies"

    args = [
        "--verbose",
        f"--aoi-vector-file={aoi_vector_file}",
        f"--tile-size-factor={tile_size_factor}",
        f"--num-workers={num_workers}",
        f"--detection-threshold={detection_threshold}",
        f"--extent-threshold={extent_threshold}",
        f"--min-valid-observations={min_valid_observations}",
        f"--raster-processing-plugin-name={raster_processing_plugin_name}",
        f"--min-polygon-size={min_polygon_size}",
        f"--max-polygon-size={max_polygon_size}",
        f"--output-directory={output_directory}",
        "--overwrite",
        f"--timeseries-directory={timeseries_directory}",
        f"--file-name-prefix={file_name_prefix}",
    ]

    result = runner.invoke(generate_polygons, args=args, catch_exceptions=True)

    assert result.exit_code == 0

    waterbodies = gpd.read_file(os.path.join(output_directory, "waterbodies.shp"))

    assert len(waterbodies) == 2

    # File clean up.
    fs = fsspec.filesystem("file")
    fs.rm(output_directory, recursive=True)
