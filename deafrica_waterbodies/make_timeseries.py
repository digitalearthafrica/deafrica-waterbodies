import datetime
import logging
import os
from pathlib import Path

import boto3
import datacube
import dateutil
import geopandas as gpd
import numpy as np
import pandas as pd
from datacube.utils.geometry import Geometry
from deafrica_tools.datahandling import wofs_fuser
from deafrica_tools.spatial import xr_rasterize
from mypy_boto3_s3.client import S3Client
from odc.stats.model import DateTimeRange
from tqdm.auto import tqdm

from deafrica_waterbodies.id_field import guess_id_field
from deafrica_waterbodies.io import check_dir_exists, check_file_exists

_log = logging.getLogger(__name__)


def get_polygon_ids_for_missing_timeseries(
    polygons_gdf: gpd.GeoDataFrame, output_directory: str | Path, s3_client: S3Client | None = None
) -> list[str]:
    """
    Get IDs for polygons whose timeseries .csv file does not exist
    in the output directory.

    Parameters
    ----------
    polygons_gdf : gpd.GeoDataFrame
        A set of water body polygons with the unique ID field set as the index.
    output_directory : str
        File URI or S3 URI of the directory containing the waterbody timeseries
        files.
    s3_client : S3Client
        A low-level client representing Amazon Simple Storage Service (S3), by default None.

    Returns
    -------
    list[str]
        List of polygon IDs whose timeseries file was not found in the output
        directory.
    """
    # Support pathlib paths.
    output_directory = str(output_directory)

    # Get the service client.
    if s3_client is None:
        s3_client = boto3.client("s3")

    polygon_ids = polygons_gdf.index.to_list()

    # Check if output_dir exists.
    if not check_dir_exists(output_directory):
        _log.error(f"Could not find directory {output_directory}!")
        raise FileNotFoundError(f"Could not find directory {output_directory}!")
    else:
        polygons_wthout_timeseries = []
        for polygon_id in polygon_ids:
            timeseries_fp = os.path.join(output_directory, polygon_id[:4], f"{polygon_id}.csv")
            # Check if file exists.
            if not check_file_exists(timeseries_fp):
                polygons_wthout_timeseries.append(polygon_id)

        return polygons_wthout_timeseries


def get_last_observation_date_from_csv(
    csv_file_path: str | Path, s3_client: S3Client | None = None
) -> pd.Timestamp:
    """
    Get the date of the last observation from a water body polygon's
    timeseries csv file.

    Parameters
    ----------
    csv_file_path : str | Path
        S3 URI or File URI of the timeseries csv file for a waterbody polygon.
    s3_client : S3Client | None
        A low-level client representing Amazon Simple Storage Service (S3), by default None.

    Returns
    -------
    pd.Timestamp
        Date of the last observation from a water body polygon's timeseries
        file.
    """
    # Get the service client.
    if s3_client is None:
        s3_client = boto3.client("s3")

    # Check if the file exists.
    if check_file_exists(csv_file_path):
        # Read file using pandas.
        # Should work for s3 files also.
        timeseries_df = pd.read_csv(csv_file_path)

        # Convert to datetime.
        timeseries_df["Observation Date"] = pd.to_datetime(timeseries_df["Observation Date"])

        # Sort in acending order
        timeseries_df.sort_values(by="Observation Date", ascending=True, inplace=True)

        last_date = timeseries_df["Observation Date"].to_list()[-1]

        return last_date
    else:
        _log.error(f"File {csv_file_path} does not exist!")
        raise FileNotFoundError(f"File {csv_file_path} does not exist!")


def generate_timeseries_from_wofs_ls(
    waterbodies_vector_file: str | Path,
    output_directory: str | Path,
    use_id: str,
    missing_only: bool = False,
    time_span: str = "all",
    temporal_range: str = None,
    subset_polygons_ids: list[str] = [],
):
    """
    Function to generate a timeseries csv file for each waterbody polygon in the
    `waterbodies_vector_file`. The timeseries csv file for a waterbody polygon
    shows the percentage of the waterbody that was wet for each `wofs_ls`
    timestep in the time range specified.

    Parameters
    ----------
    waterbodies_vector_file : str | Path
        Path to the water body polygons vector file to generate the time series
        for.
    output_directory : str | Path
        File URI or S3 URI of the directory to write the timeseries csv files to.
    use_id : str
        Name of the column/field in the waterbody polygon vector file containing
        the unique key identifier for each waterbody polygon.
    missing_only : bool, optional
        If set to True, generate the timeseries for the waterbody polygons whose
        timeseries csv file is not present in the output directory specified,
        by default False
    time_span : str, optional
        Time span to generate the timeseries for. Valid options are `"all"`,
        `"custom"`, or `"append"`,  by default "all"
    temporal_range: str | None, optional
        Time range to generate the timeseries for, if `time_span` is set to
        `"custom"`. Example '2020-05--P1M' for the month of May 2020, by default
        None
    subset_polygons_ids : list[str], optional
        A list of ids of the waterbodies to generate the timeseries for from
        the waterbodies in `waterbodies_vector_file`.
    """
    # Support pathlib paths.
    waterbodies_vector_file = str(waterbodies_vector_file)
    output_directory = str(output_directory)

    # Get the service client.
    s3_client = boto3.client("s3")

    # We will be using wofs_ls data.
    output_crs = "EPSG:6933"
    resolution = (-30, 30)
    dask_chunks = {"x": 3200, "y": 3200, "time": 1}

    # Load the waterbody polygons.
    try:
        polygons_gdf = gpd.read_file(waterbodies_vector_file)
    except Exception as error:
        _log.exception(f"Could not read file {waterbodies_vector_file}")
        raise error
    else:
        id_field = guess_id_field(polygons_gdf, use_id)
        _log.info(f"Guessed ID field: {id_field}")
        polygons_gdf.set_index(id_field, inplace=True)

    # Reproject to a projected crs.
    polygons_gdf = polygons_gdf.to_crs(output_crs)
    assert polygons_gdf.crs.is_projected

    # Select polygons using polygons using values in the id column.
    if subset_polygons_ids:
        polygons_gdf = polygons_gdf.loc[subset_polygons_ids]

    # Get the IDs for the water body polygons with no timeseries csv file in the
    # output directory.
    if missing_only:
        polygon_ids = get_polygon_ids_for_missing_timeseries(
            polygons_gdf, output_directory, s3_client=s3_client
        )
    else:
        polygon_ids = polygons_gdf.index.to_list()

    if not polygon_ids:
        _log.info("No polygons identified with missing timeseries.")
    else:
        _log.info(f"Number of polygons to generate timeseries for {len(polygons_gdf)}.")

        # Time span is mutually exclusive with start_date and end_date.
        valid_time_span_options = ["all", "custom", "append"]

        if time_span not in valid_time_span_options:
            _log.error(f"{time_span} is an invalid time span.")
            raise ValueError(
                f"Please select a valid time span option: {' '.join(valid_time_span_options)}"
            )

        # Checks.
        if time_span == "all":
            if temporal_range:
                _log.error("Time span set to `all` yet temporal range specified.")
                raise ValueError("If time span is set to 'all' do not pass a temporal range.")
            else:
                start_date_str = "1984"
                end_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        elif time_span == "append":
            # Start date will be defined in polygons_id loop.
            end_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        elif time_span == "custom":
            try:
                temporal_range_ = DateTimeRange(temporal_range)
            except ValueError:
                _log.exception(f"Failed to parse supplied temporal_range: '{temporal_range}'")
            else:
                start_date_str = temporal_range_.start.strftime("%Y-%m-%d")
                end_date_str = temporal_range_.end.strftime("%Y-%m-%d")

        # For logging purposes only.
        if time_span != "append":
            _log.info(
                f"Generating timeseries for the time range: {start_date_str} to {end_date_str}."
            )

        # Connect to the datacube
        dc = datacube.Datacube(app="deafricawaterbodies-timeseries")

        with tqdm(total=len(polygon_ids)) as bar:
            for poly_id in polygon_ids:
                # Polygon's timeseries file path.
                poly_timeseries_fp = os.path.join(output_directory, poly_id[:4], f"{poly_id}.csv")

                if time_span == "append":
                    try:
                        last_observation_date = get_last_observation_date_from_csv(
                            poly_timeseries_fp, s3_client
                        )
                    except FileNotFoundError:
                        start_date_str = "1984"
                        _log.info(
                            f"Could not find last observation date for polygon {poly_id}, defaulting to using the start date {start_date_str}."
                        )
                    else:
                        start_date = last_observation_date + dateutil.relativedelta.relativedelta(
                            days=1
                        )
                        start_date_str = start_date.strftime("%Y-%m-%d")

                time_range = (start_date_str, end_date_str)
                _log.info(
                    f"Generating timeseries for {poly_id} for the time range: {time_range[0]} to {time_range[1]}."
                )

                poly_geom = polygons_gdf.loc[poly_id].geometry
                poly_gdf = gpd.GeoDataFrame(geometry=[poly_geom], crs=output_crs)
                poly_geopolygon = Geometry(geom=poly_geom, crs=output_crs)

                # Load the Water Observations from Space.
                wofls_ds = dc.load(
                    product="wofs_ls",
                    geopolygon=poly_geopolygon,
                    time=time_range,
                    resolution=resolution,
                    output_crs=output_crs,
                    dask_chunks=dask_chunks,
                    resampling="nearest",
                    group_by="solar_day",
                    fuse_func=wofs_fuser,
                )

                # If no data is found.
                if not wofls_ds:
                    _log.info(
                        f"There is data for {poly_id} for the time range: {time_range[0]} to {time_range[1]}."
                    )
                    continue
                else:
                    wofls_da = wofls_ds.water
                    # Mask the loaded WOfS data using the rasterized waterbody polygon,
                    # if the height and width of the bounding box of the waterbody polygon
                    # are large than the length of a pixel.
                    pixel_length = abs(resolution[0])  # should be a positive number.
                    if (
                        poly_geopolygon.boundingbox.height > pixel_length
                        and poly_geopolygon.boundingbox.width > pixel_length
                    ):
                        poly_mask = xr_rasterize(poly_gdf, wofls_da)
                        wofls_da_masked = wofls_da.where(poly_mask, np.nan)
                    else:
                        _log.info(
                            f"Water body polygon bounding box length and width are smaller than pixel length {pixel_length} metres."
                        )
                        wofls_da_masked = wofls_da

                    # Compute the array at this point.
                    wofls_da_masked = wofls_da_masked.compute()

                    # Get the area of each pixel.
                    pixel_area = pixel_length**2

                    # Get the number of pixels for the waterbody.
                    pixel_count = (~np.isnan(wofls_da_masked)).sum(["x", "y"])

                    # Apply WOfS bitmasking to the Water Observation Feature Layers
                    # See: the  Applying WOfS Bitmasking notebook in the
                    # Frequently_used_code folder in the
                    # digitalearthafrica/deafrica-sandbox-notebooks Github repository.

                    # Number of pixels observed to be valid (clear) and dry.
                    valid_and_dry_count = (wofls_da_masked == 0).sum(["x", "y"])
                    # Percentage of pixels observed to be valid (clear) and dry.
                    valid_and_dry_percentage = (valid_and_dry_count / pixel_count) * 100.0
                    # Area covered by valid (clear) and dry pixels.
                    valid_and_dry_area = valid_and_dry_count * pixel_area

                    # Number of pixels observed to be valid (clear) and wet.
                    valid_and_wet_count = (wofls_da_masked == 128).sum(["x", "y"])
                    # Percentage of pixels observed to be valid (clear) and wet.
                    valid_and_wet_percentage = (valid_and_wet_count / pixel_count) * 100.0
                    # Area covered by valid (clear) and wet pixels.
                    valid_and_wet_area = valid_and_wet_count * pixel_area

                    # Number of valid (clear) pixels.
                    valid_count = valid_and_dry_count + valid_and_wet_count

                    # Number of invalid (not clear) pixels.
                    invalid_count = pixel_count - valid_count
                    # Area covered by invalid (not clear) pixels.
                    invalid_area = invalid_count * pixel_area
                    # Percentage of invalid pixels.
                    invalid_percentage = (invalid_count / pixel_count) * 100.0

                    # Create dataframes from the xarray.DataArrays.
                    valid_and_wet_percentage_df = valid_and_wet_percentage.to_dataframe(
                        name="pc_wet"
                    ).drop(columns="spatial_ref", errors="ignore")
                    valid_and_wet_count_df = valid_and_wet_count.to_dataframe(name="px_wet").drop(
                        columns="spatial_ref", errors="ignore"
                    )
                    valid_and_wet_area_df = valid_and_wet_area.to_dataframe(
                        name="area_wet_m2"
                    ).drop(columns="spatial_ref", errors="ignore")
                    valid_and_dry_percentage_df = valid_and_dry_percentage.to_dataframe(
                        name="pc_dry"
                    ).drop(columns="spatial_ref", errors="ignore")
                    valid_and_dry_count_df = valid_and_dry_count.to_dataframe(name="px_dry").drop(
                        columns="spatial_ref", errors="ignore"
                    )
                    valid_and_dry_area_df = valid_and_dry_area.to_dataframe(
                        name="area_dry_m2"
                    ).drop(columns="spatial_ref", errors="ignore")
                    invalid_percentage_df = invalid_percentage.to_dataframe(name="pc_invalid").drop(
                        columns="spatial_ref", errors="ignore"
                    )
                    invalid_count_df = invalid_count.to_dataframe(name="px_invalid").drop(
                        columns="spatial_ref", errors="ignore"
                    )
                    invalid_area_df = invalid_area.to_dataframe(name="area_invalid_m2").drop(
                        columns="spatial_ref", errors="ignore"
                    )

                    # Merge the individual dataframes into a single dataframe.
                    timeseries_df = pd.concat(
                        [
                            valid_and_wet_percentage_df,
                            valid_and_wet_count_df,
                            valid_and_wet_area_df,
                            valid_and_dry_percentage_df,
                            valid_and_dry_count_df,
                            valid_and_dry_area_df,
                            invalid_percentage_df,
                            invalid_count_df,
                            invalid_area_df,
                        ],
                        ignore_index=False,
                        join="outer",
                        axis="columns",
                    )

                    # Set pc_wet and pc_dry values to nan, which will be used if insufficient pixels are observed.
                    timeseries_df["pc_wet"] = timeseries_df.apply(
                        lambda row: np.nan if row.pc_invalid > 10.0 else row.pc_wet, axis=1
                    )
                    timeseries_df["pc_dry"] = timeseries_df.apply(
                        lambda row: np.nan if row.pc_invalid > 10.0 else row.pc_dry, axis=1
                    )

                    # Parse the datetime index.
                    timeseries_df.index = timeseries_df.index.strftime("%Y-%m-%d")

                    # Sort by date.
                    timeseries_df.sort_index(ascending=True, inplace=True)

                    if time_span == "append":
                        # Append the DataFrame to an existing csv file.
                        timeseries_df.to_csv(
                            poly_timeseries_fp, mode="a", index=False, header=False
                        )
                    else:
                        # Write the DataFrame to a new csv file.
                        timeseries_df.to_csv(poly_timeseries_fp, mode="w", index=False)
                bar.update(1)
            _log.info(f"Done! Generated timeseries for {len(polygon_ids)}.")
