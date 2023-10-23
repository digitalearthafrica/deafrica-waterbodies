import collections
import datetime
import logging
import os
import urllib
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
    start_date: datetime.datetime | None = None,
    end_date: datetime.datetime | None = None,
    subset_polygons_ids: list[str] = [],
    include_uncertainity: bool = True,
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
    start_date : datetime.datetime | None, optional
        Start date for the time range to generate the timeseries for, if `time_span`
        is set to `"custom"`, by default None
    end_date : datetime.datetime | None, optional
        End date for the time range to generate the timeseries for, if `time_span`
        is set to `"custom"`, by default None
    subset_polygons_ids : list[str], optional
        A list of ids of the waterbodies to generate the timeseries for from
        the waterbodies in `waterbodies_vector_file`.
    include_uncertainity: bool, optional
        Option to include uncertainities in the output timeseries. If you
        specify `include_uncertainity=True` then you will only filter out
        timesteps with 100% invalid pixels. If `include_uncertainity=False`
        you will filter out timesteps with more than 10% invalid pixels.
    """
    # Get the service client.
    s3_client = boto3.client("s3")

    # We will be using wofs_ls data.
    output_crs = "EPSG:6933"
    resolution = (-30, 30)
    # dask_chunks = {"x": 3200, "y": 3200, "time": 1} # TODO: Check if using dask speeds up runtime.

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

    polygons_gdf = polygons_gdf.to_crs(output_crs)

    if subset_polygons_ids:
        polygons_gdf = polygons_gdf.loc[subset_polygons_ids]

    # Get the IDs for the waterbody polygons.
    if missing_only:
        polygon_ids = get_polygon_ids_for_missing_timeseries(
            polygons_gdf, output_directory, s3_client=s3_client
        )
    else:
        polygon_ids = polygons_gdf.index.to_list()

    # Time span is mutually exclusive with start_date and end_date.
    valid_time_span_options = ["all", "custom", "append"]

    if time_span not in valid_time_span_options:
        _log.error(f"{time_span} is an invalid time span.")
        raise ValueError(
            f"Please select a valid time span option: {' '.join(valid_time_span_options)}"
        )

    # Checks.
    if time_span == "all":
        if start_date or end_date:
            _log.error("Time span set to all yet, start and end date specified.")
            raise ValueError(
                "If a time span is set to 'all' do not pass a start date nor an end date."
            )
        else:
            start_date_str = "1984"
            end_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    elif time_span == "append":
        # Start date will be defined in polygons_id loop.
        end_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    elif time_span == "custom":
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

    # For logging purposes only.
    if time_span != "append":
        _log.info(f"Generating timeseries for the time range: {start_date_str} to {end_date_str}.")

    if include_uncertainity:
        # Only filter out timesteps with 100% invalid pixels.
        invalid_percent_threshold = 100
    else:
        # Filter out timesteps with less than 90% valid pixels.
        invalid_percent_threshold = 10

    # Connect to the datacube
    dc = datacube.Datacube(app="deafricawaterbodies-timeseries")

    with tqdm(total=len(polygon_ids)) as bar:
        for poly_id in polygon_ids:
            # Polygon's timeseries file path.

            # This is specific for DE Africa waterbodies which are expected
            # to have a url pointing to the expected timeseries file for the
            # polygon.
            try:
                timeseries_url = polygons_gdf.loc[poly_id].timeseries
                path = urllib.parse.urlparse(timeseries_url).path
                csv_file = os.path.split(path)[-1]
            except AttributeError:
                csv_file = f"{poly_id}.csv"

            poly_timeseries_fp = os.path.join(output_directory, poly_id[:4], csv_file)

            if time_span == "append":
                last_observation_date = get_last_observation_date_from_csv(
                    poly_timeseries_fp, s3_client
                )
                start_date = last_observation_date + dateutil.relativedelta.relativedelta(days=1)
                start_date_str = start_date.strftime("%Y-%m-%d")

            time_range = (start_date_str, end_date_str)
            _log.debug(
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
                resampling="nearest",
                group_by="solar_day",
                fuse_func=wofs_fuser,
            )
            wofls_da = wofls_ds.water

            # If no data is found.
            if not wofls_ds:
                _log.info(
                    f"There is no new data for {poly_id} for the time range: {time_range[0]} to {time_range[1]}."
                )
                continue
            else:
                # Mask the loaded WOfS data using the rasterized waterbody polygon,
                # if the height and width of the bounding box of the waterbody polygon
                # are large than the length of a pixel.
                pixel_length = resolution[1]  # should be a positive number.
                if (
                    poly_geopolygon.boundingbox.height > pixel_length
                    and poly_geopolygon.boundingbox.width > pixel_length
                ):
                    poly_mask = xr_rasterize(poly_gdf, wofls_da)
                    wofls_da_masked = wofls_da.where(poly_mask, np.nan)
                else:
                    wofls_da_masked = wofls_da

                # Get the area of water in the waterbody for each timestep.

                timesteps = list(wofls_da_masked.time.values)

                poly_timeseries_data_dict = collections.defaultdict(list)
                for timestep in timesteps:
                    wofl = wofls_da_masked.sel(time=timestep)

                    # Number of pixels in the timestep for the water body.
                    pixel_count = np.count_nonzero(np.isnan(wofl))

                    # Apply WOfS bitmasking to the Water Observation Feature Layers
                    # See: the  Applying WOfS Bitmasking notebook in the
                    # Frequently_used_code folder in the
                    # digitalearthafrica/deafrica-sandbox-notebooks Github repository.

                    # Number of pixels observed to be valid (clear) and dry.
                    valid_and_dry_count = np.count_nonzero(wofl == 0)

                    # Number of pixels observed to be valid (clear) and wet.
                    valid_and_wet_count = np.count_nonzero(wofl == 128)

                    # Number of valid (clear) pixels.
                    valid_count = valid_and_dry_count + valid_and_wet_count

                    # Number of invalid (not clear) pixels.
                    invalid_count = pixel_count - valid_count

                    # Convert the counts into percentages.
                    try:
                        valid_and_wet_percentage = (valid_and_wet_count / pixel_count) * 100
                    except ZeroDivisionError:
                        valid_and_wet_percentage = 0
                    try:
                        valid_and_dry_percentage = (valid_and_dry_count / pixel_count) * 100
                    except ZeroDivisionError:
                        valid_and_dry_percentage = 0
                    try:
                        invalid_percentage = (invalid_count / pixel_count) * 100
                    except ZeroDivisionError:
                        invalid_percentage = 0

                    # Filter the timesteps based on the invalid pixel percentage
                    # threshold.
                    # If above threshold, set timeseries values for the timestep
                    # as empty string.
                    if invalid_percentage >= invalid_percent_threshold:
                        valid_and_wet_percentage = ""
                        valid_and_wet_count = ""
                        valid_and_dry_percentage = ""
                        valid_and_dry_count = ""
                        invalid_percentage = ""
                        invalid_count = ""

                    # Convert the timestep date from numpy.datetime64 to string.
                    observation_date = pd.to_datetime(timestep)
                    observation_date_str = observation_date.strftime("%Y-%m-%d")

                    poly_timeseries_data_dict["Observation Date"].extend([observation_date_str])
                    # poly_timeseries_data_dict["Total pixel count"].extend([pixel_count])
                    poly_timeseries_data_dict["Wet pixel percentage"].extend(
                        [valid_and_wet_percentage]
                    )
                    poly_timeseries_data_dict["Wet pixel count"].extend([valid_and_wet_count])
                    poly_timeseries_data_dict["Dry pixel percentage"].extend(
                        [valid_and_dry_percentage]
                    )
                    poly_timeseries_data_dict["Dry pixel count"].extend([valid_and_dry_count])
                    poly_timeseries_data_dict["Invalid pixel percentage"].extend(
                        [invalid_percentage]
                    )
                    poly_timeseries_data_dict["Invalid pixel count"].extend([invalid_count])

                # Convert the timeseries data dictionary for the polygon into
                # a DataFrame.
                poly_timeseries_df = pd.DataFrame(poly_timeseries_data_dict)

                if time_span == "append":
                    # Append the DataFrame to an existing csv file.
                    poly_timeseries_df.to_csv(
                        poly_timeseries_fp, mode="a", index=False, header=False
                    )
                else:
                    # Write the DataFrame to a new csv file.
                    poly_timeseries_df.to_csv(poly_timeseries_fp, mode="w", index=False)
            bar.update(1)
        _log.info(f"Done! Generated timeseries for {len(polygon_ids)}.")
