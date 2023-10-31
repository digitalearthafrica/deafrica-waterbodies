import logging
import os
from urllib.parse import urlparse

import geohash as gh
import geopandas as gpd

from deafrica_waterbodies.io import check_if_s3_uri

_log = logging.getLogger(__name__)


def assign_unique_ids(polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Function to assign a unique ID to each waterbody polygon.

    Parameters
    ----------
    polygons : gpd.GeoDataFrame
        GeoDataFrame containing the waterbody polygons.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the waterbody polygons with additional columns
        "UID" and "WB_ID".
        The "UID" column contains a unique identifier
        for each polygon.
        The "WB_ID" column contains an arbitrary numerical ID for each
        polygon with polygons close to each other numbered similarly.
    """

    crs = polygons.crs

    # Generate a unique id for each polygon.
    polygons_with_unique_ids = polygons.to_crs(epsg=4326)
    polygons_with_unique_ids["UID"] = polygons_with_unique_ids.apply(
        lambda x: gh.encode(x.geometry.centroid.y, x.geometry.centroid.x, precision=9), axis=1
    )

    # Check that our unique ID is in fact unique
    assert polygons_with_unique_ids["UID"].is_unique

    # Make an arbitrary numerical ID for each polygon. We will first sort the dataframe by geohash
    # so that polygons close to each other are numbered similarly.
    polygons_with_unique_ids_sorted = polygons_with_unique_ids.sort_values(by=["UID"]).reset_index()
    polygons_with_unique_ids_sorted["WB_ID"] = polygons_with_unique_ids_sorted.index

    # The step above creates an 'index' column, which we don't actually want, so drop it.
    polygons_with_unique_ids_sorted.drop(columns=["index"], inplace=True)

    # Reproject to the same crs as the input polygons.
    polygons_with_unique_ids_sorted = polygons_with_unique_ids_sorted.to_crs(crs)

    return polygons_with_unique_ids_sorted


def get_timeseries_s3_url(
    uid: str,
    timeseries_product_version: str,
    bucket_name: str,
    region_code: str,
    object_prefix: str,
) -> str:
    """
    Get the timeseries S3 Object URL given the unique identifier for a polygon.

    Parameters
    ----------
    uid : str
        Unique identifier
    timeseries_product_version : str
        The product version for the DE Africa Waterbodies timeseries.
    bucket_name : str
        The S3 bucket containing the timeseries csv files.
    region_code:
        The location of the S3 bucket specified by `bucket_name`.
    object_prefix:
        The folder on S3 containing the timeseries csv files.

    Returns
    -------
    str
        A S3 Object URL for the timeseries for a waterbody polygon.
    """

    version = timeseries_product_version.replace(".", "_")
    subfolder = uid[:4]
    csv_file = f"{uid}_v{version}.csv"

    # Construct the S3 Object URL
    timeseries_s3_object_url = f"https://{bucket_name}.s3.{region_code}.amazonaws.com/{object_prefix}/{subfolder}/{csv_file}"

    return timeseries_s3_object_url


def get_timeseries_fp(
    uid: str,
    timeseries_product_version: str,
    timeseries_dir: str,
) -> str:
    """
    Get the timeseries file path given the unique identifier for a polygon.

    Parameters
    ----------
    uid : str
        Polygon unique identifier
    timeseries_product_version : str
        The product version for the DE Africa Waterbodies timeseries.
    timeseries_dir : str
        The directory containing the DE Africa Waterbodies timeseries csv files.

    Returns
    -------
    str
        A file path for the timeseries for a waterbody polygon.
    """

    version = timeseries_product_version.replace(".", "_")
    subfolder = uid[:4]
    csv_file = f"{uid}_v{version}.csv"

    # Construct the file path
    timeseries_fp = os.path.join(timeseries_dir, {subfolder}, {csv_file})

    return timeseries_fp


def add_timeseries_attribute(
    polygons: gpd.GeoDataFrame,
    timeseries_product_version: str,
    timeseries_dir: str,
    region_code: str = "af-south-1",
) -> gpd.GeoDataFrame:
    """
    Function to assign a file path or S3 Object URL for the timeseries for each waterbody polygon.

    Parameters
    ----------
    polygons : gpd.GeoDataFrame
        GeoDataFrame containing the waterbody polygons.
    timeseries_product_version : str
        The product version for the DE Africa Waterbodies timeseries.
    timeseries_dir : str
        The directory containing the DE Africa Waterbodies timeseries csv files.
    region_code: str
        This is the location of the bucket if `timeseries_dir` is a S3 URI.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the waterbody polygons with an additional
        column "timeseries".
        The "timeseries" column contains the file path or S3 Object URL for the timeseries for each
        of the waterbody polygons.
    """
    if check_if_s3_uri(timeseries_dir):
        # Parse the S3 URI.
        parsed = urlparse(timeseries_dir, allow_fragments=False)
        bucket_name = parsed.netloc
        object_prefix = parsed.path.strip("/")

        polygons["timeseries"] = polygons.apply(
            lambda row: get_timeseries_s3_url(
                uid=row["UID"],
                timeseries_product_version=timeseries_product_version,
                bucket_name=bucket_name,
                region_code=region_code,
                object_prefix=object_prefix,
            ),
            axis=1,
        )
    else:
        polygons["timeseries"] = polygons.apply(
            lambda row: get_timeseries_fp(
                uid=row["UID"],
                timeseries_product_version=timeseries_product_version,
                timeseries_dir=timeseries_dir,
            ),
            axis=1,
        )

    return polygons


def add_area_and_perimeter_attributes(polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Function to add the area and perimeter for each waterbody polygon.

    Parameters
    ----------
    polygons : gpd.GeoDataFrame
        GeoDataFrame containing the waterbody polygons.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with the crs "EPSG:6933" containing the waterbody polygons
        with additional columns "area_m2" and "perim_m".
        The "area_m2" column contains the area in meters squared of each
        waterbody polygon calculated in the crs "EPS:6933".
        The "perim_m" column contains the perimeter in meters of each
        waterbody polygon calculated in the crs "EPS:6933".
    """

    # Reproject into a projected crs
    polygons_6933 = polygons.to_crs("EPSG:6933")

    # Perimeter
    polygons_6933["perim_m"] = polygons_6933.geometry.length
    polygons_6933["perim_m"] = polygons_6933["perim_m"].round(decimals=4)

    # Area
    polygons_6933["area_m2"] = polygons_6933.geometry.area
    polygons_6933["area_m2"] = polygons_6933["area_m2"].round(decimals=4)

    return polygons_6933
