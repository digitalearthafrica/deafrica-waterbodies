import logging
import os
from urllib.parse import urlparse

import geohash as gh
import geopandas as gpd
from shapely import Point, Polygon

from deafrica_waterbodies.io import check_if_s3_uri

_log = logging.getLogger(__name__)


def assign_unique_ids(polygons: gpd.GeoDataFrame, precision: int = 10) -> gpd.GeoDataFrame:
    """
    Function to assign a unique ID to each waterbody polygon.

    Parameters
    ----------
    polygons : gpd.GeoDataFrame
        GeoDataFrame containing the waterbody polygons.
    precision : int
        Precision to use when encoding a polygon's centroid using geohash to
        generate the polygon's unique identifier.
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
        lambda x: gh.encode(x.geometry.centroid.y, x.geometry.centroid.x, precision=precision),
        axis=1,
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
    subfolder = uid[:4]
    csv_file = f"{uid}.csv"

    # Construct the S3 Object URL
    timeseries_s3_object_url = f"https://{bucket_name}.s3.{region_code}.amazonaws.com/{object_prefix}/{subfolder}/{csv_file}"

    return timeseries_s3_object_url


def get_timeseries_fp(
    uid: str,
    timeseries_directory: str,
) -> str:
    """
    Get the timeseries file path given the unique identifier for a polygon.

    Parameters
    ----------
    uid : str
        Polygon unique identifier
    timeseries_directory : str
        The directory containing the DE Africa Waterbodies timeseries csv files.

    Returns
    -------
    str
        A file path for the timeseries for a waterbody polygon.
    """
    subfolder = uid[:4]
    csv_file = f"{uid}.csv"

    # Construct the file path
    timeseries_fp = os.path.join(timeseries_directory, subfolder, csv_file)

    return timeseries_fp


def add_timeseries_attribute(
    polygons: gpd.GeoDataFrame,
    timeseries_directory: str,
    region_code: str = "af-south-1",
) -> gpd.GeoDataFrame:
    """
    Function to assign a file path or S3 Object URL for the timeseries for each waterbody polygon.

    Parameters
    ----------
    polygons : gpd.GeoDataFrame
        GeoDataFrame containing the waterbody polygons.
    timeseries_directory : str
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
    if check_if_s3_uri(timeseries_directory):
        # Parse the S3 URI.
        parsed = urlparse(timeseries_directory, allow_fragments=False)
        bucket_name = parsed.netloc
        object_prefix = parsed.path.strip("/")

        polygons["timeseries"] = polygons.apply(
            lambda row: get_timeseries_s3_url(
                uid=row["UID"],
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
                timeseries_directory=timeseries_directory,
            ),
            axis=1,
        )

    return polygons


def get_polygon_length(poly: Polygon) -> float:
    """
    Calculate the length of a polygon.

    Parameters
    ----------
    poly : Polygon
        Polygon to get length for.

    Returns
    -------
    float
        Length of polygon i.e. longest edge of the mminimum bounding of the polygon.
    """
    # Calculate the minimum bounding box (oriented rectangle) of the polygon
    min_bbox = poly.minimum_rotated_rectangle

    # Get the coordinates of polygon vertices.
    x, y = min_bbox.exterior.coords.xy

    # Get the length of bounding box edges
    edge_length = (
        Point(x[0], y[0]).distance(Point(x[1], y[1])),
        Point(x[1], y[1]).distance(Point(x[2], y[2])),
    )

    # Get the length of polygon as the longest edge of the bounding box.
    length = max(edge_length)

    # Get width of the polygon as the shortest edge of the bounding box.
    # width = min(edge_length)

    return length


def add_polygon_properties(polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Function to add the area, perimeter and length for each waterbody polygon.

    Parameters
    ----------
    polygons : gpd.GeoDataFrame
        GeoDataFrame containing the waterbody polygons.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with the crs "EPSG:6933" containing the waterbody polygons
        with additional columns "area_m2", "perim_m" and "length_m".
        The "area_m2" column contains the area in meters squared of each
        waterbody polygon calculated in the crs "EPS:6933".
        The "perim_m" column contains the perimeter in meters of each
        waterbody polygon calculated in the crs "EPS:6933".
        The "length_m" column contains the major axis length in meters of each
        waterbody polygon calculated in the crs "EPS:6933".
    """

    # Reproject into a projected crs
    polygons_6933 = polygons.to_crs("EPSG:6933")

    # Get the major axis length of each polygon.
    polygons_6933["length_m"] = polygons_6933["geometry"].apply(get_polygon_length)

    # Perimeter
    polygons_6933["perim_m"] = polygons_6933.geometry.length
    polygons_6933["perim_m"] = polygons_6933["perim_m"].round(decimals=4)

    # Area
    polygons_6933["area_m2"] = polygons_6933.geometry.area
    polygons_6933["area_m2"] = polygons_6933["area_m2"].round(decimals=4)

    return polygons_6933
