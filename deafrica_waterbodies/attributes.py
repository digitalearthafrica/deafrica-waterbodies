import geohash as gh
import geopandas as gpd


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


def get_timeseries_s3_object_url(
    uid: str,
    product_version: str,
    timeseries_bucket: str,
) -> str:
    """
    Get the timeseries s3 object URL given a unique identifier for a polygon.

    Parameters
    ----------
    uid : str
        Unique identifier
    product_version : str
        The product version for the DE Africa Waterbodies service.
    timeseries_bucket : str
        The s3 bucket for the DE Africa Waterbodies service timeseries.

    Returns
    -------
    str
        A s3 object URL for the timeseries for a waterbody polygon.
    """

    # Incase storage location is local.
    if timeseries_bucket is None:
        timeseries_bucket == "deafrica-waterbodies-dev"

    version = product_version.replace(".", "-")

    subfolder = uid[:4]

    csv_file = f"{uid}.csv"

    timeseries_s3_object_url = f"https://{timeseries_bucket}.s3.af-south-1.amazonaws.com/{version}/timeseries/{subfolder}/{csv_file}"

    return timeseries_s3_object_url


def add_timeseries_attribute(
    polygons: gpd.GeoDataFrame, product_version: str, timeseries_bucket: str
) -> gpd.GeoDataFrame:
    """
    Function to assign the s3 object URL for the timeseries for each waterbody polygon.

    Parameters
    ----------
    polygons : gpd.GeoDataFrame
        GeoDataFrame containing the waterbody polygons.
    product_version : str
        The product version for the DE Africa Waterbodies service.
    timeseries_bucket : str
        The s3 bucket for the DE Africa Waterbodies service timeseries.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the waterbody polygons with an additional
        column "timeseries".
        The "timeseries" column contains the s3 object URL for the timeseries for each
        of the waterbody polygons.
    """

    polygons["timeseries"] = polygons.apply(
        lambda row: get_timeseries_s3_object_url(
            uid=row["UID"],
            product_version=product_version,
            timeseries_bucket=timeseries_bucket,
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
