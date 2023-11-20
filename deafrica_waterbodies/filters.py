"""
Filter waterbody polygons based on different criteria.
"""
import logging
import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

_log = logging.getLogger(__name__)


def filter_by_intersection(
    gpd_data: gpd.GeoDataFrame,
    gpd_filter: gpd.GeoDataFrame,
    filtertype: str = "intersects",
    invert_mask: bool = True,
    return_inverse: bool = False,
) -> gpd.GeoDataFrame:
    """
    Filter out polygons from `gpd_data` that intersect with polygons in `gpd_filter`.

    Parameters
    ----------
    gpd_data : gpd.GeoDataFrame
        Polygons to be filtered.
    gpd_filter : gpd.GeoDataFrame
        Polygons to be used as a filter.
    filtertype : str, optional
        Options = ["intersects", "contains", "within"], by default "intersects"
    invert_mask : bool, optional
        This determines whether you want polygons that DO ( = False) or
        DON'T ( = True) intersect with the filter dataset, by default True
    return_inverse : bool, optional
        If = True, then return both parts of the intersection:
        - those that intersect AND
        - those that don't as two GeoDataFrames, by default False

    Returns
    -------
    gpd_data_filtered: gpd.GeoDataFrame
        If invert_mask==True, `gpd_data_filtered` is a filtered polygon set,
        with polygons that DO intersect with `gpd_filter` removed.
        If invert_mask==False, `gpd_data_filtered` is a filtered polygon set,
        with polygons that DON'T intersect with `gpd_filter` removed.

    Optional
    --------
    if 'return_inverse = True'
    gpd_data_inverse: geopandas GeoDataFrame
        If invert_mask==True, `gpd_data_inverse` is a filtered polygon set,
        with polygons that DON'T intersect with `gpd_filter` removed (inverse of gpd_data_filtered).
        If invert_mask==False, `gpd_data_inverse` is a filtered polygon set,
        with polygons that DO intersect with `gpd_filter` removed (inverse of gpd_data_filtered).

    """
    # Check that the coordinate reference systems of both GeoDataFrames are the same.
    assert gpd_data.crs == gpd_filter.crs

    # Find the index of all the polygons in gpd_data that intersect with gpd_filter.
    intersections = gpd_filter.sjoin(gpd_data, how="inner", predicate=filtertype)
    intersect_index = np.sort(intersections["index_right"].unique())

    if invert_mask:
        # Grab only the polygons that ARE NOT in the intersect_index.
        gpd_data_filtered = gpd_data.loc[~gpd_data.index.isin(intersect_index)]
    else:
        # Grab only the polygons that ARE in the intersect_index.
        gpd_data_filtered = gpd_data.loc[gpd_data.index.isin(intersect_index)]

    if return_inverse:
        # We need to use the indices from intersect_index to find the inverse dataset, so we
        # will just swap the '~'.

        if invert_mask:
            # Grab only the polygons that ARE in the intersect_index.
            gpd_data_inverse = gpd_data.loc[gpd_data.index.isin(intersect_index)]
        else:
            # Grab only the polygons that are NOT in the intersect_index.
            gpd_data_inverse = gpd_data.loc[~gpd_data.index.isin(intersect_index)]

        return gpd_data_filtered, gpd_data_inverse
    else:
        return gpd_data_filtered


def filter_by_area(
    polygons_gdf: gpd.GeoDataFrame,
    min_polygon_size: float = 4500,
    max_polygon_size: float = math.inf,
) -> gpd.GeoDataFrame:
    """
    Filter a set of water body polygons using the minimum and
    maximum area.

    Parameters
    ----------
    polygons_gdf : gpd.GeoDataFrame
    min_polygon_size : float, optional
        Minimum area of a waterbody polygon to be included in the output polygons, by default 4500
    max_polygon_size : float, optional
        Maximum area of a waterbody polygon to be included in the output polygons, by default math.inf

    Returns
    -------
    gpd.GeoDataFrame:
        The area filtered water body polygons.
    """
    crs = polygons_gdf.crs
    assert crs.is_projected

    _log.info(
        f"Filtering {len(polygons_gdf)} polygons by minimum area {min_polygon_size} and max area {max_polygon_size}..."
    )

    polygons_gdf["area_m2"] = pd.to_numeric(polygons_gdf.area)
    area_filtered_polygons_gdf = polygons_gdf.loc[
        (
            (polygons_gdf["area_m2"] > min_polygon_size)
            & (polygons_gdf["area_m2"] <= max_polygon_size)
        )
    ]
    area_filtered_polygons_gdf = gpd.GeoDataFrame(data=area_filtered_polygons_gdf)

    _log.info(f"Filtered out {len(polygons_gdf) - len(area_filtered_polygons_gdf)} polygons.")

    return area_filtered_polygons_gdf


def pp_test_gdf(input_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Function to calculate the Polsby–Popper test values on a
    geopandas GeoDataFrame.

    Parameters
    ----------
    input_gdf : gpd.GeoDataFrame
        Polygons to calculate the Polsby–Popper test values for.

    Returns
    -------
    gpd.GeoDataFrame
        Polygons GeoDataFrame with a column `pp_test` containing the Polsby–Popper test values
        for each polygon.
    """
    crs = input_gdf.crs
    assert crs.is_projected

    input_gdf["area"] = input_gdf["geometry"].area
    input_gdf["perimeter"] = input_gdf["geometry"].length
    input_gdf["pp_test"] = (4 * math.pi * input_gdf["area"]) / (input_gdf["perimeter"] ** 2)

    return input_gdf


# From https://stackoverflow.com/a/70387141
def remove_polygon_interiors(poly: Polygon) -> Polygon:
    """
    Close polygon holes by limitation to the exterior ring.

    Parameters
    ----------
    poly : Polygon
        Input Polygon.

    Returns
    -------
    Polygon
        Input Polygon without any interior holes.
    """
    if len(poly.interiors) > 0:
        return Polygon(list(poly.exterior.coords))
    else:
        return poly


def get_largest_polygon(poly_list: list) -> Polygon:
    """
    Get the largest polygon from a list of polygons.

    Parameters
    ----------
    poly_list : list
        List of polygons to filter.

    Returns
    -------
    Polygon
        The largest polygon by area from the list of polygons.
    """
    # Get the area of each polygon in the list.
    poly_areas = [poly.area for poly in poly_list]
    # Get the largest area.
    max_area = max(poly_areas)
    # Get the index for the largest area.
    max_area_idx = poly_areas.index(max_area)
    # Use the index to get the largest polygon.
    largest_polygon = poly_list[max_area_idx]
    return largest_polygon


def fill_holes(geom: Polygon | MultiPolygon) -> Polygon | MultiPolygon:
    """
    Fill holes in polygon.

    Parameters
    ----------
    geom : Polygon | MultiPolygon
        Polygon or MultiPolygon to fill holes in.

    Returns
    -------
    Polygon | MultiPolygon
        Polygon or MultiPolygon with no holes.
    """
    if isinstance(geom, MultiPolygon):
        # For each polygon in the MultiPolygon,
        # close the polygon holes.
        closed_polygons = [remove_polygon_interiors(g) for g in geom.geoms]
        # Get the largest polygon.
        largest_polygon = get_largest_polygon(closed_polygons)
        # Get the polygons not within the largest polygon.
        outside_largest_polygon = [
            poly for poly in closed_polygons if not poly.within(largest_polygon)
        ]

        if outside_largest_polygon:
            return MultiPolygon([largest_polygon, *outside_largest_polygon])
        else:
            return largest_polygon
    elif isinstance(geom, Polygon):
        return remove_polygon_interiors(geom)


def remove_polygons_within_polygons(polygons_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Remove polygons within other polygons.

    Parameters
    ----------
    polygons_gdf : gpd.GeoDataFrame
        Set of polygons to filter.

    Returns
    -------
    gpd.GeoDataFrame
        Input polygons with polygons contained in other polygons removed.
    """
    _log.info(f"Initial polygon count {len(polygons_gdf)}")

    polygons_to_delete = []
    for row in polygons_gdf.itertuples():
        row_id = row.Index
        row_geom = row.geometry

        polygons_to_check_against = polygons_gdf.loc[polygons_gdf.index != row_id]

        # Check if the row geometry is within any of the other polygons.
        if polygons_to_check_against.geometry.contains(row_geom).any():
            polygons_to_delete.append(row_id)

    if polygons_to_delete:
        polygons_to_delete_gdf = polygons_gdf.loc[polygons_gdf.index.isin(polygons_to_delete)]
        _log.info(f"Found {len(polygons_to_delete_gdf)} polygons within polygons.")

        polygons_within_polygons_removed = polygons_gdf.loc[
            ~polygons_gdf.index.isin(polygons_to_delete)
        ]
        _log.info(
            f"Polygon count after removing polygons within polygons {len(polygons_within_polygons_removed)}."
        )

        return polygons_within_polygons_removed

    else:
        _log.info("Found no polygons within polygons.")
        return polygons_gdf
