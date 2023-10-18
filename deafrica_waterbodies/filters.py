"""
Filter waterbody polygons based on different criteria.
"""
import logging
import math
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
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
    polygons_gdf: gpd.GeoDataFrame | None,
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


def filter_using_land_sea_mask(
    primary_threshold_polygons: gpd.GeoDataFrame,
    secondary_threshold_polygons: gpd.GeoDataFrame,
    land_sea_mask_fp: str | Path = "",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Filter the primary and secondary threshold waterbody polygons using a land/sea
    mask to filter out ocean polygons.

    Parameters
    ----------
    primary_threshold_polygons : gpd.GeoDataFrame
    secondary_threshold_polygons : gpd.GeoDataFrame
    land_sea_mask_fp : str | Path, optional
        Vector file path to the polygons to use to filter out ocean waterbody polygons, by default ""

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        The filtered primary threshold polygons and the filtered
        secondary threshold polygons with ocean polygons removed.
    """
    assert primary_threshold_polygons.crs == secondary_threshold_polygons.crs
    crs = primary_threshold_polygons.crs

    # Support pathlib Paths
    land_sea_mask_fp = str(land_sea_mask_fp)

    if land_sea_mask_fp:
        _log.info(
            "Filtering out ocean polygons from the primary and secondary threshold waterbody polygons..."
        )
        try:
            land_sea_mask = gpd.read_file(land_sea_mask_fp).to_crs(crs)
        except Exception as error:
            _log.exception(f"Could not read file {land_sea_mask_fp}")
            raise error
        else:
            inland_primary_threshold_polygons = filter_by_intersection(
                gpd_data=primary_threshold_polygons,
                gpd_filter=land_sea_mask,
                filtertype="intersects",
                invert_mask=True,
                return_inverse=False,
            )
            _log.info(
                f"Filtered out {len(primary_threshold_polygons) - len(inland_primary_threshold_polygons)} primary threshold polygons."
            )

            inland_secondary_threshold_polygons = filter_by_intersection(
                gpd_data=secondary_threshold_polygons,
                gpd_filter=land_sea_mask,
                filtertype="intersects",
                invert_mask=True,
                return_inverse=False,
            )
            _log.info(
                f"Filtered out {len(secondary_threshold_polygons) - len(inland_secondary_threshold_polygons)} secondary threshold polygons."
            )

            return inland_primary_threshold_polygons, inland_secondary_threshold_polygons

    else:
        _log.info("Skipping filtering out ocean polygons step.")
        return primary_threshold_polygons, secondary_threshold_polygons


def filter_using_urban_mask(
    primary_threshold_polygons: gpd.GeoDataFrame,
    secondary_threshold_polygons: gpd.GeoDataFrame,
    urban_mask_fp: str | Path = "",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Filter out the missclassified waterbodies from the primary and secondary
    threshold polygons using an urban/CBDs mask.
    WOfS has a known limitation, where deep shadows thrown by tall CBD buildings
    are misclassified as water. This results in 'waterbodies' around these
    misclassified shadows in capital cities.

    Parameters
    ----------
    primary_threshold_polygons : gpd.GeoDataFrame
    secondary_threshold_polygons : gpd.GeoDataFrame
    urban_mask_fp : str | Path, optional
        Vector file path to the polygons to use to filter out CBDs, by default ""

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        Primary threshold polygons with missclassified waterbodies removed.
    """
    crs = primary_threshold_polygons.crs

    if urban_mask_fp:
        _log.info(
            "Filtering out CBDs polygons from the primary and secondary threshold polygons..."
        )
        try:
            urban_mask = gpd.read_file(urban_mask_fp).to_crs(crs)
        except Exception as error:
            _log.exception(f"Could not read file {urban_mask_fp}")
            raise error
        else:
            cbd_filtered_primary_threshold_polygons = filter_by_intersection(
                gpd_data=primary_threshold_polygons,
                gpd_filter=urban_mask,
                filtertype="intersects",
                invert_mask=True,
                return_inverse=False,
            )

            _log.info(
                f"Filtered out {len(primary_threshold_polygons) - len(cbd_filtered_primary_threshold_polygons)} primary threshold polygons."
            )

            cbd_filtered_secondary_threshold_polygons = filter_by_intersection(
                gpd_data=secondary_threshold_polygons,
                gpd_filter=urban_mask,
                filtertype="intersects",
                invert_mask=True,
                return_inverse=False,
            )

            _log.info(
                f"Filtered out {len(secondary_threshold_polygons) - len(cbd_filtered_secondary_threshold_polygons)} secondary threshold polygons."
            )

            return (
                cbd_filtered_primary_threshold_polygons,
                cbd_filtered_secondary_threshold_polygons,
            )
    else:
        _log.info("Skipping filtering out CBDs step.")
        return primary_threshold_polygons, secondary_threshold_polygons


def merge_primary_and_secondary_threshold_polygons(
    primary_threshold_polygons: gpd.GeoDataFrame,
    secondary_threshold_polygons: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Identify secondary threshold polygons that intersect with the primary threshold
    polygons and merge them with the primary threshold polygons.

    Parameters
    ----------
    primary_threshold_polygons : gpd.GeoDataFrame
    secondary_threshold_polygons : gpd.GeoDataFrame

    Returns
    -------
    gpd.GeoDataFrame
        Merged primary and secondary threshold polygons.
    """
    assert primary_threshold_polygons.crs == secondary_threshold_polygons.crs
    crs = primary_threshold_polygons.crs

    _log.info("Merging the primary threshold and secondary threshold polygons...")
    # Find the polygons identified using the secondary threshold that intersect with those identified
    # using the primary threshold.
    do_intersect_with_primary = filter_by_intersection(
        gpd_data=secondary_threshold_polygons,
        gpd_filter=primary_threshold_polygons,
        filtertype="intersects",
        invert_mask=False,
        return_inverse=False,
    )

    # Combine the identified polygons  with the primary threshold polygons.
    combined_polygons = gpd.GeoDataFrame(
        pd.concat([do_intersect_with_primary, primary_threshold_polygons], ignore_index=True)
    )
    # Merge overlapping polygons.
    merged_combined_polygons_geoms = combined_polygons.unary_union
    # `Explode` the multipolygon back out into individual polygons.
    merged_combined_polygons = gpd.GeoDataFrame(crs=crs, geometry=[merged_combined_polygons_geoms])
    merged_combined_polygons = merged_combined_polygons.explode(index_parts=True)

    _log.info(f"Waterbody polygons count after merge: {len(merged_combined_polygons)}.")
    return merged_combined_polygons


def filter_using_major_rivers_mask(
    waterbody_polygons: gpd.GeoDataFrame, major_rivers_mask_fp: str | Path = ""
) -> gpd.GeoDataFrame:
    """
    Filter out major rivers polygons from a set of waterbody polygons.

    Parameters
    ----------
    waterbody_polygons : gpd.GeoDataFrame
    major_rivers_mask_fp : str | Path, optional
        Vector file path to the polygons to use to filter out major river waterbody polygons, by default ""

    Returns
    -------
    gpd.GeoDataFrame
        Filtered set of waterbody polygons with major rivers polygons removed.

    """
    crs = waterbody_polygons.crs

    if major_rivers_mask_fp:
        _log.info("Filtering out major rivers polygons from the waterbody polygons...")
        try:
            major_rivers = gpd.read_file(major_rivers_mask_fp).to_crs(crs)
        except Exception as error:
            _log.exception(f"Could not read file {major_rivers_mask_fp}")
            raise error
        else:
            major_rivers_filtered_polygons = filter_by_intersection(
                gpd_data=waterbody_polygons,
                gpd_filter=major_rivers,
                filtertype="intersects",
                invert_mask=True,
                return_inverse=False,
            )
            _log.info(
                f"Filtered out {len(waterbody_polygons) - len(major_rivers_filtered_polygons)} waterbody polygons."
            )
            return major_rivers_filtered_polygons
    else:
        _log.info("Skipping filtering out major rivers polygons step.")
        return waterbody_polygons


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


def erode_dilate_v1(
    waterbody_polygons: gpd.GeoDataFrame, pp_test_threshold: float | int
) -> gpd.GeoDataFrame:
    """
    Split large polygons using the `erode-dilate-v1` method.

    Parameters
    ----------
    waterbody_polygons : gpd.GeoDataFrame
        Polygons to split.
    pp_test_threshold: float | int
        Polsby–Popper value threshold below which a polygon is a candidate for
        splitting.
    Returns
    -------
    gpd.GeoDataFrame
        Polygons with polygons below the threshold split.
    """
    crs = waterbody_polygons.crs
    assert crs.is_projected

    # Calculate the Polsby–Popper values.
    waterbody_polygons_ = pp_test_gdf(input_gdf=waterbody_polygons)

    # Get the polygons to be split.
    splittable_polygons = waterbody_polygons_[waterbody_polygons_.pp_test <= pp_test_threshold]
    not_splittable_polygons = waterbody_polygons_[waterbody_polygons_.pp_test > pp_test_threshold]

    if len(splittable_polygons) >= 1:
        _log.info(f"Splitting {len(splittable_polygons)} polygons.")

        # Buffer the polygons.
        splittable_polygons_buffered = splittable_polygons.buffer(-50)

        # Explode multi-part geometries into multiple single geometries.
        split_polygons = splittable_polygons_buffered.explode(index_parts=True).reset_index(
            drop=True
        )

        # Buffer the polygons.
        split_polygons_buffered = split_polygons.buffer(50)

        # Merge the split polygons with the not split polygons.
        large_polygons_handled = pd.concat(
            [not_splittable_polygons.geometry, split_polygons_buffered], ignore_index=True
        )

        # Convert the Geoseries into a GeoDataFrame.
        large_polygons_handled = gpd.GeoDataFrame(geometry=large_polygons_handled, crs=crs)
        _log.info(
            f"Polygon count after splitting using erode-dilate-v1 method: {len(large_polygons_handled)}"
        )
        return large_polygons_handled
    else:
        info_msg = (
            f"There are no polygons with a Polsby–Popper score above the {pp_test_threshold}. "
            "No polygons were split."
        )
        _log.info(info_msg)

        # Return the geometry column only.
        waterbody_polygons_ = gpd.GeoDataFrame(
            geometry=waterbody_polygons_.geometry, crs=crs
        ).reset_index(drop=True)

        return waterbody_polygons_


def erode_dilate_v2(
    waterbody_polygons: gpd.GeoDataFrame, pp_test_threshold: float | int
) -> gpd.GeoDataFrame:
    """
    Split large polygons using the `erode-dilate-v2` method.

    Parameters
    ----------
    waterbody_polygons : gpd.GeoDataFrame
        Polygons to split.
    pp_test_threshold: float | int
        Polsby–Popper value threshold below which a polygon is a candidate for
        splitting.
    Returns
    -------
    gpd.GeoDataFrame
        Polygons with polygons below the threshold split.
    """
    crs = waterbody_polygons.crs
    assert crs.is_projected

    # Calculate the Polsby–Popper values.
    waterbody_polygons_ = pp_test_gdf(input_gdf=waterbody_polygons)

    # Get the polygons to be split.
    splittable_polygons = waterbody_polygons_[waterbody_polygons_.pp_test <= pp_test_threshold]
    not_splittable_polygons = waterbody_polygons_[waterbody_polygons_.pp_test > pp_test_threshold]

    if len(splittable_polygons) >= 1:
        _log.info(f"Splitting {len(splittable_polygons)} polygons.")

        # Buffer the polygons.
        splittable_polygons_buffered = splittable_polygons.buffer(-100).buffer(125)

        # Get the union of all the buffered polygons as a single geometry,
        splittable_polygons_buffered_union = gpd.GeoDataFrame(
            geometry=[splittable_polygons_buffered.unary_union], crs=crs
        )

        # Get the difference of each geometry in the splittable_polygons_ and the single geometry in
        # splittable_polygons_buffered_union
        subtracted = splittable_polygons.overlay(
            splittable_polygons_buffered_union, how="difference"
        )

        # Explode multi-part geometries into multiple single geometries.
        subtracted = subtracted.explode(index_parts=True).reset_index(drop=True)

        # Get the difference of each geometry in subtracted and each geometry in splittable_polygons.
        resubtracted = splittable_polygons.overlay(subtracted, how="difference")

        # Explode multi-part geometries into multiple single geometries.
        resubtracted = resubtracted.explode(index_parts=True).reset_index(drop=True)

        # Assign each chopped-off bit of the polygon to its nearest big
        # neighbour.
        unassigned = np.ones(len(subtracted), dtype=bool)
        recombined = []

        for row in resubtracted.itertuples():
            mask = subtracted.exterior.intersects(row.geometry.exterior) & unassigned
            neighbours = subtracted[mask]
            unassigned[mask] = False
            poly = row.geometry.union(neighbours.unary_union)
            recombined.append(poly)

        recombined_gdf = gpd.GeoDataFrame(geometry=recombined, crs=crs)
        # Get only the actual geometry objects that are neither missing nor empty
        recombined_gdf_masked = recombined_gdf[
            ~(recombined_gdf.geometry.is_empty | recombined_gdf.geometry.isna())
        ]

        # All remaining polygons are not part of a big polygon.
        results = pd.concat(
            [recombined_gdf_masked, subtracted[unassigned], not_splittable_polygons],
            ignore_index=True,
        )

        # Explode multi-part geometries into multiple single geometries.
        large_polygons_handled = results.explode(index_parts=True).reset_index(drop=True)

        # Return the geometry column only.
        large_polygons_handled = gpd.GeoDataFrame(geometry=large_polygons_handled.geometry, crs=crs)

        _log.info(
            f"Polygon count after splitting using erode-dilate-v2 method: {len(large_polygons_handled)}"
        )
        return large_polygons_handled
    else:
        info_msg = (
            f"There are no polygons with a Polsby–Popper score above the {pp_test_threshold}. "
            "No polygons were split."
        )
        _log.info(info_msg)

        # Return the geometry column only.
        waterbody_polygons_ = gpd.GeoDataFrame(
            geometry=waterbody_polygons_.geometry, crs=crs
        ).reset_index(drop=True)

        return waterbody_polygons_


def split_large_polygons(
    waterbody_polygons: gpd.GeoDataFrame, pp_test_threshold: float = 0.005, method: str = "nothing"
) -> gpd.GeoDataFrame:
    """
    Function to split large polygons.

    Parameters
    ----------
    waterbody_polygons : gpd.GeoDataFrame
        Set of polygons for which to split the large polygons.
    pp_test_threshold : float, optional
        Threshold for the Polsby–Popper test values of the polygons by which to
        classify if a polygon is large or not, by default 0.005
    method : str, optional
        Method to use to split large polygons., by default "nothing"

    Returns
    -------
    gpd.GeoDataFrame
        Set of polygons with large polygons split.
    """

    warnings.simplefilter(action="ignore", category=FutureWarning)

    # Confirm the option to use.
    valid_options = ["erode-dilate-v1", "erode-dilate-v2", "nothing"]
    if method not in valid_options:
        _log.info(
            f"{method} method not implemented to handle large polygons. Defaulting to not splitting large polygons."
        )
        method = "nothing"

    # Split large polygons.
    if method == "nothing":
        info_msg = (
            "You have chosen not to split large polygons. If you meant to use this option, please "
            f"select one of the following methods: {valid_options[:2]}."
        )
        _log.info(info_msg)

        # Return the geometry column only.
        waterbody_polygons_ = gpd.GeoDataFrame(
            geometry=waterbody_polygons.geometry, crs=waterbody_polygons.crs
        ).reset_index(drop=True)

        return waterbody_polygons_
    else:
        _log.info(
            f"Splitting large polygons using the `{method}` method, using the threshold {pp_test_threshold}."
        )
        if method == "erode-dilate-v1":
            large_polygons_handled = erode_dilate_v1(
                waterbody_polygons=waterbody_polygons, pp_test_threshold=pp_test_threshold
            )
            return large_polygons_handled
        elif method == "erode-dilate-v2":
            large_polygons_handled = erode_dilate_v2(
                waterbody_polygons=waterbody_polygons, pp_test_threshold=pp_test_threshold
            )
            return large_polygons_handled


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


def filter_waterbodies(
    primary_threshold_polygons: gpd.GeoDataFrame,
    secondary_threshold_polygons: gpd.GeoDataFrame,
    min_polygon_size: float = 4500,
    max_polygon_size: float = math.inf,
    land_sea_mask_fp: str | Path = "",
    urban_mask_fp: str | Path = "",
    major_rivers_mask_fp: str | Path = "",
    handle_large_polygons: str = "nothing",
    pp_test_threshold: float = 0.005,
) -> gpd.GeoDataFrame:
    """
    Apply filters to the primary and secondary threshold waterbody
    polygons.

    Parameters
    ----------
    primary_threshold_polygons : gpd.GeoDataFrame, optional
        Waterbody polygons generated using the primary threshold.
    secondary_threshold_polygons : gpd.GeoDataFrame, optional
        Waterbody polygons generated using the secondary threshold.
    min_polygon_size : float, optional
        Minimum area of a waterbody polygon to be included in the output polygons, by default 4500
    max_polygon_size : float, optional
        Maximum area of a waterbody polygon to be included in the output polygons, by default math.inf
    land_sea_mask_fp : str | Path, optional
        Vector file path to the polygons to use to filter out ocean waterbody polygons, by default ""
    urban_mask_fp : str | Path, optional
        Vector file path to the polygons to use to filter out CBDs, by default ""
    major_rivers_mask_fp : str | Path, optional
        Vector file path to the polygons to use to filter out major river waterbody polygons, by default ""
    handle_large_polygons : str, optional
        Method to use to split large water body polygons, by default "nothing"
    pp_test_threshold : float, optional
        Polsby-Popper test value to use when splitting large polygons using the method specified in `handle_large_polygons`, by default 0.005

    Returns
    -------
    gpd.GeoDataFrame
        Filtered set of waterbody polygons.
    """
    _log.info(f"Primary threshold polygons count {len(primary_threshold_polygons)}.")
    _log.info(f"Secondary threshold polygons count {len(secondary_threshold_polygons)}.")

    (
        area_filtered_primary_threshold_polygons,
        area_filtered_secondary_threshold_polygons,
    ) = filter_by_area(
        primary_threshold_polygons=primary_threshold_polygons,
        secondary_threshold_polygons=secondary_threshold_polygons,
        min_polygon_size=min_polygon_size,
        max_polygon_size=max_polygon_size,
    )

    (
        inland_primary_threshold_polygons,
        inland_secondary_threshold_polygons,
    ) = filter_using_land_sea_mask(
        primary_threshold_polygons=area_filtered_primary_threshold_polygons,
        secondary_threshold_polygons=area_filtered_secondary_threshold_polygons,
        land_sea_mask_fp=land_sea_mask_fp,
    )

    (
        cbd_filtered_primary_threshold_polygons,
        cbd_filtered_secondary_threshold_polygons,
    ) = filter_using_urban_mask(
        primary_threshold_polygons=inland_primary_threshold_polygons,
        secondary_threshold_polygons=inland_secondary_threshold_polygons,
        urban_mask_fp=urban_mask_fp,
    )

    merged_polygons = merge_primary_and_secondary_threshold_polygons(
        primary_threshold_polygons=cbd_filtered_primary_threshold_polygons,
        secondary_threshold_polygons=cbd_filtered_secondary_threshold_polygons,
    )

    major_rivers_filtered_polygons = filter_using_major_rivers_mask(
        waterbody_polygons=merged_polygons, major_rivers_mask_fp=major_rivers_mask_fp
    )

    large_polygons_handled = split_large_polygons(
        waterbody_polygons=major_rivers_filtered_polygons,
        pp_test_threshold=pp_test_threshold,
        method=handle_large_polygons,
    )

    # Reapply the size filtering, just to check that all of the split and filtered waterbodies are
    # still in the size range we want.
    area_filtered_large_polygons_handled, _ = filter_by_area(
        primary_threshold_polygons=large_polygons_handled,
        secondary_threshold_polygons=None,
        min_polygon_size=min_polygon_size,
        max_polygon_size=max_polygon_size,
    )

    # Return a GeoDataFrame with the geometry column only.
    filtered_polygons = gpd.GeoDataFrame(
        geometry=area_filtered_large_polygons_handled["geometry"],
        crs=area_filtered_large_polygons_handled.crs,
    )

    return filtered_polygons


def filter_hydrosheds_land_mask(hydrosheds_land_mask: xr.DataArray) -> xr.DataArray:
    """
    Function to filter the HydroSHEDs Land Mask into a boolean mask.
    """
    # Indicator values: 1 = land, 2 = ocean sink, 3 = inland sink, 255 is no data.
    boolean_mask = (hydrosheds_land_mask != 255) & (hydrosheds_land_mask != 2)
    return boolean_mask
