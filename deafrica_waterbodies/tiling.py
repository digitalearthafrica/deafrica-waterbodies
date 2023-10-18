import logging
import multiprocessing
from functools import partial

import datacube
import geopandas as gpd
import tqdm
from datacube.api import GridWorkflow
from datacube.model import GridSpec
from odc.dscache.tools.tiling import parse_gridspec_with_name

_log = logging.getLogger(__name__)


def check_tile_intersects_polygons(
    polygons_gdf: gpd.GeoDataFrame | None,
    tile: tuple[tuple[int, int], datacube.api.grid_workflow.Tile],
) -> tuple[int, int]:
    """
    Check if the extent of the geobox of a tile intersects with a set of polygons.

    Parameters
    ----------
    polygons_gdf : gpd.GeoDataFrame | None
    tile : tuple[tuple[int,int], datacube.api.grid_workflow.Tile]

    Returns
    -------
    tuple
        Tile id if the extent of the geobox of a tile intersects with the polygons.
    """
    tile_id = tile[0]
    tile_extent = tile[1].geobox.extent

    if polygons_gdf is not None:
        # Reproject the extent of the geobox of a tile to match the polygons.
        tile_extent = tile_extent.to_crs(polygons_gdf.crs)
        # Get the shapely geometry of the reprojected extent of the tile's geobox.
        tile_extent_geom = tile_extent.geom
        # Check if the extent intersects with any of the polygons.
        check_intersection = polygons_gdf.geometry.intersects(tile_extent_geom).any()
        if check_intersection:
            return tile_id
        else:
            return ()
    else:
        return tile_id


def filter_tiles(
    tiles: dict[tuple[int, int], datacube.api.grid_workflow.Tile],
    polygons_gdf: gpd.GeoDataFrame | None,
    num_workers: int = 8,
) -> list[tuple[int, int]]:
    """
    Filter out tiles that do not intersect with a set of polygons, using a
    multi-process approach to run the `check_tile_intersects_polygons` function.

    Parameters
    ----------
    tiles: dict[tuple[int,int], datacube.api.grid_workflow.Tile]
        A dictionary of tiles.
    polygons_gdf : gpd.GeoDataFrame | None
        A set of polygons in a GeoDataFrame
    num_workers : int, optional
        Number of worker processes to use during multi-processing, by default 8

    Returns
    -------
    list[tuple[int, int]]
        A list of the filtered tiles ids.
    """
    with multiprocessing.Pool(processes=num_workers) as pool:
        filtered_tile_ids_ = list(
            tqdm.tqdm(
                pool.imap(partial(check_tile_intersects_polygons, polygons_gdf), tiles.items())
            )
        )

    # Remove empty tuples.
    filtered_tile_ids = [item for item in filtered_tile_ids_ if item]

    return filtered_tile_ids


def tile_wofs_ls_summary_alltime(
    tile_size_factor: float = 2,
) -> tuple[dict, datacube.api.GridWorkflow]:
    """
    Retile the `wofs_ls_summary_alltime` product into tiles `tile_size_factor`
    number of times bigger than the regular tiles.

    Parameters
    ----------
    tile_size_factor : float, optional
        Number of times to increase the regular tile size by, by default 2

    Returns
    -------
    tuple[dict, datacube.api.GridWorkflow]
        Tiles for the `wofs_ls_summary_alltime` product.
        GridWorkflow to use to load the tiles.
    """
    # Define a spatial grid with tiles tile_size_factor times the size of the regular
    # wofs_ls_summary_alltime grid.

    # Regular grid.
    grid = "africa_30"
    grid, gridspec = parse_gridspec_with_name(grid)

    # Multiply the tile size.
    tile_size = tuple(tile_size_factor * elem for elem in gridspec.tile_size)
    _log.info(f"New tile size is {tile_size}.")

    # Define new grid.
    gs = GridSpec(
        crs=gridspec.crs,
        tile_size=tile_size,
        resolution=gridspec.resolution,
        origin=gridspec.origin,
    )

    # Connect to the datacube
    dc = datacube.Datacube()

    # Define the grid workflow.
    gw = GridWorkflow(index=dc.index, grid_spec=gs)

    # Tile the wofs_ls_summary_alltime product using the new grid.
    tiles = gw.list_cells(product="wofs_ls_summary_alltime")

    _log.info(f"Number of wofs_ls_summary_alltime tiles: {len(tiles)}")

    return tiles, gw


def get_tiles(
    aoi_gdf: gpd.GeoDataFrame | None, tile_size_factor: float = 2, num_workers: int = 8
) -> tuple[dict, datacube.api.GridWorkflow]:
    """
    Get the tile of the WOfS All Time Summary whose extents intersect
    with any of the area of interest polygons.

    Parameters
    ----------
    aoi_gdf : gpd.GeoDataFrame | None
        Area of interest
    tile_size_factor : float, optional
        Number of times to increase the regular tile size when tiling the
        wofs_ls_summary_alltime product by, by default 2
    num_workers : int, datasetsoptional
        Number of worker processes to use when filtering tiles, by default 8
    Returns
    -------
    tuple[dict, datacube.api.GridWorkflow]
        WOfS All Time Summary tiles whose extents intersect
        with any of the area of interest polygons.
        GridWorkflow to use to load the tiles.
    """

    tiles, grid_workflow = tile_wofs_ls_summary_alltime(tile_size_factor=tile_size_factor)

    # Filter the tiles to the area of interest.
    if aoi_gdf is not None:
        filtered_tile_ids = filter_tiles(tiles=tiles, polygons_gdf=aoi_gdf, num_workers=num_workers)
        filtered_tiles = {k: v for k, v in tiles.items() if k in filtered_tile_ids}
        return filtered_tiles, grid_workflow
    else:
        return tiles, grid_workflow
