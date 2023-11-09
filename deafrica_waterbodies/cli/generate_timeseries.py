import click

from deafrica_waterbodies.cli.logs import logging_setup
from deafrica_waterbodies.make_timeseries import generate_timeseries_from_wofs_ls


@click.command(
    "generate-timeseries",
    no_args_is_help=True,
)
@click.option(
    "--waterbodies-vector-file",
    type=click.Path(),
    help="REQUIRED. Path to the waterbody polygons vector file you "
    "want to run the time series generation for.",
)
@click.option(
    "--use-id",
    type=str,
    default=None,
    help="Optional. Unique key id in polygons vector file.",
)
@click.option(
    "--output-directory",
    type=click.Path(),
    default=None,
    help="REQUIRED. File URI or S3 URI of the directory to write the " "timeseries csv files to.",
)
@click.option(
    "--time-span",
    type=click.Choice(["all", "append", "custom"]),
    default="all",
    help="Sets the time range for the waterbody timeseries queries. "
    "If you select APPEND, then only times since the latest dates in "
    "the waterbody timeseries will be run. If --time-span = custom, "
    "then --start-date and --end-date must also be specified.",
)
@click.option(
    "--temporal-range",
    type=str,
    default=None,
    help="Time range to generate the timeseries for, if `time_span` is set to"
    "`custom`. Example '2020-05--P1M' for the month of May 2020, by default",
)
@click.option(
    "--missing-only/--not-missing-only",
    default=False,
    help="Specifies whether you want to only run "
    "waterbody polygons that DO NOT already have a .csv file "
    "in the --output-directory directory. The default option is to run "
    "every waterbody polygon in the --waterbodies-vector-file file, and overwrite "
    "any existing csv files.",
)
@click.option(
    "--subset-polygon-ids",
    default=None,
    help="List of polygon ids in the --waterbodies-vector-file " "to generate the timeseries for.",
)
@click.option("-v", "--verbose", count=True)
def generate_timeseries(
    waterbodies_vector_file,
    use_id,
    output_directory,
    time_span,
    temporal_range,
    missing_only,
    subset_polygon_ids,
    verbose,
):
    """
    Generate timeseries for a set of waterbody polygons.
    """
    logging_setup(verbose=verbose)

    # Parse string to list.
    if subset_polygon_ids is not None:
        subset_polygon_ids = subset_polygon_ids.split(",")
    else:
        subset_polygon_ids = []

    generate_timeseries_from_wofs_ls(
        waterbodies_vector_file=waterbodies_vector_file,
        output_directory=output_directory,
        missing_only=missing_only,
        time_span=time_span,
        temporal_range=temporal_range,
        subset_polygons_ids=subset_polygon_ids,
    )
