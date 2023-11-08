import click
from datacube.ui.click import parse_expressions

from deafrica_waterbodies.cli.logs import logging_setup
from deafrica_waterbodies.make_timeseries import generate_timeseries_from_wofs_ls


@click.command(
    "generate-timeseries",
    no_args_is_help=True,
)
@click.option(
    "--waterbodies-vector-file",
    type=click.Path(),
    default=None,
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
    "--start-date",
    type=str,
    default=None,
    help="Date string. E.g. 2019-01-01. "
    "The start date for the waterbody timeseries query. If --start-date "
    "is provided --end-date must also be provided.",
)
@click.option(
    "--end-date",
    type=str,
    default=None,
    help="Date string. E.g. 2019-12-01. "
    "The end date for the waterbody timeseries query. If --end-date is "
    "provided --start-date must also be provided.",
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
    "--include-uncertainity/--do-not-include-uncertainity",
    default=True,
    help="Option to include uncertainities in the output timeseries."
    "If you specify --include-uncertainity then you will only "
    "filter out timesteps with 100% invalid pixels. Else you will "
    "filter out timesteps with more than 10% invalid pixels",
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
    start_date,
    end_date,
    missing_only,
    include_uncertainity,
    subset_polygon_ids,
    verbose,
):
    """
    Generate timeseries for a set of waterbody polygons.
    """
    logging_setup(verbose=verbose)

    # Convert strings to datetime.
    if time_span == "custom":
        time_expression = parse_expressions(f"time in [{start_date}, {end_date}]")
        start_date_dt = time_expression["time"].begin
        end_date_dt = time_expression["time"].end
    else:
        start_date_dt = None
        end_date_dt = None

    generate_timeseries_from_wofs_ls(
        waterbodies_vector_file=waterbodies_vector_file,
        output_directory=output_directory,
        use_id=use_id,
        missing_only=missing_only,
        time_span=time_span,
        start_date=start_date_dt,
        end_date=end_date_dt,
        subset_polygons_ids=subset_polygon_ids,
        include_uncertainity=include_uncertainity,
    )
