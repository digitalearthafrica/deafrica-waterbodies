import click

import deafrica_waterbodies.__version__
from deafrica_waterbodies.cli.generate_polygons import generate_polygons
from deafrica_waterbodies.cli.generate_timeseries import generate_timeseries

@click.version_option(package_name="deafrica_waterbodies", version=deafrica_waterbodies.__version__)
@click.group(help="Run deafrica-waterbodies.")
def main():
    pass


main.add_command(generate_polygons)
main.add_command(generate_timeseries)
