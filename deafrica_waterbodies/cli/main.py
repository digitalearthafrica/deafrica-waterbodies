import click

import deafrica_waterbodies.__version__
from deafrica_waterbodies.cli.filter_waterbody_polygons import filter_waterbody_polygons
from deafrica_waterbodies.cli.write_final_output import write_final_output


@click.version_option(package_name="deafrica_waterbodies", version=deafrica_waterbodies.__version__)
@click.group(help="Run deafrica-waterbodies.")
def main():
    pass


main.add_command(filter_waterbody_polygons)
main.add_command(write_final_output)
