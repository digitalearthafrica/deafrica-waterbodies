import logging
import os
import re
import urllib
import uuid
from pathlib import Path

import boto3
import fsspec
import geopandas as gpd
from botocore.client import ClientError
from mypy_boto3_s3 import S3Client

_log = logging.getLogger(__name__)

# File extensions to recognise as Parquet files.
PARQUET_EXTENSIONS = {".pq", ".parquet"}


def check_if_s3_uri(file_path: str | Path) -> bool:
    """
    Checks if a file path is an S3 URI.

    Parameters
    ----------
    file_path : str | Path
        File path to check

    Returns
    -------
    bool
        True if the file path is an S3 URI.
    """
    # "Support" pathlib Paths.
    file_path = str(file_path)

    file_scheme = urllib.parse.urlparse(file_path).scheme

    valid_s3_schemes = ["s3"]

    if file_scheme in valid_s3_schemes:
        return True
    else:
        return False


def check_dir_exists(dir_path: str | Path) -> bool:
    """
    Checks if a specified path is an existing directory.

    Parameters
    ----------
    dir_path : str | Path
        Path to check.

    Returns
    -------
    bool
        True if the path exists and is a directory.
        False if the path does not exists or if the path exists and it is not a directory.
    """
    # "Support" pathlib Paths.
    dir_path = str(dir_path)

    if check_if_s3_uri(dir_path):
        fs = fsspec.filesystem("s3")
    else:
        fs = fsspec.filesystem("file")

    if fs.exists(dir_path):
        if fs.isdir(dir_path):
            return True
        else:
            return False
    else:
        return False


def check_file_exists(file_path: str | Path) -> bool:
    """
    Checks if a specified path is an existing file.

    Parameters
    ----------
    file_path : str | Path
        Path to check.

    Returns
    -------
    bool
        True if the path exists and is a file.
        False if the path does not exists or if the path exists and it is not a file.
    """
    # "Support" pathlib Paths.
    file_path = str(file_path)

    if check_if_s3_uri(file_path):
        fs = fsspec.filesystem("s3")
    else:
        fs = fsspec.filesystem("file")

    if fs.exists(file_path):
        if fs.isfile(file_path):
            return True
        else:
            return False
    else:
        return False


def check_s3_bucket_exists(bucket_name: str, s3_client: S3Client = None):
    """
    Check if a bucket exists and if the user has permission to access it.

    Parameters
    ----------
    bucket_name : str
        Name of s3 bucket to check.
    s3_client : S3Client
        A low-level client representing Amazon Simple Storage Service (S3), by default None.

    """
    # Get the service client.
    if s3_client is None:
        s3_client = boto3.client("s3")

    try:
        response = s3_client.head_bucket(Bucket=bucket_name)  # noqa E501
    except ClientError as error:
        error_code = int(error.response["Error"]["Code"])

        if error_code == 403:
            raise PermissionError(f"{bucket_name} is a private Bucket. Forbidden Access!")
        elif error_code == 404:
            raise FileNotFoundError(f"Bucket {bucket_name} Does Not Exist!")
    except Exception as error:
        _log.exception(error)
        raise error


# From https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html
def upload_file_to_s3(
    file_name: str, bucket_name: str, object_name: str = None, s3_client: S3Client = None
) -> bool:
    """
    Upload a file to an S3 bucket.


    Parameters
    ----------
    file_name : str
        File path of the file to upload.
    bucket_name : str
        Name of the s3 bucket to upload to.
    object_name : str, optional
        S3 object name. If not specified then `file_name` is used, by default None
    s3_client : S3Client
        A low-level client representing Amazon Simple Storage Service (S3), by default None.

    Returns
    -------
    bool
        True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Get the service client.
    if s3_client is None:
        s3_client = boto3.client("s3")

    # Upload the file
    try:
        response = s3_client.upload_file(file_name, bucket_name, object_name)  # noqa F841
    except ClientError as error:
        _log.exception(error)
        return False
    else:
        return True


def write_waterbodies_to_file(
    waterbodies_gdf: gpd.GeoDataFrame,
    product_version: str,
    output_directory: str | Path,
    file_name_prefix: str = "waterbodies",
):
    """
    Function to write waterbody polygons to an ESRI Shapefile.

    Parameters
    ----------
    waterbodies_gdf : gpd.GeoDataFrame
        The waterbody polygons.
    product_version: str,
        The DE Africa Waterbodies service product version.
    output_directory : str | Path,
        S3 URI or File URI of the directory to write the waterbody polygons to.
    file_name_prefix: str, optional
        Prefix to use when naming the output file(s).
    """
    output_fn = f"{file_name_prefix}v{product_version.replace('.', '-')[0]}.shp"
    output_fp = os.path.join(output_directory, output_fn)

    if check_if_s3_uri(output_directory):
        # Get the service client.
        s3_client = boto3.client("s3")

        # Get the bucket name and object prefix.
        output_bucket_name = urllib.parse.urlparse(output_directory).netloc
        output_object_prefix = urllib.parse.urlparse(output_directory).path.lstrip("/").rstrip("/")

        # Make a temporary folder locally.
        fs = fsspec.filesystem("file")
        myuuid = str(uuid.uuid1())[:6]
        local_temp_dir = f"temp_{myuuid}"
        fs.mkdirs(local_temp_dir)

        # Write the waterbodies to the temporary folder.
        local_temp_shapefile_fp = os.path.join(local_temp_dir, output_fn)
        waterbodies_gdf.to_file(local_temp_shapefile_fp)

        # Upload each object in the temporary folder into s3.
        local_temp_files = [i["name"] for i in fs.listdir(local_temp_dir)]
        for local_temp_file in local_temp_files:
            upload_file_to_s3(
                file_name=local_temp_file,
                bucket_name=output_bucket_name,
                object_name=f"{output_object_prefix}/{os.path.split(local_temp_file)[-1]}",
                s3_client=s3_client,
            )

        # Remove the temporary directory
        fs.rm(local_temp_dir, recursive=True)
    else:
        waterbodies_gdf.to_file(output_fp)

    _log.info(f"Waterbody polygons written to {output_fp}")


def find_parquet_files(path: str | Path, pattern: str = ".*") -> [str]:
    """
    Find Parquet files matching a pattern.

    Arguments
    ---------
    path : str | Path
        Path (s3 or local) to search for Parquet files.

    pattern : str
        Regex to match file names against.

    Returns
    -------
    [str]
        List of paths.
    """
    pattern = re.compile(pattern)

    # "Support" pathlib Paths.
    path = str(path)

    if check_if_s3_uri(path):
        file_system = fsspec.filesystem("s3")
    else:
        file_system = fsspec.filesystem("file")

    pq_file_paths = []

    files = file_system.find(path)
    for file in files:
        _, file_extension = os.path.splitext(file)
        if file_extension not in PARQUET_EXTENSIONS:
            continue
        else:
            _, file_name = os.path.split(file)
            if not pattern.match(file_name):
                continue
            else:
                pq_file_paths.append(file)

    if check_if_s3_uri(path):
        pq_file_paths = [f"s3://{file}" for file in pq_file_paths]

    return pq_file_paths


def convert_shapefile_2_parquet(shapefile_fp: str | Path):
    """
    Convert a shapefile to a parquet file.

    Parameters
    ----------
    shapefile_fp : str | Path
        File path or S3 URI of the shapefile to convert.

    """
    shapefile_fp = str(shapefile_fp)

    # Get the parent directory of the shapefile.
    dir_name = os.path.dirname(shapefile_fp)
    # Get the file name of the shapefile without the file extenstion.
    base_name = os.path.splitext(os.path.basename(shapefile_fp))[0]

    # Get the parquet file path.
    parquet_fp = os.path.join(dir_name, f"{base_name}.parquet")

    # Read the shapefile.
    try:
        shapefile_gdf = gpd.read_file(shapefile_fp)
    except Exception as error:
        _log.exception(f"Could not read file {shapefile_fp}")
        _log.error(error)
        raise error

    # Save the GeoDataFrame to a parquet file.
    shapefile_gdf.to_parquet(parquet_fp)
    _log.info(f"Saved to {parquet_fp}")
