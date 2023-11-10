#!/bin/bash

# Add the wofs_ls_summary_alltime datasets.
s3-to-dc "s3://deafrica-services/wofs_ls_summary_alltime/1-0-0/x194/y117/*/*.json" --stac --no-sign-request --skip-lineage 'wofs_ls_summary_alltime'

# Add the wofs_ls datasets covering the waterbody UID: sm9rtw98n
s3-to-dc "s3://deafrica-services/wofs_ls/1-0-0/187/038/2023/01/*/*.json" --stac --no-sign-request --skip-lineage 'wofs_ls'
s3-to-dc "s3://deafrica-services/wofs_ls/1-0-0/188/037/2023/01/*/*.json" --stac --no-sign-request --skip-lineage 'wofs_ls'
s3-to-dc "s3://deafrica-services/wofs_ls/1-0-0/188/038/2023/01/*/*.json" --stac --no-sign-request --skip-lineage 'wofs_ls'
s3-to-dc "s3://deafrica-services/wofs_ls/1-0-0/189/037/2023/01/*/*.json" --stac --no-sign-request --skip-lineage 'wofs_ls'
s3-to-dc "s3://deafrica-services/wofs_ls/1-0-0/189/038/2023/01/*/*.json" --stac --no-sign-request --skip-lineage 'wofs_ls'
