#!/usr/bin/env bash

set -euo pipefail
set -x

# Install anaconda-client via pip and upload artifacts without requiring conda
python -m pip install --upgrade anaconda-client
anaconda -t "$ANACONDA_STAGING_TOKEN" upload --force -u gurobi-machinelearning-wheels-staging dist/artifact/*
