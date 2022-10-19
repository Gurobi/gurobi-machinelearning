#!/bin/bash

set -e
set -x

export PATH=$CONDA/bin:$PATH
conda create -n upload -y python=3.10
source activate upload
conda install -y anaconda-client

anaconda -t $ANACONDA_TOKEN upload --force -u gurobi-machinelearning-wheels-staging dist/artifact/*
