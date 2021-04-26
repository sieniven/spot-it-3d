#!/bin/bash
set -e

# source conda environment
export PATH="/home/$USER/anaconda3/bin:$PATH"
source activate mcmt-track

# run mcmt-tracker
python ./mcmt/mcmt_tracker.py