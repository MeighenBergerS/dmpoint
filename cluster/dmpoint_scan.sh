#!/bin/bash

source /home/ga78fed/miniconda3/bin/activate
# source "$HOME/.virtualenvs/AdS/bin/activate"
# source /home/ga76zas/.virtualenvs/Ads/bin/activate
# source /home/ga76zas/miniconda3/bin/activate
# conda activate dmpoint
python --version
which python
echo "Current population ${1}"
python $HOME/projects/dmpoint/cluster/env_check.py
python $HOME/projects/dmpoint/cluster/dm_source_batch.py "${1}"
