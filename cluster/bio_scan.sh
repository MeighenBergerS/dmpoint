#!/bin/bash

source /home/ga78fed/miniconda3/bin/activate
# source "$HOME/.virtualenvs/AdS/bin/activate"
# source /home/ga76zas/.virtualenvs/Ads/bin/activate
# source /home/ga76zas/miniconda3/bin/activate
# conda activate pythonenv
python --version
which python
python $HOME/projects/prob_model/prob_model/env_check.py
python $HOME/projects/prob_model/prob_model/prob_run_batch.py "${1}"
