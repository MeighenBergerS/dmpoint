#!/bin/bash
#SBATCH --job-name=bio_1_sim_fourth_day-%j
#SBATCH --output=/home/ga78fed/output/bio_sim_fourth_day-%j.out
#SBATCH --error=/home/ga78fed/output/bio_sim_fourth_day_error-%j.out
#SBATCH --time=02:00:00
#SBATCH --mem=2G
#SBATCH --array=1-1000

bash $HOME/projects/prob_model/prob_model/bio_scan.sh ${SLURM_ARRAY_TASK_ID}
