#!/bin/bash
#SBATCH --job-name=dmpoint-%j
#SBATCH --output=/home/ga78fed/output/dmpoint-%j.out
#SBATCH --error=/home/ga78fed/output/dmpoint_error-%j.out
#SBATCH --time=01:00:00
#SBATCH --mem=2G
#SBATCH --array=1001-2000

bash $HOME/projects/dmpoint/cluster/dmpoint_scan.sh ${SLURM_ARRAY_TASK_ID}
