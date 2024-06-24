#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on our clusters. 
# ---------------------------------------------------------------------
#SBATCH --account=def-wainberg
#SBATCH --cpus-per-task=64
#SBATCH --time=05:00:00
#SBATCH --mem=900G

pseudobulk.py