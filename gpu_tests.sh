#!/bin/bash
for filename in slurm_scripts/gpu_scaling/*.dat; do
    sbatch filename
done