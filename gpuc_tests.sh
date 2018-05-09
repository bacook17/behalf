#!/bin/bash
FILES=./slurm_scripts/gpuc_scalings_v2/*.slurm
for filename in $FILES; do
    echo $filename
    sbatch $filename
done