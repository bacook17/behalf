#!/bin/bash
FILES=./slurm_scripts/gpuc_scalings/*.slurm
for filename in $FILES; do
    echo $filename
    sbatch $filename
done