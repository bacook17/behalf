#!/bin/bash
#SBATCH -p shared
#SBATCH -J gal_test_1 # Job Name
#SBATCH -n 8 # Number of MPI tasks
#SBATCH -N 1 #ensure that all cores are on one machine
#SBATCH -t 0-01:00 #runtime in D-HH:MM
#SBATCH --mem-per-cpu 4000 #memory per MPI task
#SBATCH -o logs/gal_test_1.out
#SBATCH -e logs/gal_test_1.err
#SBATCH --mail-type=BEGIN,END,FAIL #alert when done
#SBATCH --mail-user=bcook@cfa.harvard.edu #Email to send to

mpiexec -n $SLURM_NTASKS run_behalf.py --run-name gal_test_1 --clobber --N-parts 100 
RESULT=${PIPESTATUS[0]}
sacct -j "${SLURM_JOB_ID}".batch --format=JOBID%20,JobName,NTasks,AllocCPUs,AllocGRES,Partition,Elapsed,MaxRSS,MaxVMSize,MaxDiskRead,MaxDiskWrite,State 
exit $RESULT
