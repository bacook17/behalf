#!/bin/bash
#SBATCH -p conroy
#SBATCH -J gal_test_2 # Job Name
#SBATCH -n 8 # Number of MPI tasks
#SBATCH -N 4 #ensure that all cores are NOT on one machine
#SBATCH -t 0-04:00 #runtime in D-HH:MM
#SBATCH --mem-per-cpu 4000 #memory per MPI task
#SBATCH -o logs/gal_test_2.out
#SBATCH -e logs/gal_test_2.err
#SBATCH --mail-type=BEGIN,END,FAIL #alert when done
#SBATCH --mail-user=bcook@cfa.harvard.edu #Email to send to

mpiexec -n $SLURM_NTASKS python run_sim.py
RESULT=${PIPESTATUS[0]}
sacct -j "${SLURM_JOB_ID}".batch --format=JOBID%20,JobName,NTasks,AllocCPUs,AllocGRES,Partition,Elapsed,MaxRSS,MaxVMSize,MaxDiskRead,MaxDiskWrite,State 
exit $RESULT
