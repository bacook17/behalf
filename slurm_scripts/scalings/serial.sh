#!/bin/bash
#SBATCH -p hernquist
#SBATCH -J serial_test # Job Name
#SBATCH -n 1 # Number of MPI tasks
#SBATCH -N 1 #ensure that all cores are on one machine
#SBATCH -t 0-01:00 #runtime in D-HH:MM
#SBATCH --mem-per-cpu 1000 #memory per MPI task
#SBATCH -o ../../logs/%x.out #logs/%x.out
#SBATCH -e ../../logs/%x.err #logs/%x.err
#SBATCH --mail-type=BEGIN,END,FAIL #alert when done
#SBATCH --mail-user=ana-roxana.pop@cfa.harvard.edu #Email to send to

mpiexec -n $SLURM_NTASKS ./../../bin/run_behalf.py --run-name $SLURM_JOB_NAME --clobber --verbose --N-parts 10 --N-steps 10 --dt 0.1
RESULT=${PIPESTATUS[0]}
sacct -j "${SLURM_JOB_ID}".batch --format=JOBID%20,JobName,NTasks,AllocCPUs,AllocGRES,Partition,Elapsed,MaxRSS,MaxVMSize,MaxDiskRead,MaxDiskWrite,State 
exit $RESULT
