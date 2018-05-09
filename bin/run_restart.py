#!/usr/bin/env python

"""
Restarts an existing simulation, using a results directory
For usage information, execute:

$ run_restart.py -h
"""

import numpy as np
from mpi4py import MPI
from time import time
from behalf import initialConditions
from behalf import integrator
from behalf import utils
import sys
import argparse
from glob import glob
import os

#hacky fix for this run
sys.setrecursionlimit(5000)

if __name__ == '__main__':
    # Initialize MPI communication
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Our unit system:
    # Length: kpc
    # Time: Myr
    # Mass: 10^9 M_sun

    GRAV_CONST = 4.483e-3  # Newton's Constant, in kpc^3 GM_sun^-1 Myr^-2

    # read command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run-name', help='REQUIRED - Name of the run',
                        type=str, required=True)
    parser.add_argument('--N-steps', help='Number of total steps',
                        type=int, default=-1)
    parser.add_argument('--verbose', help='Should diagnostics be printed?',
                        action='store_true')
    parser.add_argument('--production', help="Remove intentional slow-down for profiling",
                        action='store_true')
    args = parser.parse_args()

    run_name = args.run_name  # Unique run-name required, or --clobber set
    path = 'results/{:s}/'.format(run_name)
    with open(path + 'overview.txt', 'r') as f:
        overview_lines = f.readlines()
    overview_lines = [l.split(': ')[-1].split(' ')[0] for l in overview_lines]
    M_total = float(overview_lines[3]) / 1e9
    N_parts = int(overview_lines[2])
    M_part = M_total / N_parts  # mass of each particle (in 10^9 M_sun)
    if args.N_steps > 0:
        N_steps = args.N_steps
    else:
        N_steps = int(overview_lines[7])
    dt = float(overview_lines[6])
    softening = float(overview_lines[8])
    steps = []
    # figure out what time steps were saved
    for infile in glob( os.path.join(path, '*.dat') ):
        steps.append(int(infile.split('/')[-1].lstrip('step_').strip('.dat')))
    steps = sorted(steps)
    save_every = steps[-1] - steps[-2]
    final_step = steps[-1]
    THETA = float(overview_lines[5])
    verbose = args.verbose
    production = args.production  # If False, will synchronize MPI after each step
    
    # If we split "N_parts" particles into "size" chunks,
    # which particles does each process get?
    part_ids_per_process = np.array_split(np.arange(N_parts), size)
    # How many particles are on each processor?
    N_per_process = np.array([len(part_ids_per_process[p])
                              for p in range(size)])
    # data-displacements, for transferring data to-from processor
    displacements = np.insert(np.cumsum(N_per_process * 3), 0, 0)[0:-1]
    # How many particles on this process?
    N_this = N_per_process[rank]

    pos_full = None
    vel_full = None
    if rank == 0:
        t_start = time()
        results_dir = 'results/' + run_name + '/'
        masses = np.ones(N_parts) * M_part
        # load previous results!
        data_full = np.loadtxt(path + 'step_{:d}.dat'.format(final_step))
        pos_full = data_full[:, :3].copy(order='C')
        vel_full = data_full[:, 3:].copy(order='C')
        # Track how long each step takes
        timers = utils.TimerCollection()

    # The main integration loop
    for i in range(final_step+1, N_steps):
        # Construct the tree and compute forces
        if rank == 0:
            timers.start('Overall')
            timers.start('Tree Construction')
            tree = utils.construct_tree(pos_full, masses)
            timers.stop('Tree Construction')
            timers.start('Tree Broadcast')
        else:
            tree = None
        # broadcast the tree
        tree = comm.bcast(tree, root=0)
        if rank == 0:
            timers.stop('Tree Broadcast')
            timers.start('Scatter Particles')
        # scatter the positions and velocities
        pos = np.empty((N_this, 3))
        vel = np.empty((N_this, 3))
        comm.Scatterv([pos_full, N_per_process*3, displacements, MPI.DOUBLE],
                      pos, root=0)
        comm.Scatterv([vel_full, N_per_process*3, displacements, MPI.DOUBLE],
                      vel, root=0)
        if rank == 0:
            timers.stop('Scatter Particles')
            timers.start('Force Computation')
        # compute forces
        accels = utils.compute_accel(tree, part_ids_per_process[rank],
                                     THETA, GRAV_CONST, eps=softening)
        if not production:
            comm.Barrier()
        if rank == 0:
            timers.stop('Force Computation')
            timers.start('Time Integration')
        # forward one time step
        pos, vel = integrator.cuda_timestep(pos, vel, accels, dt)
        if not production:
            comm.Barrier()
        if rank == 0:
            timers.stop('Time Integration')
            timers.start('Gather Particles')
        # gather the positions and velocities
        pos_full = None
        vel_full = None
        if rank == 0:
            pos_full = np.empty((N_parts, 3))
            vel_full = np.empty((N_parts, 3))
        comm.Gatherv(pos, [pos_full, N_per_process*3, displacements, MPI.DOUBLE],
                     root=0)
        comm.Gatherv(vel, [vel_full, N_per_process*3, displacements, MPI.DOUBLE],
                     root=0)
        if rank == 0:
            timers.stop('Gather Particles')
            timers.stop('Overall')
            # Save the results to output file
            if ((i % save_every) == 0) or (i == N_steps - 1):
                utils.save_results(results_dir + 'step_{:d}.dat'.format(i), pos_full, vel_full, masses, t_start, i, N_steps,
                                   size, timers=timers)
                timers.clear()
            # Print status
            if verbose:
                print('Iteration {:d} complete. {:.1f} seconds elapsed.'.format(i, time() - t_start))
            sys.stdout.flush()
            sys.stderr.flush()
