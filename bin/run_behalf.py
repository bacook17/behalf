#!/usr/bin/env python

"""
Runs a N-body simulation with Plummer sphere initial conditions,
using the Behalf code. Takes command-line arguments for configuration.

For usage information, execute:

$ run_behalf.py -h
"""

import numpy as np
from mpi4py import MPI
from time import time
from behalf import initialConditions
from behalf import integrator
from behalf import utils
import sys
import argparse
import os

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
    parser.add_argument('--N-parts', help='REQUIRED - Number of particles',
                        type=int, required=True)
    parser.add_argument('--total-mass', help=('Total mass of the system '
                                              '(in GMsun)'),
                        type=float, default=1e5)
    parser.add_argument('--radius', help='Scale Radius (in kpc)',
                        type=float, default=10.)
    parser.add_argument('--N-steps', help='Number of time steps',
                        type=int, default=1000)
    parser.add_argument('--dt', help='Size of time step (in Myr)',
                        type=float, default=0.01)
    parser.add_argument('--softening', help='Softening length (in kpc)',
                        type=float, default=0.01)
    parser.add_argument('--save-every', help='How often to save results',
                        type=int, default=10)
    parser.add_argument('--THETA', help='Barnes-Hut Approximation Range',
                        type=float, default=0.5)
    parser.add_argument('--rand-seed', help='Random seed to initialize',
                        type=int, default=1234)
    parser.add_argument('--clobber', help='Overwrite previous results?',
                        action='store_true')
    parser.add_argument('--verbose', help='Should diagnostics be printed?',
                        action='store_true')
    parser.add_argument('--production', help="Remove intentional slow-down for profiling",
                        action='store_true')
    parser.add_argument('--no-cython', help="Dont use Cython",
                        action='store_true')
    args = parser.parse_args()

    run_name = args.run_name  # Unique run-name required, or --clobber set
    M_total = args.total_mass  # total mass of system (in 10^9 M_sun)
    N_parts = args.N_parts  # how many particles?
    M_part = M_total / N_parts  # mass of each particle (in 10^9 M_sun)
    a = args.radius  # scale radius (in kpc)
    N_steps = args.N_steps  # how many time steps?
    dt = args.dt  # size of time step (in Myr)
    softening = args.softening  # softening length (in kpc)
    save_every = args.save_every  # how often to save output results
    THETA = args.THETA  # Barnes-Hut approximation range - 0.5 works well
    seed = args.rand_seed  # Initialize state identically every time
    clobber = args.clobber
    verbose = args.verbose
    production = args.production  # If False, will synchronize MPI after each step
    use_cython = ~args.no_cython
    
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
    
    if rank == 0:
        t_start = time()
        results_dir = 'results/' + run_name + '/'
        try:
            os.makedirs(results_dir)
        except FileExistsError:
            if clobber:
                pass
            else:
                assert False, 'Directory already exists, and clobber not set'
        # Save overview of parameters of this run
        utils.summarize_run(results_dir + 'overview.txt', run_name, size,
                            N_parts, M_total, a, THETA, dt, N_steps, softening,
                            seed)
        # Set Plummer Sphere (or other) initial conditions
        pos_init, vel_init = initialConditions.plummer(N_parts, a, m=M_part,
                                                       G=GRAV_CONST, seed=seed)
        # Set center-of-mass and mean velocity to zero
        pos_init -= np.mean(pos_init, axis=0)
        vel_init -= np.mean(vel_init, axis=0)
        masses = np.ones(N_parts) * M_part
        
    # Self-start the Leap-Frog algorithm
    if rank == 0:
        # Construct the tree and compute forces
        tree = utils.construct_tree(pos_init, masses)
    else:
        tree = None
        pos_init = None
        vel_init = None
    # broadcast the initial tree
    tree = comm.bcast(tree, root=0)
    # scatter the initial positions and velocities
    pos = np.empty((N_this, 3))
    vel = np.empty((N_this, 3))
    comm.Scatterv([pos_init, N_per_process*3, displacements, MPI.DOUBLE],
                  pos, root=0)
    comm.Scatterv([vel_init, N_per_process*3, displacements, MPI.DOUBLE],
                  vel, root=0)
    # compute forces
    accels = utils.compute_accel(tree, part_ids_per_process[rank],
                                 THETA, GRAV_CONST, eps=softening, cython=use_cython)
    # Half-kick
    _, vel = integrator.cuda_timestep(pos, vel, accels,
                                      dt/2.)
    # Full-drift
    pos, _ = integrator.cuda_timestep(pos, vel, accels, dt)
    # From now on, the Leapfrog algorithm can do Full-Kick + Full-Drift
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
    
    # The main integration loop
    if rank == 0:
        if verbose:
            print('Starting Integration Loop')
        sys.stdout.flush()
        sys.stderr.flush()
        # Track how long each step takes
        timers = utils.TimerCollection()
    for i in range(N_steps):
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
                                     THETA, GRAV_CONST, eps=softening, cython=use_cython)
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
