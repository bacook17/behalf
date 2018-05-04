import numpy as np
from mpi4py import MPI
from time import time
from initial_conditions import plummerSphere
import integrator
import utils


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Our unit system:
    # Length: kpc
    # Time: Myr
    # Mass: 10^9 M_sun

    GRAV_CONST = 4.483e-3  # Newton's Constant, in kpc^3 GM_sun^-1 Myr^-2
    THETA = 0.5
    
    M_total = 1e5  # total mass of system (in 10^9 M_sun)
    N_parts = 1000  # how many particles?
    M_part = M_total / N_parts  # mass of each particle (in 10^9 M_sun)
    a = 10.0  # scale radius (in kpc)
    N_steps = 100  # how many time steps?
    dt = 0.1  # size of time step (in Myr)
    softening = 0.01  # softening length (in kpc)
    save_every = 1  # how often to save output results
    seed = 1234  # Initialize state identically every time
    
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
        results_dir = 'testrun_1/'

        # Set Plummer Sphere (or other) initial conditions
        pos_init, vel_init = plummerSphere.plummer(N_parts, a, m=M_part,
                                                   G=GRAV_CONST, seed=seed)
        pos_init -= np.mean(pos_init, axis=0)
        vel_init -= np.mean(vel_init, axis=0)
        masses = np.ones(N_parts) * M_part
        
        # Self-start the Leap-Frog algorithm, all on the main node
        # Construct the tree and compute forces
        tree = utils.construct_tree(pos_init, masses)
        accels = utils.compute_accel(tree, np.arange(N_parts),
                                     THETA, GRAV_CONST)
        # Half-kick
        _, vel_full = integrator.cuda_timestep(pos_init, vel_init, accels,
                                               dt/2.)
        # Full-drift
        pos_full, _ = integrator.cuda_timestep(pos_init, vel_full, accels, dt)
        # From now on, the Leapfrog algorithm can do Full-Kick + Full-Drift
    else:
        pos_full, vel_full = None, None
    
        # The main integration loop
    if rank == 0:
        t_start = time()
    for i in range(N_steps):
        # Construct the tree and compute forces
        if rank == 0:
            tree = utils.construct_tree(pos_full, masses)
        else:
            tree = None
        # broadcast the tree
        tree = comm.bcast(tree, root=0)
        # scatter the positions and velocities
        pos = np.empty((N_this, 3))
        vel = np.empty((N_this, 3))
        comm.Scatterv([pos_full, N_per_process*3, displacements, MPI.double],
                      pos, root=0)
        comm.Scatterv([vel_full, N_per_process*3, displacements, MPI.double],
                      vel, root=0)
        # compute forces
        accels = utils.compute_accel(tree, part_ids_per_process[rank],
                                     THETA, GRAV_CONST) / M_part
        # forward one time step
        pos, vel = integrator.cuda_timestep(pos, vel, accels, dt)
        # gather the positions and velocities
        pos_full = None
        vel_full = None
        if rank == 0:
            pos_full = np.empty((N_parts, 3))
            vel_full = np.empty((N_parts, 3))
        comm.Gatherv(pos, [pos_full, N_per_process*3, displacements, MPI.double],
                     root=0)
        comm.Gatherv(vel, [vel_full, N_per_process*3, displacements, MPI.double],
                     root=0)

        # Save the results to output file
        if rank == 0:
            if ((i % save_every) == 0) or (i == N_steps - 1):
                utils.save_results(pos_full, vel_full, t_start, i, N_steps,
                                   size, results_dir + 'step_{:d}.dat'.format(i))
