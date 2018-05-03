import integrator
import utils
from mpi4py import MPI
### import tree_calc
### import initial_conditions
import argparse
from time import time
from datetime import datetime

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N_parts = 1000  # how many particles?
N_steps = 100  # how many time steps?
dt = 0.1  # size of time step
save_every = 1  # how often to save output results

# If we split "N_parts" particles into "size" chunks,
# how many particles does this process get?
parts_per_process = [utils.split_size(N_parts, size, r)
                     for r in range(size)]
N_rec = parts_per_process[rank]

if rank == 0:
    results_dir = 'testrun_1/'

    # Set Plummer Sphere (or other) initial conditions
    #######################
    # pos_init, vel_init = ### initial_conditions.plummer(N_parts)
    #######################

    # Self-start the Leap-Frog algorithm, all on the main node
    # Construct the tree and compute forces
    #######################
    # tree = tree_calc.build(pos_init)
    # accels = tree.get_forces(pos_init)
    #######################
    # get v_1/2 = v_0 + a_0 * (dt / 2)
    _, vel = integrator.cuda_timestep(pos_init, vel_init, accels, dt/2.)
    # get x_1 = x_0 + v_1/2 * dt
    pos, _ = integrator.cuda_timestep(pos_init, vel, accels, dt)

# The main integration loop
if rank == 0:
    t_start = time()
for i in range(N_steps):
    # Construct the tree and compute forces
    if rank == 0:
        #######################
        # tree = tree_calc.build(pos)
        #######################
    else:
        tree = None
    # broadcast the tree
    tree = comm.bcast(tree, root=0)
    # scatter the positions and velocities
    if rank == 0:
        pos_send = np.array_split(pos, parts_per_process)
        vel_send = np.array_split(vel, parts_per_process)
    else:
        data_send = None
    pos = np.empty((N_rec, 3))
    vel = np.empty((N_rec, 3))
    comm.Scatter(pos_send, pos, root=0)
    comm.Scatter(vel_send, vel, root=0)
    # compute forces
    ################
    # accels = tree.get_forces(pos)
    ################
    # forward one time step
    pos, vel = integrator.cuda_timestep(pos, vel, accels, dt)
    # gather the positions and velocities
    pos_rec = None
    vel_rec = None
    if rank == 0:
        pos_rec = [np.empty((N, 3)) for N in parts_per_process]
        vel_rec = [np.empty((N, 3)) for N in parts_per_process]
    comm.Gather(pos, pos_rec, root=0)
    comm.Gather(vel, vel_rec, root=0)
    # Merge positions and velocities together
    if rank == 0:
        pos = np.concatenate(pos_rec)
        vel = np.concatenate(vel_rec)

    # Save the results to output file
    if ((i % save_every) == 0) or (i == N_steps - 1):
        utils.save_results(pos, vel, t_start, i, N_steps, size,
                           results_dir + 'step_{:d}.dat'.format(i))
