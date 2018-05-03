import numpy as np
from time import time
from datetime import datetime, timedelta


def compute_energy(pos, vel, mass, G=6.67e-11):
    return compute_potential_energy(pos, mass, G=G) +\
        compute_kinetic_energy(vel, mass)


def compute_potential_energy(pos, mass, G=6.67e-11):
    pos = np.array(pos).astype(float)
    mass = np.array(mass).astype(float)
    N_part = pos.shape[0]
    assert mass.shape == (N_part,), ("input masses must match length of "
                                     "input positions")
    U = 0.
    for i in range(N_part):
        for j in range(i+1, N_part):
            r = np.sqrt(np.sum((pos[i] - pos[j])**2))
            U -= G * mass[i] * mass[j] / r
    return U


def compute_kinetic_energy(vel, mass):
    vel = np.array(vel).astype(float)
    mass = np.array(mass).astype(float)
    N_part = vel.shape[0]
    assert mass.shape == (N_part,), ("input masses must match length of "
                                     "input velocities")
    return np.sum(vel.T**2. * mass) * 0.5


def save_results(pos, vel, t_start, iter_num, iter_total, num_cores, out_file):
    header = ""
    header += 'Num Particles: {:d}\n'.format(len(pos))
    header += 'Num Cores: {:d}\n'.format(num_cores)
    header += 'Iterations: {:d} of {:d}\n'.format(iter_num, iter_total)
    header += 'Current Time: {:s}\n'.format(str(datetime.now()))
    dt = time()-t_start
    header += 'Elapsed Time: {:s}\n'.format(str(timedelta(seconds=dt)))
    ave_dt = dt / iter_num
    header += 'Avg. Step Time: {:s}\n'.format(str(timedelta(seconds=ave_dt)))
    header += '\n'
    header += 'x\ty\tz\tvx\tvy\tvz\n'
    np.savetxt(out_file, np.append(pos, vel, axis=-1), header=header,
               fmt='%+8.4f', delimiter='\t')


def split_size(N_parts, N_chunks, i):
    return (N_parts // N_chunks) + int((N_parts % N_chunks) > i)
