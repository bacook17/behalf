from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from behalf import octree
from builtins import range
from builtins import open
from builtins import int
from builtins import str
from future import standard_library
from builtins import object
import numpy as np
from time import time
from datetime import datetime, timedelta
standard_library.install_aliases()


def construct_tree(pos, mass):
    sim_box = octree.bbox(np.array([np.min(pos, axis=0),
                                    np.max(pos, axis=0)]).T)
    return octree.octree(pos, mass, sim_box)


def compute_accel(tree, part_ids, theta, G, eps=0.1):
    if type(part_ids) == int:
        return tree.accel(theta, part_ids, G, eps=eps)
    else:
        return np.array([tree.accel(theta, p_id, G, eps=eps)
                         for p_id in part_ids])


class TimerCollection(object):
    def __init__(self):
        self.start_times = {}
        self.completed_times = {}

    def start(self, watch_name):
        self.start_times[watch_name] = time()
        if watch_name not in self.completed_times:
            self.completed_times[watch_name] = []

    def stop(self, watch_name):
        try:
            dt = time() - self.start_times.pop(watch_name)
            self.completed_times[watch_name].append(dt)
        except KeyError:
            raise KeyError('No such timer started')

    def iter_medians(self):
        for k in sorted(self.completed_times.keys()):
            yield k, np.median(self.completed_times[k])

    def clear(self):
        self.__init__()

def compute_energy(pos, vel, mass=None, G=4.483e-3):
    """
    Returns the total energy of the system defined by
    input positions (pos), velocities (vel), and masses (defaults
    to 1.0 for all particles), using units defined by choice of G.

    Total energy is the sum of potential and kinetic energy
    (see: compute_potential_energy and compute_kinetic_energy)

    Input:
       pos - positions (N x d)
       vel - velocities (N x d)
       mass - masses (optonal. Default: np.ones(N))
       G - Newton's Constant (optional. Default=1.)

    Output:
       E - total energy (float)
    """
    return compute_potential_energy(pos, mass=mass, G=G) +\
        compute_kinetic_energy(vel, mass=mass)


def compute_potential_energy(pos, mass=None, G=4.483e-3):
    """
    Returns the gravitational potential energy of the system defined by input
    positions (pos) and masses (defaults to 1.0 for all particles),
    using units defined by choice of G.
    
    Potential energy is defined as:
    U = - sum_i ( sum_[j > i] ( G * mass[i] * mass[j] / r_ij))
    r_ij = || pos[i] - pos[j] ||

    Input:
       pos - positions (N x d)
       mass - masses (optonal. Default: np.ones(N))
       G - Newton's Constant (optional. Default=1.)

    Output:
       U - Gravitational potential energy (float)
    """
    pos = np.array(pos).astype(float)
    N_part = pos.shape[0]
    if mass is None:
        mass = np.ones(N_part)
    elif type(mass) is float:
        mass = np.ones(N_part) * mass
    else:
        mass = np.array(mass).astype(float)
    assert mass.shape == (N_part,), ("input masses must match length of "
                                     "input positions")
    U = 0.
    for i in range(N_part):
        m_i = mass[i]
        m_j = mass[i+1:]
        dr = np.linalg.norm(pos[i] - pos[i+1:], axis=1)
        U -= np.sum(G * m_i * m_j / dr)
    return U


def compute_kinetic_energy(vel, mass=None):
    """
    Returns the kinetic of the system defined by input
    velocities and mass (defaults to 1.0 for all particles).
    
    Kinetic energy is defined as:
    K = 1/2 sum_i (mass[i] * ||vel[i]||**2)

    Input:
       vel - velocities (N x 3)
       mass - masses (optonal. Default: np.ones(N))

    Output:
       K - kinetic energy (float)
    """
    vel = np.array(vel).astype(float)
    N_part = vel.shape[0]
    if mass is None:
        mass = np.ones(N_part)
    elif type(mass) is float:
        mass = np.ones(N_part) * mass
    else:
        mass = np.array(mass).astype(float)
    N_part = vel.shape[0]
    assert mass.shape == (N_part,), ("input masses must match length of "
                                     "input velocities")
    return np.sum(vel.T**2. * mass) * 0.5


def save_results(out_file, pos, vel, mass, t_start, iter_num, iter_total,
                 num_cores, G=4.483e-3, timers=None):
    """
    Saves the current state of the simulation to "out_file".
    
    Input:
       out_file - filename to save results to
       pos - array of particle positions (N x d)
       vel - array of particle velocities (N x d)
       mass - array of particle masses (N)
       t_start - start time (in seconds) of the simulation
       iter_num - current time step of the simulation
       iter_total - total number of iterations the simulation will run for
       num_cores - number of cores used for computation
       timers - TimerCollection of custom timers to save
    """
    header = ""
    header += 'Iterations: {:d} of {:d}\n'.format(iter_num+1, iter_total)
    K = compute_kinetic_energy(vel, mass=mass)
    U = compute_potential_energy(pos, mass=mass, G=G)
    E_total = K+U
    header += 'Total Energy: {:.3e}\n'.format(E_total)
    header += '   Kinetic Energy: {:.3e}\n'.format(K)
    header += '   Potential Energy: {:.3e}\n'.format(U)
    header += 'Current Time: {:s}\n'.format(str(datetime.now()))
    dt = time()-t_start
    header += 'Elapsed Time: {:s}\n'.format(str(timedelta(seconds=dt)))
    if timers is not None:
        header += '\nMed. Times for Sections\n'
        for name, avg in timers.iter_medians():
            header += '   {:s}:\t\t{:.6f}\n'.format(name, avg)
            header += '       ['
            for t in timers.completed_times[name]:
                header += '{:.6f}, '.format(t)
            header += ']\n'
    header += '\n'
    header += 'x\ty\tz\tvx\tvy\tvz\n'
    np.savetxt(out_file, np.append(pos, vel, axis=-1), header=header,
               fmt='%+8.4f', delimiter='\t')


def summarize_run(out_file, run_name, N_cores, N_parts, M_total, a, theta, dt,
                  N_steps, softening, seed):
    with open(out_file, 'w') as f:
        f.write('# Run Name: {:s}\n'.format(run_name))
        f.write('# Number of Cores: {:d}\n'.format(N_cores))
        f.write('# Num Particles: {:d}\n'.format(N_parts))
        f.write('# Total Mass: {:.2e} M_sun\n'.format(M_total * 1e9))
        f.write('# Scale Radius: {:.2f} kpc\n'.format(a))
        f.write('# Theta: {:.2f}\n'.format(theta))
        f.write('# Time Step: {:.2g} Myr\n'.format(dt))
        f.write('# Number of Steps: {:d}\n'.format(N_steps))
        f.write('# Force Softening: {:.2f} kpc\n'.format(softening))
        f.write('# Random Seed: {:d}\n'.format(seed))

        
def split_size(N_parts, N_chunks, i):
    """
    Returns number of particles (out of N_parts) distributed to
    chunk i of N_chunks

    Input:
       N_parts - number of particles (int)
       N_chunks - number of chunks to distribute to (int)
       i - which chunk to compute number of particles (int, 0-indexed)

    Example: splitting 1000 particles across 10 chunks
    >>> split_size(1000, 11, 0)
    91
    >>> split_size(1000, 11, 10)
    90
    """
    return (N_parts // N_chunks) + int((N_parts % N_chunks) > i)


