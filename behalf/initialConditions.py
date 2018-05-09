from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from future import standard_library
import numpy as np
standard_library.install_aliases()


def rand_unit_vector(d, rand=np.random):
    """
    Returns d-dimensional random unit vector (norm = 1)
    """
    phi = rand.uniform(0, np.pi*2)
    costheta = rand.uniform(-1, 1)
    theta = np.arccos(costheta)
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    vec = np.array([x, y, z])
    return vec


def plummer(Npart, a, m=1., G=4.483e-3, seed=None):
    """
    Returns the positions and velocities of particles in the Plummer sphere
    input:
        Npart - total number of particles to be initialized
        a - Plummer radius (scale parameter setting the cluster core)
        m - mass of particles (if single value, all particles will be
               initialized with the same mass; if array, each particles will
               have mass m_i)
        G - Gravitational constant; by default, work in Kpc-GM_sun-Myr units
        seed - random generator seed (int, optional: defaults to numpy seed)
    output:
        [pos, vel] - [(Npart x 3), (Npart x 3)] of positions and velocities
                        in cartesian coordinates

    """
    if seed is None:
        rand = np.random
    else:
        rand = np.random.RandomState(seed=seed)
    Npart = int(Npart)
    pos = PlummerDist_3d_xyz(Npart, a, rand=rand)
    if np.size(m) == 1:
        M = Npart * m  # if all particles have the same mass
    else:
        M = np.sum(m)
    vel = velDist_Plummer(Npart, pos, M, a, G, rand=rand)
    return [pos, vel]


# Spatial Distribution for Plummer Model
def PlummerDist_3d_xyz(Npart, a, rand=np.random):
    """
    Initializes particles in 3d with Plummer density profile.
    input:
        Npart - total number of particles to be initialized
        a - Plummer radius (scale parameter setting the cluster core)
        rand - RandomState generator (optional: defaults to numpy.random)
    output:
        pos   - (Npart x 3) array of positions in cartesian coordinates
    """
    Npart = int(Npart)
    r = np.zeros((Npart))
    pos = np.zeros((Npart, 3))
    for i in range(Npart):
        # Let enclosed mass fraction f_mi be random number between 0 and 1
        f_mi = rand.uniform(0., 1.)
        r[i] = a/np.sqrt(f_mi**(-2./3.)-1.)
        pos[i] = r[i]*rand_unit_vector(3, rand=rand)
    return pos


# Initial velocities for particles in the Plummer model
def velEscape_Plummer(r, M, a, G):
    """
    Returns the escape velocity of a particle in the Plummer model
    at a radius r from the center of the Plummer sphere
    input:
        r - radius away from the center of the Plummer sphere
        M - total mass of the Plummer sphere
        a - Plummer radius (scale parameter setting the cluster core)
        G - gravitational constant
    output:
        escape velocity of a particle at radius r inside a Plummer sphere
    """
    # r_abs = np.linalg.norm(r)
    pref = np.sqrt(2.*G*M/a)
    return pref*(1.+(r*r)/(a*a))**(-0.25)


def rejTech_velPlummer(r, M, a, G, rand=np.random):
    """
    Uses the rejection technique to return a velocity modulus drawn at random
    from the velocity distribution of particles at radius r in Plummer model
    input:
        r - radius away from the center of the Plummer sphere
        M - total mass of the Plummer sphere
        a - Plummer radius (scale parameter setting the cluster core)
        G - gravitational constant
        rand - RandomState generator (optional: defaults to numpy.random)
    output:
        absoulte velocity of a particle at radius r inside a Plummer sphere
    """
    x0 = 0.
    gmax = 0.1  # slightly bigger than g_max = g(\sqrt(2/9)) = 0.092
    g0 = gmax
    while g0 > g_Plummer(x0):
        # 0 <= v <= v_esc or 0 <= x <= 1 where x = v/v_esc
        x0 = rand.uniform(0., 1.)
        # 0 <= g <= g_max
        g0 = rand.uniform(0., gmax)
    return x0*velEscape_Plummer(r, M, a, G)


def g_Plummer(x):
    """
    Plummer gravitational acceleration profile
    """
    return x*x * (1. - x*x)**3.5


def velDist_Plummer(Npart, r, M, a, G, rand=np.random):
    """
    Returns velocities from the velocity distribution of particles
    at radius r in the Plummer model
    input:
        Npart - total number of particles to be initialized
        r     - array of radii away from the center of the sphere (size Npart)
        M     - total mass of the Plummer sphere
        a     - Plummer radius (scale parameter setting the cluster core)
        G     - gravitational constant
        rand - RandomState generator (optional: defaults to numpy.random)
    output:
        vel   - array of velocities of particles in Plummer model at radii r
    """
    vel = np.zeros((Npart, 3))
    for i in range(Npart):
        r_abs = np.linalg.norm(r[i])
        vel_mod = rejTech_velPlummer(r_abs, M, a, G, rand=rand)
        vel[i, :] = rand_unit_vector(3, rand=rand)*vel_mod
    return vel


def circVel_Plummer(r, M, a, G, rand=np.random):
    """
    Returns the circular velocity at the given radii of the Plummer model.
    input:
        r     - array of radii away from the center of the sphere (size Npart)
        M     - total mass of the Plummer sphere
        a     - Plummer radius (scale parameter setting the cluster core)
        G     - gravitational constant
        rand - RandomState generator (optional: defaults to numpy.random)
    output:
        vel   - array of circular velocities of particles at radii r
    """
    N = r.shape[0]
    vel = np.zeros((N, 3))
    for i in range(N):
        v_hat = rand_unit_vector(3, rand=rand)
        vel[i, :] = v_hat * velEscape_Plummer(np.linalg.norm(r[i]), M, a, G)/2.
    return vel

