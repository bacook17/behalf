import numpy as np


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
