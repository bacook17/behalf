# example_accelerator.py
# Ben Cook (bcook@cfa.harvard.edU)

import numpy as np


def single_force(pos_a, pos_b, mass_a, mass_b, G=6.67e-11):
    r = pos_b - pos_a
    r3 = np.sum(r**2) ** 1.5
    return G * r * mass_a * mass_b / r3


def serial_newtonian(pos, mass, G=6.67e-11):
    """

    """
    pos = np.array(pos).astype(float)
    mass = np.array(mass).astype(float)
    
    N_part = pos.shape[0]
    assert mass.shape == (N_part,), ("input masses must match length of "
                                     "input positions")
    accel = np.zeros_like(pos, dtype=float)
    for i in range(N_part):
        for j in range(i+1, N_part):
            force_itoj = single_force(pos[i], pos[j], mass[i], mass[j], G=G)
            accel[i] += force_itoj / mass[i]
            accel[j] += -1. * force_itoj / mass[j]
    return accel
