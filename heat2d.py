# Code animates heat flow simulation on 2D square plate

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# constants
L = 0.01  # side lengths of square plate in m
D = 4.25*10**-6  # units: m^2 s^-1
N = 100  # Number of increments
ds = L/N  # small increment of plate sides, as dy=dx
dt_c = 0.00050  # dt for convergance
dt_d = 0.00059  # dt for divergance
Th, Tm, Tl = 400., 250., 200.  # initial temperatures in K
tend = 7.  # total time for heat exchange
fig = plt.figure()


def heat_transfer(dt, filename, nframes):
    global N
    T = np.full([N+1, N+1], Tm)
    T[0, :] = T[N, :] = Th
    T[:, 0] = T[:, N] = Tl
    # Array of initial temp on the plate
    c = dt*D/(ds ** 2)
    t = 0.0  # initial time
    ims = []  # initial list of frames
    n = 0.0  # iteration count
    while t < tend:
        N = T.shape[0]
        T[1:-1, 1:-1] += c * (T[1:-1, 2:N] + T[1:-1, 0:-2] + T[0:-2, 1:-1] +
                              T[2:N, 1:-1] - 4*T[1:-1, 1:-1])
        # array slicing to update the system
        t += dt
        n += 1
        if n % nframes == 0.0:  # Collect frames every 1000 itterations
            ims.append((plt.imshow(np.copy(T), ),))
    imani = animation.ArtistAnimation(fig, ims, repeat=False)
    imani.save('heat2d_{}' .format(filename))

# Animate
heat_transfer(dt_c, 'converged.mp4', 400)  # Convergant animation
heat_transfer(dt_d, 'diverged.mp4', 200)  # Divergant animation

