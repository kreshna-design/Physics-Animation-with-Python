# Generates animation to show kinetic motion of particles in gas
# Plots the Maxwell-Boltzmann Distribution of the system
# Plots the Boltzmann Distribution of the system
# Bonus:Caclulates the temperature based on the plots of the system
# best fit found of both plots, mean T is taken for better result
# Bonus:Can't be stuck together and jitter back and forth
# This happens as dot product of relative v and position is
# negative when they approach each other, and + otherwise

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import combinations
import scipy.optimize

p_rad = 0.00001  # radius of particles in m
p_mass = 2.672*10**-26  # mass of particles in kg
Kb = 1.38064852*10**-23  # Boltzman constant in m^2kg*s^-2*K^-1

npoint = 400  # number of particles
nframe = 1000
xmin, xmax, ymin, ymax = 0, 1, 0, 1
fig, ax = plt.subplots()
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
Dt = 0.00001  # timestep in s


def update_velocity(vx, vy, x, y, p1, p2):
    # p1 and p2 are the colliding particles
    a = np.array([vx[p1]-vx[p2], vy[p1]-vy[p2]])
    # relative velocities of particles
    b = np.array([x[p1] - x[p2], y[p1] - y[p2]])
    # relative positions of particles
    c = np.dot(a, b) / np.dot(b, b)
    vx1f = vx[p1] - c * (x[p1] - x[p2])
    vx2f = vx[p2] - c * (x[p2] - x[p1])
    vy1f = vy[p1] - c * (y[p1] - y[p2])
    vy2f = vy[p2] - c * (y[p2] - y[p1])
    if np.dot(a, b) < 0:
        # when particles are gonna collide
        return np.array([vx1f, vx2f, vy1f, vy2f])
    else:
        return np.array([vx[p1], vx[p2], vy[p1], vy[p2]])
    # Makes sure only updates if moving towards each other


def update_point(num):
    global x, y, vx, vy
    print(num)
    indx = np.where((x < xmin) | (x > xmax))
    indy = np.where((y < ymin) | (y > ymax))
    vx[indx] = -vx[indx]
    vy[indy] = -vy[indy]
    # particle bounces off the wall
    xx = np.asarray(list(combinations(x, 2)))
    yy = np.asarray(list(combinations(y, 2)))
    dd = (xx[:, 0] - xx[:, 1])**2 + (yy[:, 0] - yy[:, 1])**2
    # list of all particles paired with other particles
    touch = np.where(dd <= 2*p_rad)
    # find what particles are close enough to interact
    index = np.asarray(list(combinations(range(npoint), 2)))
    # updates the velocity where particles interact
    for i in index[touch]:
        update = update_velocity(vx, vy, x, y, i[0], i[1])
        vx[i[0]] = update[0]
        vx[i[1]] = update[1]
        vy[i[0]] = update[2]
        vy[i[1]] = update[3]
    # updates to next time interval
    dx = Dt * vx
    dy = Dt * vy
    x = x + dx
    y = y + dy
    data = np.stack((x, y), axis=-1)
    im.set_offsets(data)
    # grabs frames for animation

x = np.random.random(npoint)
y = np.random.random(npoint)
vx = -500. * np.ones(npoint)  # horizontal velocity either 500 or -500
vy = np.zeros(npoint)  # vertical velocity starts at 0
vx[np.where(x <= 0.5)] = -vx[np.where(x <= 0.5)]
s = np.array([10])

colour = np.where(x < 0.5, 'blue', 'red')
# particles coloured by their initial velocities

# Animation
im = ax.scatter(x, y, color=colour)
im.set_sizes(s)
animation = animation.FuncAnimation(fig, update_point, nframe,
                                    interval=10, repeat=False)
animation.save('collisions.mp4')
plt.clf()

v = np.sqrt(vx ** 2+vy ** 2)
# Total velocity of particles
E = 0.5 * p_mass * v ** 2
# Kinetic Energy of particles


def f(v, T):
    return p_mass * v/(Kb * T) * np.e ** -(0.5 * p_mass * v ** 2/(Kb * T))
# Function for Maxwell-Boltzmann distribution of speed


def g(E, T):
    return 1/(Kb*T) * np.e ** -(E/(Kb * T))
# Function for Boltzmann distribution of kinetic Energy

fig = plt.hist(v, 100, normed=True)
fig1 = plt.hist(E, 100, normed=True)
# Taking lots of bins makes approximation better

# Maxwell-Boltzmann Distribution
plt.subplot(2, 1, 1)
T1, pcov = scipy.optimize.curve_fit(f, fig[1][0:100], fig[0], p0=300)
# fig[0] is probability distribution, fig[1] is velocity
v_line = np.linspace(min(v), max(v), 1000)
plt.hist(v, 100, normed=True, label='Probablitiy Distribution', color='yellow')
MB = f(v_line, T1)
plt.plot(v_line, MB, color='blue', label='Best Fit', linewidth=4)
leg = plt.legend(loc='upper right', fancybox=True)
leg.get_frame().set_facecolor('wheat')  # change the colour of the legend box
plt.title('Maxwell-Boltzmann Distribution',
          fontsize=22, weight='bold', color='red')
plt.xlabel('Velocity (m/s)', weight='bold', color='red')
plt.ylabel('Probability Density', weight='bold', color='red')
# Boltzmann Distribution
plt.subplot(2, 1, 2)
plt.title('Boltzmann-Distribution', fontsize=22, weight='bold', color='red')
plt.xlabel('Kinetic Energy (J)', weight='bold', color='red')
plt.ylabel('Probability Density', weight='bold', color='red')
T2, pcov = scipy.optimize.curve_fit(g, fig1[1][0:100], fig1[0], p0=300)
plt.hist(E, 100, normed=True, label='Probablitiy Distribution', color='red')
E_line = np.linspace(min(E), max(E), 1000)
MB = g(E_line, T2)
plt.plot(E_line, MB, color='black', label='Best Fit', linewidth=4)
leg1 = plt.legend(loc='upper right', fancybox=True)
leg1.get_frame().set_facecolor('wheat')  # change the colour of the legend box
plt.tight_layout()  # Cleans text in plot from overlapping
plt.savefig('distributions.pdf', facecolor='purple')

f = open('collisions.txt', 'w')
# Opening text file to write temperature
f.write('Temperature from Maxwell-Boltzmann Plot is {} Kelvin\n'.format(T1[0]))
f.write('Temperature from Boltzmann Plot is {} Kelvin\n'.format(T2[0]))
f.write('The temperature of the system is {} Kelvin'.format((T1[0]+T2[0])/2))
f.close()

