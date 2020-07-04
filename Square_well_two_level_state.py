import numpy as np
import matplotlib.pyplot as plt


def two_scales(ax1, time, data1, data2, c1, c2, lab1, lab2):
    """

    Parameters
    ----------
    ax : axis
        Axis to put two scales on

    time : array-like
        x-axis values for both datasets

    data1: array-like
        Data for left hand scale

    data2 : array-like
        Data for right hand scale

    c1 : color
        Color for line 1

    c2 : color
        Color for line 2

    Returns
    -------
    ax : axis
        Original axis
    ax2 : axis
        New twin axis
    """
    ax2 = ax1.twinx()

    ax1.plot(time, data1, color=c1)
    ax1.set_xlabel('x/a')
    ax1.set_ylabel(lab1)

    ax2.plot(time, data2, color=c2)
    ax2.set_ylabel(lab2)
    return ax1, ax2


# Create psi and psi^2, recalling that psi is complex
#
t=1/6*np.pi  # THIS IS WHERE YOU CHANGE t
#
x = np.arange(0.01, 1.0, 0.01)
psi = np.sin(np.pi*x)*np.exp(-1j*t)+np.sin(np.pi*2*x)*np.exp(-4*1j*t)
psisq = np.conj(psi)*psi
momentum = 8/3*np.sin(3*t)

# Create axes
fig, ax = plt.subplots()
ax1, ax2 = two_scales(ax, x, np.real(psi), psisq, 'r', 'b', 'real(psi)', 'psi^2')


# Change color of each axis
def color_y_axis(ax, color):
    """Color your axes."""
    for t in ax.get_yticklabels():
        t.set_color(color)
    return None
color_y_axis(ax1, 'r')
color_y_axis(ax2, 'b')
plt.title('t in units of pi ='+str(t/np.pi)+'\n'+ 'Momentum at this t is: '+str(momentum))

fig, axB = plt.subplots()
ax1, ax2 = two_scales(axB, x, np.imag(psi), psisq, 'g', 'b','imag(psi)', 'psi^2')

color_y_axis(ax1, 'g')
color_y_axis(ax2, 'b')

plt.show()
