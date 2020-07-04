import scipy as sp
from scipy import constants, integrate, linalg
import matplotlib.pyplot as plt
from matplotlib import animation

### Paramters of interest ###
############################################################
# Initial wavefunction (t=0)
def wf_0(x,A,a):
	return A*x*(a-x)
num_states = 15	# Number of states in the basis
num_steps = 1000 # Number of steps in the animation
time_per_step = 10e3/num_steps # [s]
############################################################

### Specify the infinite square well domain
a = 1
xs = sp.linspace(start=0, stop=a, num=int(1e3), endpoint=True) # Basis is (0 <= x <= a)

### Initialize the wave function
A = 1 # set to 1 before normalization
# Initial probability density (t=0)
def rho_0(x,A,a):
	return abs(wf_0(x, A, a))**2

### Normalize the wavefunction
(y, abserr) = integrate.quad(func=rho_0, args=(A,a), a=0, b=a)
A = sp.sqrt(1/y)
print('Normalization Constant: {:0.3e}'.format(A))

### Expand the wave function over energy eigenstates
# Function to create the n'th eigenstate
def psi_n_inf_sqr_well(x, a, n):
	psi_n = sp.sqrt(2/a)*sp.sin(n*sp.pi/a*x)
	return psi_n
# Function to generate a "basis" out of the first n eigenstates
def basis_inf_sqr_well(x, a, num_states): 
	basis = sp.array([]).reshape(len(x),0)
	for n in range(1, num_states+1):
		psi_n = psi_n_inf_sqr_well(x, a, n)
		basis = sp.hstack((basis, psi_n[:,sp.newaxis]))
	return basis
basis = basis_inf_sqr_well(xs, a, num_states) # Create a basis to expand over
cns = linalg.lstsq(basis, wf_0(xs,A,a))[0] # Find the least-squares projection of wf_0 onto the basis
print('|c_1|^2 = {:0.6f}'.format(abs(cns[0])**2)) # Compare to Griffith's value
print('Sum(|c_n|^2) = {:0.6f}'.format(sp.sum(abs(cns)**2))) # Ensure coefficient's sum to 1

### Plot the initial wavefunction, its expansion, and the time evolved solution
fig, ax = plt.subplots(2,1)
ax[0].plot(xs,abs(wf_0(xs,A,a)), label='$|\\mathrm{wf}_0|$')
ax[0].plot(xs,abs(sp.dot(basis,cns)), label='$|\\mathrm{wf}_0$ Expansion|')
wf_t_plt, = ax[0].plot([],[], label='$|\\mathrm{wf}_{t}|$')
ax[0].set_title('Initial, Expanded, and Time-Evolved Wavefunction')
ax[0].set_xlabel('x [m]')
ax[0].set_ylabel('$|\\Psi(x,t)|$')
ax[0].legend(loc='upper right')

### plot magnitude-squard c_n values
ax[1].stem(
	sp.arange(1,len(cns)+1),
	abs(cns)**2,
	linefmt='C1-',
	markerfmt='C1o',
	basefmt='k')
ax[1].set_title('Expansion Coefficients')
ax[1].set_xlabel('n')
ax[1].set_ylabel('$|c_n|^2$')
plt.tight_layout()

### Propagate wavefucntion in time
# Function to create the time-dependence of the n'th eigenstate
def time_dependence_inf_sqr_well(t, a, num_states, m=1):
	n = sp.arange(1, num_states+1)
	En = (n*sp.pi*constants.hbar/a)**2/(2*m)
	phi_n = sp.exp(-1j*En*t/constants.hbar) 
	return phi_n
def init(): # initialization function: plot the background of each frame
    wf_t_plt.set_data([], [])
    return wf_t_plt,
def animate(i): # animation function.  This is called sequentially
	phi_t = time_dependence_inf_sqr_well(i*time_per_step, a, num_states, m=constants.electron_mass)
	wf_t = abs(sp.dot(basis,cns*phi_t))
	wf_t_plt.set_data(xs,wf_t)
	return wf_t_plt,

### Call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=(num_steps+1), interval=20, blit=True, repeat=False)

### Save the animation
# anim.save('gaussian_sqr_well.mp4', fps=30, extra_args=['-vcodec', 'libx264'], dpi=300)

### Show Plot
plt.show()


