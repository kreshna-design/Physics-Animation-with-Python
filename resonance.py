
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


#defined constants and variables
g=9.8 #m/s^2
m=1 #kg
L=1 #m
F=0.1
b=0.15
I=m*L**2

#equation of motion for pendulum
def pendulum(y,t,w):
 theta,omega=y
 c1=(b/I)
 c2=(m*g*L)/I
 v=y[1] # this is dvdt
 dydt= -c1*omega-c2*np.sin(theta)+(F/I)*np.cos(w*t)*np.cos(theta)
 return [v,dydt]


t=np.linspace(0,143,600)#time interval
v0=[0,0] #initial conditions

new_array=np.zeros(100)#creates an and array with size 100
z=np.linspace(2,5,100)
for w, i in zip(z,range(0,100)): #solve odeint function for different values of w and find maximum Θ(t)
  y=odeint(pendulum,v0,t,args=(w,))
  max_amp=np.amax(y[:,0])
  new_array[i]=max_amp


plt.plot(z,new_array)
plt.xlabel("Angular Frequency(w)")
plt.ylabel("Maximum amplitude(Θ(t)")
plt.savefig('resonance.pdf')

#ANIMATION PART
#ASUMME W=3.1
w=3.1
y1=odeint(pendulum,v0,t,args=(w,))

#set some properties for animation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(0.6, 1.5), ylim=(-0.1, 1.1))
ax.grid()
line, = ax.plot([], [], 'o-', lw=2)

#uptade plots for animation
def update_line(num):
 line.set_data([1-np.sin(y1[:,0][num]),1],[1-np.cos(y1[:,0][num]),1])
 return line,

number_of_frames=600
imani = animation.FuncAnimation(fig, update_line, number_of_frames,interval=100,repeat=False)
imani.save('pendulum.mp4')


#PLOTS showing Θ(t) and its time derivative dΘ/dt as function of time
# ω=ω0
w=np.sqrt(g/L)
y2=odeint(pendulum,v0,t,args=(w,))
plt.figure(1)
plt.subplot(311)
plt.xlabel("Time(s)")
plt.plot(t,y2[:,0],label="Θ(t) in radians")
plt.plot(t,y2[:,1],label= "dΘ/dt in m/s")
plt.title("w=w_0")
plt.legend()

# ω<<ω0
w=0.2
y3=odeint(pendulum,v0,t,args=(w,))

plt.subplot(312)
plt.xlabel("Time(s)")
plt.plot(t,y3[:,0],label="Θ(t)in randians")
plt.plot(t,y3[:,1],label="dΘ/dt in m/s")
plt.legend()
plt.title("w<<ω_0")
#ω>>ω0

w=10.0
y4=odeint(pendulum,v0,t,args=(w,))
plt.subplot(313)
plt.xlabel("Time(s)")
plt.plot(t,y3[:,0],label="Θ(t) in radians")
plt.plot(t,y3[:,1],label="dΘ/dt in m/s")
plt.legend()
plt.title("w >>ω_0")
plt.savefig('phase_space.pdf')

