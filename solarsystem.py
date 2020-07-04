import numpy as np 
from scipy.integrate import odeint # for solving differential equation 
import matplotlib.pyplot as plt # for plotting
import matplotlib.animation as animation # for animation

# constants
G = 6.67408e-11 # gravitational constant (m^2 kg^-1 s^-2)
m_sun = 1.989e30 # mass of Sun (kg)
m_earth = 5.972e24 # mass of Earth (kg)
m_mars = 6.39e23 # mass of Mars (kg)
AU = 149597870000 # 1 astronomical unit (m)
day = 86400 # seconds in a day
mars_year = 687*day # 1 martian year (s)

# Function for solve differential equations
def dzdtLF(zLF, tLF):
    # x,y positions of Earth, Sun, Mars
    xe = zLF[0]
    ye = zLF[1]
    xs = zLF[2]
    ys = zLF[3]
    xm = zLF[4]
    ym = zLF[5]
    # x,y velocities of Earth, Sun, Mars
    vxe = zLF[6]
    vye = zLF[7]
    vxs = zLF[8]
    vys = zLF[9]
    vxm = zLF[10]
    vym = zLF[11]
    # x y accelarations (velocity derivative) of Earth, Sun, Mars
    # derived from Newton's Law of Gravity
    axe = -G * ((m_sun*(xe-xs))/(((xe-xs)**2+(ye-ys)**2)**1.5) + (m_mars*(xe-xm))/(((xe-xm)**2+(ye-ym)**2)**1.5))
    aye = -G * ((m_sun*(ye-ys))/(((xe-xs)**2+(ye-ys)**2)**1.5) + (m_mars*(ye-ym))/(((xe-xm)**2+(ye-ym)**2)**1.5))
    axs = -G * ((m_earth*(xs-xe))/(((xs-xe)**2+(ys-ye)**2)**1.5) + (m_mars*(xs-xm))/(((xs-xm)**2+(ys-ym)**2)**1.5))
    ays = -G * ((m_earth*(ys-ye))/(((xs-xe)**2+(ys-ye)**2)**1.5) + (m_mars*(ys-ym))/(((xs-xm)**2+(ys-ym)**2)**1.5))
    axm = -G * ((m_sun*(xm-xs))/(((xm-xs)**2+(ym-ys)**2)**1.5) + (m_earth*(xm-xe))/(((xm-xe)**2+(ym-ye)**2)**1.5))
    aym = -G * ((m_sun*(ym-ys))/(((xm-xs)**2+(ym-ys)**2)**1.5) + (m_earth*(ym-ye))/(((xm-xe)**2+(ym-ye)**2)**1.5))
    # return derivatives of rxe, rye, rxs, rys, rxm, rym, vxe, vye, vxs, vys, vxm, vym respectively
    return [vxe, vye, vxs, vys, vxm, vym, axe, aye, axs, ays, axm, aym]




#find some initials contions when sun on mars opposition
d_es = 1.0123*AU # distance from Earth to Sun on Mars Opposition (astronomical unit)
d_ms = 1.5224*AU # distance from Mars to Sun on Mars Opposition (astronomical unit)
v_es = 30000 # average speed of Earth around Sun (m/s)
v_ss = 0 # average speed of Sun around Sun (m/s)
v_ms = (86871) / 3.6 # average speed of Mars around sun (m/s)
theta = 331 * np.pi/180 # angle formed with horizon when centre of Sun is at origin (radians)
z0LF = [d_es*np.cos(theta),d_es*np.sin(theta),0,0,d_ms*np.cos(theta),d_ms*np.sin(theta),v_es*np.sin(2*np.pi-theta),v_es*np.cos(2*np.pi-theta),0,0,v_ms*np.sin(2*np.pi-theta),v_ms*np.cos(2*np.pi-theta)]
# setting up time frame for differential equation

t0 = 0.0 # my origin of time
tmax= 2*mars_year # my final time
steps= 400 # number of time step
tLF = np.linspace(t0, tmax, steps+1) # time array



zM = odeint(dzdtLF, z0LF, tLF)


# Animation part

nframe = steps #  number of frames
fig, ax = plt.subplots() # plot for animation
# x y boundaries
xmin, xmax, ymin, ymax = -2*AU, 2*AU, -2*AU, 2*AU
# boundaries of plot
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)

# x y positions of Earth, Sun, Mars
# Earth
xe = zM[:,0]
ye = zM[:,1]
# Sun
xs = zM[:,2]
ys = zM[:,3]
# mars
xm = zM[:,4]
ym = zM[:,5]

#  plotting of Earth, Sun, Mars
ime = ax.scatter(xe[0],ye[0], s=300, c='g') # Earth
ims = ax.scatter(xs[0],ys[0], s=1200, c='orange') # Sun
imm = ax.scatter(xm[0],ym[0], s=450, c='r') # Mars
imep= ax.plot(zM[:,0],zM[:,1])#EARTH PATH
immp= ax.plot(zM[:,4],zM[:,5])#MARS PATH

def update_position(num):
    global xe,ye,xs,ys,xm,ym, new_xe, new_ye, new_xs, new_ys, new_xm, new_ym
    for i in range(num) :
        print(num)
        new_xe = xe[i]

        new_ye = ye[i]

        new_xs = xs[i]

        new_ys = ys[i]

        new_xm = xm[i]

        new_ym = ym[i]



    data_e = np.stack((new_xe,new_ye),axis=-1)

    data_s = np.stack((new_xs,new_ys),axis=-1)

    data_m = np.stack((new_xm,new_ym),axis=-1)


    ime.set_offsets(data_e)
    ims.set_offsets(data_s)
    imm.set_offsets(data_m)

update_position(nframe)

imani = animation.FuncAnimation(fig,update_position, nframe,interval=1,repeat=False)
imani.save('solarsystem.mp4')

#retrograde.pdf part

#plot
#distance between sun and earth 
a1=np.sqrt((xe-xs)**2+(ye-ys)**2)
#distance between sun and mars
b1=np.sqrt((xs-xm)**2+(ys-ym)**2)
#disntance between earth and mars
c1=np.sqrt((xe-xm)**2+(ye-ym)**2)
#angle by cos teorem 3 sides known
angle=np.pi-np.arccos((c1**2+a1**2-b1**2)/(2*c1*a1)) 
plt.plot(tLF ,angle)
plt.xlabel('time')
plt.ylabel('angle(radians)')
plt.show()
