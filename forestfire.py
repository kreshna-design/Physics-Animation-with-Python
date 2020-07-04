import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

p = 0.01 #An empty space fills with a tree with probability p
f = 0.001 #A tree ignites with probability f even if no neighbor is burning
initial_trees = 0.55

tree=0 #indicate trees as one 
space=1 #indicate trees as 0
burning=-1 #indicate tress as )
a=np.zeros(shape=(100,100))
#defined initial jungle 
def initialise():
  for x in range(0,100):
     for y in range(0,100):
       a[x,y]= (tree if random.random() <= initial_trees else space)
  return a 


#this function returns to new arry with the changing conditions    
def gnew(grid):
          newgrid=np.zeros(shape=(100,100))
          for x in range(0,100):
              for y in range(0,100):
                  if grid[x,y]==-1:
                      newgrid[x,y]=1
                  elif grid[x,y]==1:
                      newgrid[x,y]= 0 if random.random()<=p else 1
                  elif grid[x,y]==0 and x!=99 and y!=99 and x-1>-1 and y-1>-1:
                       newgrid[x,y]= -1 if grid[x-1,y-1]==-1 or grid[x-1,y]==-1 or grid[x-1,y+1]==-1 or grid[x,y-1]==-1 or grid[x,y+1]==-1 or grid[x+1,y-1]==-1 or grid[x+1,y]==-1 or grid[x+1,y+1]==-1 or random.random()<= f else 0
          return newgrid

#below cods helps to make animation
fig = plt.figure()
ims = []
q=initialise()
ims.append((plt.pcolormesh(q),))
#creates and array with the 300 diffrenet plots
for add in np.arange(300):
 q=gnew(q)
 ims.append((plt.pcolormesh(q),))
imani = animation.ArtistAnimation(fig, ims, interval=301,
repeat=False)
imani.save('forestfire.mp4')
plt.show()
