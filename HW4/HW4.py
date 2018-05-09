import matplotlib
matplotlib.use("Agg")
import numpy as np, time, matplotlib.pyplot as plt, matplotlib.animation as anim
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\squas\\Documents\\GitHub\\Computational-Physics\\HW4\\ffmpeg-20180507-29eb1c5-win64-static\\bin\\ffmpeg.exe'

'''
Simulation of waves on a string, with one end fixed, and one end driven sinusoidally.
The 'r' parameter, string length, and wave speed are fixed at 1.
'''

def relax(gridsize, duration):
    '''
    Find the solution to Laplace's equation for a specific set of initial conditions.
    Grid will be a 3D array, composed of 2D slices which represent iterations of the array.
    '''

    # Setup the grid.    
    # Populate all with random values.
    grid = np.random.random((gridsize, gridsize))
    #grid = np.zeros((gridsize, gridsize))

    # Set top to zero.
    grid[0] = [0 for i in grid[0]]

    # Set ends of middle rows to zero.
    for i in range(1,gridsize-1):
        grid[i][0], grid[i][-1] = 0, 0
    
    # Set bottom row to zero.
    grid[-1] = [0 for i in grid[-1]]

    # Set middle 5x5 grid to 1. This is dynamic based on gridsize.
    middle = int(gridsize/2)
    for i in range(middle-2, middle+3):
        for j in range(middle-2, middle+3):
            grid[i][j] = 1

    timeline = np.empty((duration, gridsize, gridsize))

    def unfold(grid):
        # 'Unfold' the solved octant.
        for i in range(2, middle+1):
            for j in range(1, i):
                grid[i][j] = grid[j][i]
        
        # 'Unfold' the (now) quadrant.
        for i in range(1, middle+1):
            for j in range(middle+1, gridsize-1):
                grid[i][j] = grid[i][gridsize-j-1]
        
        # 'Unfold' the (now) half.
        for i in range(middle+1, gridsize-1):
            for j in range(1, gridsize-1):
                grid[i][j] = grid[gridsize-i-1][j]
        return grid
    
    start = time.time()

    # Solve the problem!
    for k in range(duration):
        # Iterate over only a single octant.
        for i in range(1, middle-2):
            for j in range(i, middle+1):
                if j == i: # If element is on diagonal border.
                    grid[i][j] = (2*grid[i-1][j] + 2*grid[i][j+1])/4
                elif j == middle: # If element is on vertical edge.
                    grid[i][j] = (grid[i-1][j] + grid[i+1][j] + 2*grid[i][j-1])/4
                else: # If in the main body of the octant.
                    grid[i][j] = (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][j+1])/4
        
        # Reiterate, but backwards, to maintain (information symmetry).
        for i in range(middle-2, 1):
            for j in range(middle+1, i):
                if j == i: # If element is on diagonal border.
                    grid[i][j] = (2*grid[i-1][j] + 2*grid[i][j+1])/4
                elif j == middle: # If element is on vertical edge.
                    grid[i][j] = (grid[i-1][j] + grid[i+1][j] + 2*grid[i][j-1])/4
                else: # If in the main body of the octant.
                    grid[i][j] = (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][j+1])/4
        
        iterations = k
        timeline[k] = unfold(grid)

    end = time.time()
    elapsed = end - start
    print('Finished in ', round(elapsed, 1), ' sec.')
    
    return timeline, iterations

gridsize = 101
duration = 2500
grid, iterations = relax(gridsize, duration)
'''
plt.figure(figsize=(5,5))
plt.pcolor(grid, cmap='Greys')
plt.ylim(gridsize, 0)
plt.show()
'''



# Make an animation of the wave.
fig, ax = plt.subplots(figsize=(5,5))
plt.title('%s Iterations'%(iterations))
#x = np.linspace(0, 1, 1/dx)
#plt.ylim(-1, 1)
#plt.xlim(1,0)

def blitter(): # draws an empty frame.
    return ax.pcolor(np.nan*np.ones((gridsize, gridsize)))
def animate(i): # Returns frames to the animator.
    return ax.pcolor(grid[i], cmap='Greys')

ani = anim.FuncAnimation(fig, animate, init_func=blitter, interval = 200)
FFwriter = anim.FFMpegWriter(fps=15, bitrate=3600)
ani.save('relax.mp4', writer=FFwriter)

#plt.show()