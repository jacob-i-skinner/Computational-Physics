import matplotlib
import numpy as np, time, matplotlib.pyplot as plt
from matplotlib import rcParams
import time
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

norm = np.linalg.norm
randint = np.random.randint
array = np.array

def maxExtent(grid):
    '''
    Calculate the maximum extent of growth
    present in the cluster.

    Parameters
    ----------
    grid : numpy array (n, n)
        The cluster.
    
    Returns
    -------
    r_max : int
        The maximum extent of cluster growth
    '''
    
    # Find index of central element.
    middle = int(grid.shape[0]/2)

    # Pick elements with nonzero values.
    max_y = np.asarray(np.nonzero(grid.sum(axis=0))[0])

    # Find the distances of those elements.
    max_y = np.abs(middle*np.ones(max_y.shape)-max_y)

    # Grab the largest value.
    max_y = np.sort(max_y)[-1]

    # Repeat this process for the x values.
    max_x = np.asarray(np.nonzero(grid.sum(axis=1))[0])
    max_x = np.abs(middle*np.ones(max_x.shape)-max_x)
    max_x = np.sort(max_x)[-1]

    return round(np.sqrt(max_y**2 + max_x**2))
def DLA_round(n_walkers, gridsize):
    '''
    Perform DLA with walkers at random points
    around the cluster.
    '''
    start = time.time()

    # Create the grid
    grid = np.zeros((gridsize,gridsize))
    middle = int(round(gridsize/2))
    grid[middle,middle] = 1

    for w in range(n_walkers):
        # Define start radius.
        r_start = 5 + maxExtent(grid)

        # If the cluster is too big, kill the process.
        # If this step is skipped, weird artifacts may appear.
        #if r_start-5 >= gridsize/2:
        #    print('Structure reached maximum size.')
        #    break

        # Define start angle.
        angle = 2*np.pi*np.random.rand()
        
        # Define start position.
        x, y = int(round(r_start*np.cos(angle))) + middle, int(round(r_start*np.sin(angle))) + middle
        p_start = array([x, y])
        p_current = p_start

        # While the walker hasn't drifted too far, perform the walk.
        while norm(p_current-p_start) < 1.5*r_start:
            step = randint(0,4)
            if step == 0:
                x += 1
            elif step == 1:
                y += 1
            elif step == 2:
                x += -1
            else:
                y += -1

            # If walker approaches the bounds, kill it.
            if x >= gridsize-2 or y >= gridsize-2 or x == 1 or y == 1:
                w += -1
                break

            #print(x,y)
            # If the walker is bordering the structure, bond it.
            neighbors = grid[y+1,x] + grid[y-1,x] + grid[y,x+1] + grid[y,x-1]
            #print(neighbors)
            if not neighbors == 0.0:
                grid[y,x] = 1
                break
            
            p_current = array([x, y])
    
    end = time.time()
    print('Finished in ', round(end-start, 2), ' seconds')
    return grid

grid = DLA_round(100000, 41)

plt.figure(figsize=(5,5))
plt.pcolormesh(grid, cmap='Greys')

plt.show()