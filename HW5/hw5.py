import numpy as np, time, matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
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
def DLARound(n_walkers, gridsize):
    '''
    Perform DLA with walkers at random points
    around the cluster.
    '''
    start = time.time()

    # Create the grid
    grid = np.zeros((gridsize,gridsize))
    middle = int(gridsize/2)
    grid[middle,middle] = 1

    for w in range(n_walkers):
        # Define start radius.
        r_start = 5 + maxExtent(grid)

        # If the cluster is too big, kill the process.
        # If this step is skipped, weird artifacts may appear.
        if r_start-5 >= gridsize/2:
            print('Structure reached maximum size.')
            break

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

            # If walker approaches the bounds, kill it and send another.
            if x >= gridsize-2 or y >= gridsize-2 or x <= 1 or y <= 1:
                w += -1
                break

            #print(x,y)
            # If the walker is bordering the structure, bond it.
            neighbors = grid[y+1,x] + grid[y-1,x] + grid[y,x+1] + grid[y,x-1]
            #print(neighbors)
            if not neighbors == 0.0:
                grid[y,x] = 1
                break
            
            p_current[0], p_current[1] = x, y

    
    end = time.time()
    print('Finished in ', round(end-start, 2), ' seconds')
    return grid
def DLALinear(n_walkers, gridsize):
    '''
    Perform DLA with walkers at random points
    around the cluster.
    '''
    start = time.time()

    # Create the grid
    grid = np.zeros((gridsize,gridsize))
    middle = int(gridsize/2)
    grid[middle,middle] = 1

    for w in range(n_walkers):
        # Define start radius.
        r_start = 5 + maxExtent(grid)

        # If the cluster is too big, kill the process.
        # If this step is skipped, weird artifacts may appear.
        if r_start-5 >= gridsize/2:
            print('Structure reached maximum size.')
            break
        
        # Define start position as point along y axis.
        y = int(round(middle + r_start))

        # Define x start position somewhere on that line 
        x = np.random.randint(middle-r_start, middle+r_start)
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

            # If walker approaches the bounds, kill it and send another.
            if x >= gridsize-2 or y >= gridsize-2 or x <= 1 or y <= 1:
                w += -1
                break

            #print(x,y)
            # If the walker is bordering the structure, bond it.
            neighbors = grid[y+1,x] + grid[y-1,x] + grid[y,x+1] + grid[y,x-1]
            #print(neighbors)
            if not neighbors == 0.0:
                grid[y,x] = 1
                break
            
            p_current[0], p_current[1] = x, y

    
    end = time.time()
    print('Finished in ', round(end-start, 2), ' seconds')
    return grid
def seedWalk(steps, gridsize):
    grid = np.zeros((gridsize,gridsize))
    middle = int(gridsize/2)
    grid[middle,middle] = 1

    x, y = middle, middle

    for i in range(steps):
        step = randint(0,4)
        if step == 0:
            x += 1
        elif step == 1:
            y += 1
        elif step == 2:
            x += -1
        else:
            y += -1
        grid[x%gridsize,y%gridsize] += 1
    
    return grid
def dimension(aggregator, gridsize):
    '''
    measures the average dimensionality of
    a DLA grown cluster.
    '''

    # Grow 3 large clusters.
    grid1 = aggregator(gridsize**2, gridsize)
    grid2 = aggregator(gridsize**2, gridsize)
    grid3 = aggregator(gridsize**2, gridsize)
    grid4 = aggregator(gridsize**2, gridsize)

    middle = int(gridsize/2)
    mass = np.zeros(middle)
    radius = np.zeros(middle)
    for r in range(middle):
        radius[r] = r
        for y in range(int(round(middle-r)),int(round(middle+r))):
            low_bound =  int(round(-np.sqrt(r**2 - (y-middle)**2)+middle))
            high_bound = int(round( np.sqrt(r**2 - (y-middle)**2)+middle))
            for x in range(low_bound,high_bound):
                mass[r] += (grid1[y,x] + grid2[y,x] + grid3[y,x]  + grid4[y,x])/4


    return mass, radius

# Minimum gridsize is 3.
gridsize = 121
grid = DLALinear(gridsize**2, gridsize)
#grid = DLALinear(gridsize**2, gridsize)
#grid = seedWalk(10000000, gridsize)
#mass, radius = dimension(DLALinear, gridsize)

'''
# Perform a linear regression on the log values.
mass   = np.log10(mass)
radius = np.log10(radius)
slope, intercept, r_value, p_value, std_err = stats.linregress(radius[1:int(gridsize/4)],mass[1:int(gridsize/4)])
x = np.linspace(radius[1], radius[int(gridsize/4)], 100)
y = slope*x + intercept
# Plot the mass/radius relation.
plt.plot(radius, mass)
plt.plot(radius, mass, '.', label='All Points')
plt.plot(radius[1:int(gridsize/4)], mass[1:int(gridsize/4)], '.', label='Points Used For Fit')
plt.plot(x, y, label='d=%s$\pm%s$'%(round(slope, 2), round(std_err, 2)))
plt.xlabel('log(radius)', fontsize=18)
plt.ylabel('log(mass)', fontsize=18)
plt.legend(fontsize=14)
plt.savefig('dimensionality_linear.pdf', bbox_inches='tight')
'''

# Plot the cluster.
print(int(np.sum(grid, axis=None)), ' cluster sites')
plt.figure(figsize=(7,7))
plt.pcolormesh(grid + DLALinear(gridsize**2, gridsize) + DLALinear(gridsize**2, gridsize) + DLALinear(gridsize**2, gridsize), cmap='Greys')
plt.xticks([])
plt.yticks([])
plt.savefig('%sx%s_linear_walk.pdf'%(gridsize,gridsize), bbox_inches='tight')

plt.show()
