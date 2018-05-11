import matplotlib
import numpy as np, time, matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

'''
Gauss-Seidel relaxation method for a square potential.
'''

# Relaxation with interpolated initial conditions.
def relax_inter(gridsize, duration):
    '''
    Find the solution to Laplace's equation for a specific set of initial conditions.
    Grid will be a 3D array, composed of 2D slices which represent iterations of the array.
    '''

    # Setup the grid.    
    # Populate all with random values.
    grid = np.random.random((gridsize, gridsize))

    # Define the middle of the grid.
    middle = int(gridsize/2)

    # Interpolate over the grid.
    R = np.sqrt(2)*(middle-1)
    for i in range(gridsize):
        for j in range(gridsize):
            distance = np.sqrt((middle-i)**2 + (middle-j)**2)
            grid[i][j] = 1-distance/R


    # Set top to zero.
    grid[0] = [0 for i in grid[0]]

    # Set ends of middle rows to zero.
    for i in range(1,gridsize-1):
        grid[i][0], grid[i][-1] = 0, 0
    
    # Set bottom row to zero.
    grid[-1] = [0 for i in grid[-1]]

    # Set center 5x5 grid to 1.
    for i in range(middle-2, middle+3):
        for j in range(middle-2, middle+3):
            grid[i][j] = 1

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

    # Relax!
    for k in range(duration):

        # save an old copy of the grid before updating it
        old_grid = np.copy(grid)
        
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
        #grid = unfold(grid)


        # Break the loop if the average change of the changeable element is <= 1e-9.
        avg_diff = abs(np.sum(grid - old_grid)/(gridsize**2 - 4*gridsize - 21))
        if avg_diff <= 1e-9:
            break
        

    end = time.time()
    elapsed = end - start
    print('Finished in ', round(elapsed, 1), ' sec.')
    
    # Return everything which was filled by the relaxation.
    return unfold(grid), iterations, avg_diff

# Relaxation wiht constant initial conditions.
def relax_const(gridsize, duration, weight):
    '''
    Find the solution to Laplace's equation for a specific set of initial conditions.
    Grid will be a 3D array, composed of 2D slices which represent iterations of the array.
    '''

    # Setup the grid.    
    # Populate all with random values.
    #grid = np.random.random((gridsize, gridsize))
    #grid = np.zeros((gridsize, gridsize))
    grid = weight*np.ones((gridsize, gridsize))

    # Define the middle of the grid.
    middle = int(gridsize/2)

    # (Roughly) Interpolate over the grid.
    # Determine diagonal distance of grid
    #R = np.sqrt(2)*(middle-1)
    #for i in range(gridsize):
    #    for j in range(gridsize):
    #        distance = np.sqrt((middle-i)**2 + (middle-j)**2)
    #        grid[i][j] = 1-distance/R


    # Set top to zero.
    grid[0] = [0 for i in grid[0]]

    # Set ends of middle rows to zero.
    for i in range(1,gridsize-1):
        grid[i][0], grid[i][-1] = 0, 0
    
    # Set bottom row to zero.
    grid[-1] = [0 for i in grid[-1]]

    # Set center 5x5 grid to 1.
    for i in range(middle-2, middle+3):
        for j in range(middle-2, middle+3):
            grid[i][j] = 1

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

    # Relax!
    for k in range(duration):

        # save an old copy of the grid before updating it
        old_grid = np.copy(grid)
        
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
        #grid = unfold(grid)


        # Break the loop if the average change of the changeable element is <= 1e-9.
        avg_diff = abs(np.sum(grid - old_grid)/(gridsize**2 - 4*gridsize - 21))
        if avg_diff <= 1e-9:
            break
        

    end = time.time()
    elapsed = end - start
    print('Finished in ', round(elapsed, 1), ' sec.')
    
    # Return everything which was filled by the relaxation.
    return unfold(grid), iterations, avg_diff

# Relaxation with random initial conditions.
def relax_rand(gridsize, duration, weight):
    '''
    Find the solution to Laplace's equation for a specific set of initial conditions.
    Grid will be a 3D array, composed of 2D slices which represent iterations of the array.
    '''

    # Setup the grid.    
    # Populate all with random values.
    grid = weight*np.random.random((gridsize, gridsize))
    #grid = np.zeros((gridsize, gridsize))
    #grid = np.ones((gridsize, gridsize))

    # Define the middle of the grid.
    middle = int(gridsize/2)

    # (Roughly) Interpolate over the grid.
    # Determine diagonal distance of grid
    #R = np.sqrt(2)*(middle-1)
    #for i in range(gridsize):
    #    for j in range(gridsize):
    #        distance = np.sqrt((middle-i)**2 + (middle-j)**2)
    #        grid[i][j] = 1-distance/R


    # Set top to zero.
    grid[0] = [0 for i in grid[0]]

    # Set ends of middle rows to zero.
    for i in range(1,gridsize-1):
        grid[i][0], grid[i][-1] = 0, 0
    
    # Set bottom row to zero.
    grid[-1] = [0 for i in grid[-1]]

    # Set center 5x5 grid to 1.
    for i in range(middle-2, middle+3):
        for j in range(middle-2, middle+3):
            grid[i][j] = 1

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

    # Relax!
    for k in range(duration):

        # save an old copy of the grid before updating it
        old_grid = np.copy(grid)
        
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
        #grid = unfold(grid)


        # Break the loop if the average change of the changeable element is <= 1e-9.
        avg_diff = abs(np.sum(grid - old_grid)/(gridsize**2 - 4*gridsize - 21))
        if avg_diff <= 1e-9:
            break
        

    end = time.time()
    elapsed = end - start
    print('Finished in ', round(elapsed, 1), ' sec.')
    
    # Return everything which was filled by the relaxation.
    return unfold(grid), iterations, avg_diff

# Generate error(iterations) plot.
def iterative_error():
    # Find the error as a function of the number of iterations used.
    iterations = np.linspace(1,500,500)
    errors = [relax(25, x)[2] for x in range(1, 501)]
    plt.semilogy(iterations, errors, 'k')
    plt.xlim(1,500)
    plt.xlabel('Iterations of Relaxation', fontsize=18)
    plt.ylabel('Solution Error', fontsize=18)
    plt.savefig('iteration error.pdf', bbox_inches='tight')
    plt.show()
#iterative_error()

# Generate error(size) plot.
def spatial_error():
    # Create the first instance
    prime_grid = relax(15, 40000000)[0]
    max_size = 151
    size = np.linspace(17,max_size,67)
    
    # Calcluate the sum of the element-wise differences between the prime grid and subgrid
    errors = [np.sum(relax(x, 40000000)[0][(int(x/2)-7):(int(x/2)+8),(int(x/2)-7):(int(x/2)+8)]-prime_grid) for x in range(17,max_size,2)]
    
    plt.plot(size,errors, 'k')
    plt.xlim(17,max_size)
    plt.xlabel('Gridsize', fontsize=18)
    plt.ylabel('Solution Difference', fontsize=18)
    plt.savefig('size error.pdf', bbox_inches='tight')
    plt.show()
#spatial_error()

# Generate iterations(initial conditions) plot.
def initial_error():
    weight = np.linspace(0,1,100)

    #Perform multiple instances of random starts
    '''
    iterations = [relax_rand(25, 40000, x)[1] for x in weight]
    plt.plot(weight,iterations, 'k', alpha=0.2, label='Random Start')
    for i in range(9):
        iterations = [relax_rand(25, 40000, x)[1] for x in weight]
        plt.plot(weight,iterations, 'k', alpha=0.2)
    '''

    iterations = [relax_const(51, 40000, x)[1] for x in weight]
    plt.plot(weight, iterations, label='Uniform Start')
    #plt.plot(weight, np.ones(len(weight))*relax_inter(25, 40000)[1], '--', label='Interpolated Start (no weight)')
    #plt.legend()
    #plt.xlim(0.2,0.4)
    plt.xlabel('Weight', fontsize=18)
    plt.ylabel('Iterations Required', fontsize=18)
    #plt.savefig('iterations.pdf', bbox_inches='tight')
    plt.show()
#initial_error()


# Perform a simple relaxation with optimal initial conditions
gridsize = 51
duration = 2000
grid, iterations, error = relax_const(gridsize, duration, 0.26)

plt.figure(figsize=(5,5))
plt.title('%s Iterations'%(iterations+1))
#plt.xlabel('Precision: %s'%(error))
plt.pcolor(grid)
plt.axis('off')
#plt.ylim(gridsize, 0)
#plt.savefig('initial.pdf', bbox_inches='tight')
plt.show()
plt.clf()