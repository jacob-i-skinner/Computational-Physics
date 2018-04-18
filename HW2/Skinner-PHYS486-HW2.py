import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

# Define input variables and constants.
g = np.array([0,-9.81]) #m/s^2

# Initial velocity of objects 1 & 2.
v0 = np.array([0,0]) #m/s
v1 = np.array([10,0])

# Initial position of the objects
h0 = 3.16557766e3
p = np.array([0,h0])

# These functions perform the actual calculations.
def pathFinder(position, velocity, feather_factor, dt):
    '''
    Simulate the path of an object with a given initial position and veloctiy.
    Run until object collides with the ground. Return the time at which the object
    collides with the ground, and a list of timestamped coordinates, giving
    the path the object took through the air. With drag calculations.
    '''
    
    norm = np.linalg.norm
    old_position = position
    old_velocity = velocity
    t=0
    path = np.array([[t, old_position[0], old_position[1]]])

    while 1:
        new_position = old_position + old_velocity*dt
        # This loop shrinks dt just before collision
        # with the ground, unless dt is already very small.
        while new_position[1] < 0 and dt > 1e-9:
            dt = dt/2
            new_position = old_position + old_velocity*dt
        if new_position[1] < 0 and dt <= 1e-9:
            break
        
        if old_velocity[0] == 0 and old_velocity[1] == 0:
            new_velocity = old_velocity + g*dt
        else:
        #                             air density * c/m            v**2                     -v hat
            new_velocity = old_velocity + (61.25e-3*(feather_factor)*(norm(old_velocity)**2)*(-old_velocity/norm(old_velocity)) + g)*dt

        old_position = new_position
        old_velocity = new_velocity

        t += dt
        path = np.append(path, [[t, old_position[0], old_position[1]]], axis=0)

    return path
def noDrag(position, velocity, dt):
    '''
    Simulate the path of an object with a given initial position and veloctiy.
    Run until object collides with the ground. Return the time at which the object
    collides with the ground, and a list of timestamped coordinates, giving
    the path the object took through the air. without drag calculations.
    '''
    def norm(v):
        if v.all() == 0:
            return 1
        return np.linalg.norm(v)

    old_position = position
    old_velocity = velocity
    t=0
    path = np.array([[t, old_position[0], old_position[1]]])
    while 1:
        new_position = old_position + old_velocity*dt
        # This loop shrinks dt just before collision
        # with the ground, unless dt is already very small.
        while new_position[1] < 0 and dt > 1e-9:
            dt = dt/2
            new_position = old_position + old_velocity*dt
        if new_position[1] < 0 and dt <= 1e-9:
            break
        
        new_velocity = old_velocity +  g*dt

        old_position = new_position
        old_velocity = new_velocity

        t += dt
        path = np.append(path, [[t, old_position[0], old_position[1]]], axis=0)

    return path
def diff_with_h():
    # This function finds the difference in landing
    # times as a function of height.
    t_diff = np.empty(1000)
    for i in range(0,1000):
        path2 = np.transpose(pathFinder(np.array([0,i]), v2, 5e-3, 0.01))
        path1 = np.transpose(pathFinder(np.array([0,i]), v1, 5e-3, 0.01))
        t_diff[i] = path2[0][-1] - path1[0][-1]
    return t_diff
# t_diff = diff_with_h()
def diff_with_v():
    # This function find the difference in landing
    # times as a function of launch velocity.
    t_diff = np.empty(1000)
    for i in range(0,1000):
        path2 = np.transpose(pathFinder(p, np.array([i,0]), 5e-3, 0.01))
        path1 = np.transpose(pathFinder(p, np.array([0,0]), 5e-3, 0.01))
        t_diff[i] = path2[0][-1] - path1[0][-1]
    return t_diff
#t_diff = diff_with_v()
def diff_with_ff():
    # This function find the difference in landing
    # times as a function of the "feather factor" C/m.
    t_diff = np.empty(1000)
    for i in range(0,1000):
        path2 = np.transpose(pathFinder(p, np.array([0,0]), i/1000, 0.01))
        path1 = np.transpose(pathFinder(p, np.array([0,0]), 5e-3, 0.01))
        t_diff[i] = path2[0][-1] - path1[0][-1]
    return t_diff
#t_diff = diff_with_ff()

path1 = np.transpose(pathFinder(p, v1, 0.01, 0.1))
plt.semilogy(path1[0], path1[2])
del path1
path2 = np.transpose(pathFinder(p, v1, 0.01, 0.01))
plt.semilogy(path2[0], path2[2])
del path2
path3 = np.transpose(pathFinder(p, v1, 0.01, 0.001))
plt.semilogy(path3[0], path3[2])
del path3
path4 = np.transpose(pathFinder(p, v1, 0.01, 0.0001))
plt.semilogy(path4[0], path4[2])
del path4
plt.ylim(0.01,40)
plt.xlim(33.9,33.94)
plt.savefig('differing dt.pdf', bbox_inches='tight')
plt.show()


# Assorted plot code (very boring)
'''
plt.plot(np.linspace(0,0.999,num=1000), t_diff)
plt.xlim(0,1)
plt.ylim(0,t_diff[-1])
plt.ylabel('Difference in Landing Time (s)', fontsize=20)
plt.xlabel('C/m (units?)', fontsize=20)
#plt.savefig('time and ff.pdf', bbox_inches='tight')
plt.show()

path0 = np.transpose(noDrag(p, np.array([0,0]), 0.01))
path1 = np.transpose(pathFinder(p, np.array([0,0]), 5e-3, 0.01))
#path2 = np.transpose(pathFinder(p, np.array([0.00001,0]), 5e-3, 0.01))
plt.plot(path0[0], path0[2])
plt.plot(path1[0], path1[2])
#plt.plot(path2[0], path2[2])
#plt.xlim(25,26)
#plt.ylim(0,50)
plt.show()
plt.clf()

plt.plot(path0[1], path0[2])
plt.plot(path1[1], path1[2])
#plt.plot(path2[1], path2[2])
plt.show()
'''