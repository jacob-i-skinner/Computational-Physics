import numpy as np, time, matplotlib.pyplot as plt, matplotlib.animation as anim
from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

'''
Simulation of waves on a string, with one end fixed.
The 'r' parameter, string length, and wave speed are fixed at 1.
'''

def wavy(dx, duration, profile, driver):
    '''
    Simulate a wave on a string. Returns a
    2D array with rows representing snapshots in time.

    Parameters
    ----------

    dx : float
        Spatial resolution of string.
    
    duration : float
        Time length of simulation.

    profile : string
        Defines the initial profile of the string.
        values: flat, gaussian, plucked


    driver  : string
        Defines the driving profile of the string.
        values: sine, square, fixed, free
    '''

    sin = np.sin

    A = 0.5            # Driving amplitude.
    T = 1               # Tension in string.
    u = 1               # Linear mass-density of string.
    c = np.sqrt(T/u)    # Wave speed.

    dt = dx/c           # r is constrained to 1.
    t = 0

    # Initialize the wave variable.
    wave = np.zeros((int(duration/dt), int(1/dx)))
    
    # Populate the driven end of the wave.
    if   driver == 'sine':
        for i in range(int(0.25/dt)):
            wave[i][-1] = A*sin(8*np.pi*i*dt) # i*dt = the 'real' time.
    #elif driver == 'square':
        # square pulse in here
    elif driver == 'free': # This requires a modified propagator which includes the left end in its calculations
        # Simulate the wave propagation.
        for n in range(int(duration/dt)-1):
            for i in range(int(1/dx)-1):
                wave[n+1][i] = wave[n][i-1] - wave[n-1][i] + wave[n][i+1] # This may throw an out-of-bounds error, but will hopefully wrap around.
        return wave

    # Simulate the wave propagation.
    for n in range(1, int(duration/dt)-1):
        for i in range(1, int(1/dx)-1):
            wave[n+1][i] = wave[n][i-1] - wave[n-1][i] + wave[n][i+1]

    return wave

wave = wavy(0.005, 60, 'flat', 'sine')

# Make an animation of the wave.
fig, ax = plt.subplots(figsize=(16,4))
x = np.linspace(0, 1, 1/0.005)
plt.ylim(-1, 1)
plt.xlim(1,0)
def blitter(): # Clears the frame for the animator.
    return ax.plot([np.nan] * len(x))
def animate(i): # Returns frames to the animator.
    return ax.plot(x, wave[i], 'k', lw=4)

ani = anim.FuncAnimation(fig, animate, init_func=blitter, blit=True, interval = 17)

plt.show()