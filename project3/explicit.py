import numpy as np

class Explicit():
    """
    Explicit solver for the diffusion equation using the explicit forward Euler method.

    """

    # length is 1
    def __init__(self, h, dt):
        """
        Initialise instance.

        arguments:
            h stepsize in spatial dimention
            T Total time
            dt timestep-size
        """

        # getting the parameters
        self.h = h
        self.dt = dt

        # check stability
        assert self.dt/self.h**2 <= 0.5, "Stability criterion not met"
        
        # useful variables
        self.Nt = int(1/dt)
        self.Nx = int(1/h)

        # creating the storage in which to have the evolution
        self.storage = np.zeros((self.Nt, self.Nx))
    
    def __call__(self):
        self.storage[0] = self.initialise()

        for i in range(self.Nt-1):
            self.storage[i+1] = self.evolve(self.storage[i])

        return self.storage

    def evolve(self, current):
        """
        Evolves argument array one step further in the time dimension.

        arguments:
            array containing the current state of the system.
        """
        evolved = np.zeros_like(current)
        
        for i in range(1, self.Nx-1):
            uxx = (current[i+1] - 2*current[i] + current[i-1])/self.h**2
            evolved[i] = uxx*self.dt + current[i]

        return evolved

    def initialise(self):
        """
        Creates and returns the initial state with initial condition sin(pi * x). Applying Dirichlet boundary conditions.

        returns:
            initial state.
        """
        # creating the array of t=0
        u = np.linspace(0, 1, self.Nx)
        
        # filling in the initial values
        i = 0
        for val in u:
            u[i] = np.sin(np.pi * val)
            i += 1
        
        # setting the boundary conditions
        u[0] = 0
        u[-1] = 0

        return u

    def compare_analytic(self):
        import matplotlib.pyplot as plt
        # creating the array of t=0
        x = np.linspace(0, 1, self.Nx)
        time = np.linspace(0, 1, self.Nt)

        analytic = np.zeros((self.Nt, self.Nx))

        ind = 0
        for i in time:
            analytic[ind] = np.sin(np.pi * x) * np.exp(-np.pi**2 * i)

        plt.plot(x, self.storage[int(len(x)/2)])
        plt.plot(x, analytic[int(len(x)/2)], label="analytic")
        plt.legend()
        plt.show()