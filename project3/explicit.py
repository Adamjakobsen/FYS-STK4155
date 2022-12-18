import numpy as np
import plot_nn as nn
import matplotlib.pyplot as plt

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
        if (self.dt/(self.h**2) <= 0.5) :
            print("Stability criteria met")
        else:
            print("Stability criteria not met, setting largest possible dt")
            self.dt = 0.5 * self.h**2
        
        # useful variables
        self.Nt = int(1/self.dt) + 1
        self.Nx = int(1/self.h) + 1

        # creating the storage in which to have the evolution
        self.storage = np.zeros((self.Nt, self.Nx))
    
    def __call__(self):
        self.storage[0] = self.initialise()

        for i in range(self.Nt-1):
            self.storage[i+1] = self.evolve(self.storage[i])

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
        
        nn.turn_the_lights_down_low()   # set_theme
        
        # creating the array of t=0
        x = np.linspace(0, 1, self.Nx)
        t = np.linspace(0, 1, self.Nt)
        temp_t = np.linspace(0, 1, 50)
        time = [int(self.Nt*temp_t[1]), int(self.Nt*temp_t[10]), int(self.Nt*temp_t[20]), int(self.Nt*temp_t[35])]

        figure, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 10))
        # figure.tight_layout()
        
        for i in range(4):
            if i < 2:
                col = i
                row = 0
            else:
                col = i-2
                row = 1

            hd = np.linspace(0, 1, 500)
            # plot
            ax[row, col].plot(x, self.storage[time[i],:], '-', lw=2, label="FTCS")
            ax[row, col].plot(hd, self.analytic(hd, t[time[i]]), "--", lw=2, label="Analytic")
            ax[row, col].text(0.3, 0.5, "MSE = %.2g" %self.mse(self.storage[time[i], :], self.analytic(x, t[time[i]])))
            ax[row, col].set_ylim((-0.01, 1.05))
            ax[row, col].set_xlim((-0.01, 1.01))
            ax[row, col].set_title("t=%.3f" %(t[time[i]]))
        ax[0, 1].legend()
        figure.supxlabel("x")
        figure.supylabel("f(x, t)")
        figure.savefig("./figs/compare_euler_analytic_%.2f.pdf" %self.h)
        plt.close(figure)

        fig = plt.figure(figsize=(12, 11))
        mse_list = []

        for i in range(t.size):
            mse_list.append(self.mse(self.storage[i, :], self.analytic(x, t[i])))
        
        plt.plot(t, mse_list, lw=2)
        plt.xlim((-0.01, 1.01))
        plt.ylim((1e-12, 1e-4))
        plt.xlabel("t")
        plt.ylabel("MSE")
        plt.yscale("log")
        fig.savefig("./figs/mse_euler_analytic_%.2f.pdf" %self.h)
        plt.close(fig)

        # plot close up at temp_t[45]
        fig = plt.figure(figsize=(12, 10))
        plt.plot(x, self.storage[time[-1], :], '-', lw=2, label="NN")
        plt.plot(np.linspace(0, 1, 500), self.analytic(np.linspace(0, 1, 500), t[time[-1]]), "--", lw=2, label="Analytic")
        plt.xlim((-0.01, 1.01))
        plt.xlabel("x")
        plt.ylabel("f(x, t=%.2f)" %t[time[-1]])
        fig.text(0.3, 0.5, "MSE = %.2g" %self.mse(self.storage[time[-1], :], self.analytic(x, t[time[-1]])))
        plt.legend()
        fig.savefig("./figs/ecplicit_zoom_%.2f.pdf" %self.h)
        plt.close(fig)

        return [t, mse_list]
    
    def analytic(self, x, t):
        return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)
    
    def mse(self, a , b):
        return np.mean((a-b)**2)