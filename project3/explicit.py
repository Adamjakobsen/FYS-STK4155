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
        import seaborn as sns

        context = {'font.size': 16.0,
                    'axes.labelsize': 16.0,
                    'axes.titlesize': 16.0,
                    'xtick.labelsize': 16.0,
                    'ytick.labelsize': 16.0,
                    'legend.fontsize': 16.0,
                    'legend.title_fontsize': None,
                    'axes.linewidth': 0.8,
                    'grid.linewidth': 0.8,
                    'lines.linewidth': 1.5,
                    'lines.markersize': 6.0,
                    'patch.linewidth': 1.0,
                    'xtick.major.width': 0.8,
                    'ytick.major.width': 0.8,
                    'xtick.minor.width': 0.6,
                    'ytick.minor.width': 0.6,
                    'xtick.major.size': 3.5,
                    'ytick.major.size': 3.5,
                    'xtick.minor.size': 2.0,
                    'ytick.minor.size': 2.0}

        sns.set_theme(context=context, style="whitegrid", palette="colorblind", font="sans-serif", font_scale=1)
        # creating the array of t=0
        x = np.linspace(0, 1, self.Nx)
        t = np.linspace(0, 1, self.Nt)
        time = [0, self.Nt*0.2, self.Nt*0.5, self.Nt*0.8]

        figure, ax = plt.subplots(2, 2, sharex=True, figsize=(12, 10))
        figure.tight_layout()
        
        for i in range(4):
            if i < 2:
                col = i
                row = 0
            else:
                col = i-2
                row = 1

            hd = np.linspace(0, 1, 500)
            # plot
            ax[row, col].plot(x, self.storage[int(time[i])], '-', lw=2, label="FTCS")
            ax[row, col].plot(hd, self.analytic(hd, t[int(time[i])]), "--", lw=2, label="Analytic")
            ax[row, col].text(0.5, 0.5, "MSE = %.2g" %self.mse(self.storage[int(time[i])], self.analytic(x, t[int(time[i])])))
            ax[row, col].set_ylim((-0.01, 1.05))
            ax[row, col].set_xlim((-0.01, 1.01))
            ax[row, col].set_xlabel("t")
            ax[row, col].set_ylabel("x")
        ax[0, 1].legend()
        plt.savefig("./figs/compare_euler_analytic.pdf")

        fig = plt.figure(figsize=(12, 10))
        mse_list = []

        ind = 0
        for n in self.storage:
            mse_list.append(self.mse(n, self.analytic(n, t[ind])))
            ind += 1
        
        plt.plot(t, mse_list, lw=2)
        plt.xlim((-0.01, 1.01))
        plt.xlabel("t")
        plt.ylabel("MSE")
        plt.yscale("log")
        fig.savefig("./figs/mse_euler_analytic.pdf")

        return [t, mse_list]
    
    def analytic(self, x, t):
        return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)
    
    def mse(self, a , b):
        return np.mean((a-b)**2)