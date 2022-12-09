import NN_PDE as nn
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

def turn_the_lights_down_low():
    """
    Function for setting the plotting environment equal for all plots
    """

    context = {'font.size': 20.0,
                'axes.labelsize': 20.0,
                'axes.titlesize': 20.0,
                'xtick.labelsize': 20.0,
                'ytick.labelsize': 20.0,
                'legend.fontsize': 20.0,
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
    plt.rcParams['text.usetex'] = True

def plot_nn(n_epochs, learning, n_x, n_t):
    n_hidden = 100
    
    x0 = 0
    x1 = 1
    x  = np.linspace(x0, x1, n_x)
    t  = np.linspace(x0, x1, n_t)

    turn_the_lights_down_low()  # set_theme

    #Define architecture
    layers = [  {"nodes":n_hidden,"activation":"tanh"},
                {"nodes":n_hidden,"activation":"sigmoid"}]
    #Build model
    network = nn.NN(layers,x,t)
    model = network.model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning)
    epochs = n_epochs

    network.train(model, epochs, optimizer)


    X_test, T_test = tf.meshgrid(
        tf.linspace(0, 1, n_x), tf.linspace(
            0, 1, n_t)
    )
    x_test, t_test = tf.reshape(X_test, [-1, 1]), tf.reshape(T_test, [-1,1])

    
    g_nn = tf.reshape(network.g(model, x_test, t_test), [n_t, n_x])

    # creating the array of t=0
    time = [0.01, 0.2, 0.5, 0.8]

    figure, ax = plt.subplots(2, 2, sharex=True, figsize=(13, 10))
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
        ax[row, col].plot(x, g_nn[int(n_t*time[i]),:], '-', lw=2, label="NN")
        ax[row, col].plot(hd, analytic(hd, time[i]), "--", lw=2, label="Analytic")
        ax[row, col].text(0.3, 0.5, "MSE = %.2g" %mse(g_nn[int(n_t*time[i]),:], analytic(x, time[i])))
        ax[row, col].set_ylim((-0.01, 1.05))
        ax[row, col].set_xlim((-0.01, 1.01))
        ax[row, col].set_xlabel("x")
        ax[row, col].set_ylabel("f(x, t=%.1f)" %time[i])
    ax[0, 1].legend()
    figure.savefig("./figs/compare_nn_analytic_epoch%d_dx%d_dt%d.pdf" %(n_epochs, n_x, n_t))
    del figure
    del ax

    fig = plt.figure(figsize=(13, 10))
    mse_list = []

    for i in range(len(t)):
        mse_list.append(mse(g_nn[i,:], analytic(x, t[i])))
    
    plt.plot(t, mse_list, lw=2)
    plt.xlim((-0.01, 1.01))
    plt.xlabel("t")
    plt.ylabel("MSE")
    plt.yscale("log")
    fig.savefig("./figs/mse_nn_analytic_epoch%d_dx%d_dt%d.pdf" %(n_epochs, n_x, n_t))
    del fig

    # plot close up at 0.8 t
    fig = plt.figure(figsize=(13, 10))
    plt.plot(x, g_nn[int(n_t*0.8),:], '-', lw=2, label="NN")
    plt.plot(np.linspace(0, 1, 500), analytic(np.linspace(0, 1, 500), 0.8), "--", lw=2, label="Analytic")
    plt.xlim((-0.01, 1.01))
    plt.xlabel("x")
    plt.ylabel("f(x, t=0.8)")
    fig.text(0.3, 0.5, "MSE = %.2g" %mse(g_nn[int(n_t*0.8),:], analytic(x, 0.8)))
    plt.legend()
    
    fig.savefig("./figs/nn_zoom_epoch%d_dx%d_dt%d.pdf" %(n_epochs, n_x, n_t))
    del fig

    return [t, mse_list]

def analytic(x, t):
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)

def mse(a , b):
    return np.mean((a-b)**2)