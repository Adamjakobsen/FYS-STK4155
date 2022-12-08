import NN_PDE as nn
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_nn():
    n_hidden = 100
    
    x0 = 0
    x1 = 1
    x  = np.linspace(x0, x1, 20)
    t  = np.linspace(x0, x1, 20)

    

    #Define architecture
    layers = [  {"nodes":n_hidden,"activation":"tanh"},
                {"nodes":n_hidden,"activation":"sigmoid"}
                
                ]
    #Build model
    network = nn.NN(layers,x,t)
    model = network.model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    epochs = 3000

    network.train(model, epochs, optimizer)

    num_points = 41
    X_test, T_test = tf.meshgrid(
        tf.linspace(0, 1, num_points), tf.linspace(
            0, 1, num_points)
    )
    x_test, t_test = tf.reshape(X_test, [-1, 1]), tf.reshape(T_test, [-1,1])

    
    g_nn = tf.reshape(network.g(model, x_test, t_test), [num_points, num_points])

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
    time = [0, 0.2, 0.5, 0.8]
    xt = np.linspace(0, 1, num_points)

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
        ax[row, col].plot(xt, g_nn[int(num_points*time[i]),:], '-', lw=2, label="NN")
        ax[row, col].plot(hd, analytic(hd, time[i]), "--", lw=2, label="Analytic")
        ax[row, col].text(0.5, 0.5, "MSE = %.2g" %mse(g_nn[int(num_points*time[i]),:], analytic(xt[int(num_points * time[i])], time[i])))
        ax[row, col].set_ylim((-0.01, 1.05))
        ax[row, col].set_xlim((-0.01, 1.01))
        ax[row, col].set_xlabel("t")
        ax[row, col].set_ylabel("x")
    ax[0, 1].legend()
    plt.savefig("./figs/compare_nn_analytic.pdf")

    fig = plt.figure(figsize=(12, 10))
    mse_list = []

    for i in range(num_points):
        mse_list.append(mse(g_nn[i,:], analytic(xt[i], xt[i])))
    
    plt.plot(xt, mse_list, lw=2)
    plt.xlim((-0.01, 1.01))
    plt.xlabel("t")
    plt.ylabel("MSE")
    plt.yscale("log")
    fig.savefig("./figs/mse_nn_analytic.pdf")

    return [xt, mse_list]

def analytic(x, t):
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)

def mse(a , b):
    return np.mean((a-b)**2)