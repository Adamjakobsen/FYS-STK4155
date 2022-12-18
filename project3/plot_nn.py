import NN_PDE as nn
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

def turn_the_lights_down_low():
    """
    Function for setting the plotting environment equal for all plots
    """

    context = {'font.size': 25.0,
                'axes.labelsize': 25.0,
                'axes.titlesize': 25.0,
                'xtick.labelsize': 25.0,
                'ytick.labelsize': 25.0,
                'legend.fontsize': 25.0,
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

    plt.rcParams['text.usetex'] = False
    sns.set_theme(context=context, style="whitegrid", palette="colorblind", font="sans-serif", font_scale=1)

def solve(n_epochs, learning, n_x, n_t):
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

    return [network, model]



def plot_nn(network, model, n_epochs, learning, n_x, n_t):
    t = np.linspace(0, 1, n_t)
    x = np.linspace(0, 1, n_x)

    turn_the_lights_down_low()

    X_test, T_test = tf.meshgrid(
        tf.linspace(0, 1, n_x), tf.linspace(
            0, 1, n_t)
    )
    x_test, t_test = tf.reshape(X_test, [-1, 1]), tf.reshape(T_test, [-1,1])
    
    g_nn = tf.reshape(network.g(model, x_test, t_test), [n_t, n_x])

    figure, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 10))
    # figure.tight_layout()
    
    ind = [int(1*(n_t/50)), int(10*(n_t/50)), int(20*(n_t/50)), int(35*(n_t/50))]

    for i in range(4):
        if i < 2:
            col = i
            row = 0
        else:
            col = i-2
            row = 1

        hd = np.linspace(0, 1, 500)
        # plot
        ax[row, col].plot(x, g_nn[ind[i],:], '-', lw=2, label="NN")
        ax[row, col].plot(hd, analytic(hd, t[ind[i]]), "--", lw=2, label="Analytic")
        ax[row, col].text(0.3, 0.5, "MSE = %.2g" %mse(g_nn[ind[i],:], analytic(x, t[ind[i]])))
        ax[row, col].set_ylim((-0.01, 1.05))
        ax[row, col].set_xlim((-0.01, 1.01))
        ax[row, col].set_title("t=%.3f" %t[ind[i]])

    ax[0, 1].legend()
    figure.supylabel("f(x, t)")
    figure.supxlabel("x")
    figure.savefig("./figs/compare_nn_analytic_epoch%d_dx%d_dt%d.pdf" %(n_epochs, n_x, n_t))
    plt.close(figure)

    fig = plt.figure(figsize=(12, 10))
    mse_list = []

    j = 0
    for i in t:
        mse_list.append(mse(g_nn[j,:], analytic(x, i)))
        j += 1
    
    plt.plot(t, mse_list, lw=2)
    plt.xlim((-0.01, 1.01))
    plt.ylim((1e-12, 1e-4))
    plt.xlabel("t")
    plt.ylabel("MSE")
    plt.yscale("log")
    fig.savefig("./figs/mse_nn_analytic_epoch%d_dx%d_dt%d.pdf" %(n_epochs, n_x, n_t))
    plt.close(fig)

    # close up plot
    placer = int(35*(n_t/50))
    fig = plt.figure(figsize=(12, 10))
    plt.plot(x, g_nn[placer,:], '-', lw=2, label="NN")
    plt.plot(np.linspace(0, 1, 500), analytic(np.linspace(0, 1, 500), t[placer]), "--", lw=2, label="Analytic")
    plt.xlim((-0.01, 1.01))
    plt.xlabel("x")
    plt.ylabel("f(x, t=%.2f)" %t[placer])
    fig.text(0.3, 0.5, "MSE = %.2g" %mse(g_nn[placer,:], analytic(x, t[placer])))
    plt.legend()
    
    fig.savefig("./figs/nn_zoom_epoch%d_dx%d_dt%d.pdf" %(n_epochs, n_x, n_t))
    plt.close(fig)

    # plotting the learning history
    epoch, learning = network.learning_history()
    fig = plt.figure(figsize=(12, 10))
    plt.plot(epoch, learning, lw=2)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    
    fig.savefig("./figs/learning_epoch%d_dx%d_dt%d.pdf" %(n_epochs, n_x, n_t))
    del epoch, learning
    plt.close(fig)

    return [t, mse_list]

def analytic(x, t):
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)

def mse(a , b):
    return np.mean((a-b)**2)

def test(network, model, n_x_test, n_t_test):

    x = np.linspace(0, 1, n_x_test)
    t = np.linspace(0, 1, n_t_test)

    turn_the_lights_down_low()

    X_test, T_test = tf.meshgrid(
        tf.linspace(0, 1, n_x_test), tf.linspace(
            0, 1, n_t_test)
    )
    x_test, t_test = tf.reshape(X_test, [-1, 1]), tf.reshape(T_test, [-1,1])
    
    g_nn = tf.reshape(network.g(model, x_test, t_test), [n_t_test, n_x_test])

    figure, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 10))
    # figure.tight_layout()
    
    ind = [int(1*(n_t_test/50)), int(10*(n_t_test/50)), int(20*(n_t_test/50)), int(35*(n_t_test/50))]

    for i in range(4):
        if i < 2:
            col = i
            row = 0
        else:
            col = i-2
            row = 1

        hd = np.linspace(0, 1, 500)
        # plot
        ax[row, col].plot(x, g_nn[ind[i],:], '-', lw=2, label="NN")
        ax[row, col].plot(hd, analytic(hd, t[ind[i]]), "--", lw=2, label="Analytic")
        ax[row, col].text(0.3, 0.5, "MSE = %.2g" %mse(g_nn[ind[i],:], analytic(x, t[ind[i]])))
        ax[row, col].set_ylim((-0.01, 1.05))
        ax[row, col].set_xlim((-0.01, 1.01))
        ax[row, col].set_title("t=%.3f" %t[ind[i]])

    ax[0, 1].legend()
    figure.supylabel("f(x, t)")
    figure.supxlabel("x")
    figure.savefig("./figs/compare_nn_analytic_test_dx%d_dt%d.pdf" %(n_x_test, n_t_test))
    plt.close(figure)

    fig = plt.figure(figsize=(12, 10))
    mse_list = []

    j = 0
    for i in t:
        mse_list.append(mse(g_nn[j,:], analytic(x, i)))
        j += 1
    
    plt.plot(t, mse_list, lw=2)
    plt.xlim((-0.01, 1.01))
    plt.ylim((1e-12, 1e-4))
    plt.xlabel("t")
    plt.ylabel("MSE")
    plt.yscale("log")
    fig.savefig("./figs/mse_nn_analytic_test_dx%d_dt%d.pdf" %(n_x_test, n_t_test))
    plt.close(fig)

    # close up plot
    placer = int(35*(n_t_test/50))
    fig = plt.figure(figsize=(12, 10))
    plt.plot(x, g_nn[placer,:], '-', lw=2, label="NN")
    plt.plot(np.linspace(0, 1, 500), analytic(np.linspace(0, 1, 500), t[placer]), "--", lw=2, label="Analytic")
    plt.xlim((-0.01, 1.01))
    plt.xlabel("x")
    plt.ylabel("f(x, t=%.2f)" %t[placer])
    fig.text(0.3, 0.5, "MSE = %.2g" %mse(g_nn[placer,:], analytic(x, t[placer])))
    plt.legend()
    
    fig.savefig("./figs/nn_zoom_test_dx%d_dt%d.pdf" %(n_x_test, n_t_test))
    plt.close(fig)

    # plotting the learning history
    epoch, learning = network.learning_history()
    fig = plt.figure(figsize=(12, 10))
    plt.plot(epoch, learning, lw=2)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    
    fig.savefig("./figs/learning_test_dx%d_dt%d.pdf" %(n_x_test, n_t_test))
    del epoch, learning
    plt.close(fig)

    return [t, mse_list]