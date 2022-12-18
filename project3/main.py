import explicit as ex
import matplotlib.pyplot as plt
import plot_nn as pls
import time

h = [0.1, 0.01]

pls.turn_the_lights_down_low()

n_x = int(1/0.1)
n_t = int(1/(0.5 * (0.1**2)))

n_epochs = [3000, 5000, 7000]
learning = [0.01] # [0.5, 0.1, 0.05, 0.01, 0.005]
points = [30, 50]

networks_models = []
social_network = []

for ep in n_epochs:
    for lr in learning:
        start = time.process_time()
        temp_list = pls.solve(ep, lr, n_x, n_t)
        end = time.process_time()
        print("Train model with %d epochs took %.3f seconds" %(ep, (end-start)))
        networks_models.append(temp_list)
        social_network.append(pls.plot_nn(temp_list[0], temp_list[1], ep, lr, n_x, n_t))   # [t, mse_list]

figure, ax = plt.subplots(1, 1, figsize=(12, 10))

for i in range(len(n_epochs)):
    ax.plot(social_network[i][0][:], social_network[i][1][:], lw=2, label="NN %.1g epochs" %(n_epochs[i]))

for dx in h:
    start = time.process_time()
    exp = ex.Explicit(dx, 1)
    exp()
    end = time.process_time()
    print("Explicit solution with %.3f as dx took %.3f seconds" %(dx, (end-start)))
    explicit = exp.compare_analytic() # [t, mse_list]
    ax.plot(explicit[0], explicit[1], lw=2, label="FTCS dx=%.2f" %dx)


ax.set_xlim((-0.01, 1.01))
ax.set_ylim((1e-12, 1e-4))
ax.set_xlabel("t")
ax.set_ylabel("MSE")
ax.set_yscale("log")
ax.legend()
figure.savefig("./figs/mse_compare_epochs.pdf")
del ax
plt.close(figure)

# testing a model for different resolutions

temp_list = networks_models[1]   # for 5000 epochs
mse_list = social_network[1]

figure, ax = plt.subplots(1, 1, figsize=(12, 10))

for n in [30, 50, 70]:
    start = time.process_time()
    _t, _mse = pls.test(temp_list[0], temp_list[1], n, n)
    end = time.process_time()
    print("Testing the model with %d points took %.3f seconds" %(n, (end-start)))
    ax.plot(_t, _mse, lw=2, label="NN dx=dt=%.3f" %(1/n))

ax.plot(mse_list[0][:], mse_list[1][:], lw=2, label="NN criteria")

ax.set_xlim((-0.01, 1.01))
ax.set_ylim((1e-12, 1e-4))
ax.set_xlabel("t")
ax.set_ylabel("MSE")
ax.set_yscale("log")
ax.legend()
figure.savefig("./figs/mse_compare_test.pdf")
del ax
plt.close(figure)

'''
Train model with 3000 epochs took 885.496 seconds
Train model with 5000 epochs took 1167.319 seconds
Train model with 7000 epochs took 1604.898 seconds
Stability criteria not met, setting largest possible dt
Explicit solution with 0.100 as dx took 0.003 seconds
Stability criteria not met, setting largest possible dt
Explicit solution with 0.010 as dx took 2.331 seconds
Testing the model with 30 points took 2.038 seconds
Testing the model with 50 points took 2.297 seconds
Testing the model with 70 points took 2.217 seconds

Train model with 3000 epochs took 708.615 seconds
Train model with 5000 epochs took 1133.958 seconds
Train model with 7000 epochs took 1473.614 seconds
Stability criteria not met, setting largest possible dt
Explicit solution with 0.100 as dx took 0.003 seconds
Stability criteria not met, setting largest possible dt
Explicit solution with 0.010 as dx took 2.620 seconds
Testing the model with 30 points took 2.093 seconds
Testing the model with 50 points took 2.219 seconds
Testing the model with 70 points took 2.163 seconds

Train model with 3000 epochs took 631.080 seconds
Train model with 5000 epochs took 1052.703 seconds
Train model with 7000 epochs took 1561.141 seconds
Stability criteria not met, setting largest possible dt
Explicit solution with 0.100 as dx took 0.003 seconds
Stability criteria not met, setting largest possible dt
Explicit solution with 0.010 as dx took 2.876 seconds
Testing the model with 30 points took 2.273 seconds
Testing the model with 50 points took 2.485 seconds
Testing the model with 70 points took 2.134 seconds

'''