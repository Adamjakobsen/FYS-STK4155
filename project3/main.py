import explicit as ex
import matplotlib.pyplot as plt
from plot_nn import plot_nn

exp = ex.Explicit(0.1, 0.005)
exp()
explicit = exp.compare_analytic() # [t, mse_list]

nn = plot_nn()


# exp2 = Explicit(0.01, 0.00005)
# res = exp2()
# exp2.compare_analytic()
