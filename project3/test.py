from explicit import *
import matplotlib.pyplot as plt

exp = Explicit(0.1, 0.005)
res = exp()
exp.compare_analytic()

exp2 = Explicit(0.01, 0.00005)
res = exp2()
exp2.compare_analytic()
