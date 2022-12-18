# FYS-STK 4155/3155: Project 3

Hello and welcome to our repository.
We give a short introduction to our files and how to run the programs to reproduce our results presented in the report.

You need to have `TensorFlow` installed for these programs to run. You'll also need general packages such as `seaborn`, `numpy` and `matplotlib`.

## `pde.py`
This file produces the results for the PDE part of the project. It produces quite a few figures which are sent to the `./figs` folder.
Within this file there are plentiful of variables which you can change to produce other interesting results if you'd like.

This file is run by typing in the terminal in the location of the file:

`python3 pde.py`

## `NN_eigenvalues.py`
This file produces the results for the eigenvalue problem part of the project. It produces the necessary figures and sends them to the `./figs` folder.
You can change the seed number if you want different results, though seednumber 5 is the one we have used.

This file is run by typing in the terminal in the location of the file:

`python3 NN_eigenvalues.py`

### Other programs

`NN_PDE.py` contains the code for solving the diffusion equation using a neural network. You are not supposed to run this program though you can.

`explicit.py` contains the code for solving the one dimensional diffusion equation using the FTCS method (which is an explicit method). You are not supposed to run this one either.

`plot_nn.py` is a helping file for plotting the result of the neural network for solving the diffusion equation. You are not supposed to run this one either.