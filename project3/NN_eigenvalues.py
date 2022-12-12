
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.keras.utils.set_random_seed(42)
tf.keras.backend.set_floatx("float64")


class NN_eigen():
    """
    Neural Network class for solving PDE"s. Specificly, the loss function is designed for solving
    the diffusion equation.

    """

    def __init__(self, layers, A, t,max=True):
        """
        Initialize the neural network

        arguments:

            layers(dict): dictionary of the layers in the neural network.
            A(numpy array): the A matrix
            t(numpy array): the t values
            max(boolean): if true, the max eigenvalue is computed, else the min eigenvalue is computed

                    
        """
        
        self.n=tf.shape(A)[0]
        layer_list = []
        #layer_list.append(tf.keras.layers.Dense(self.n,input_shape=(1,), name="input"))
        for layer in layers:
            layer_list.append(tf.keras.layers.Dense(
                layer["nodes"], activation=layer["activation"]))

        layer_list.append(tf.keras.layers.Dense(self.n, name="output"))
        self.model = tf.keras.Sequential(layer_list)


        self.A = tf.cast(tf.convert_to_tensor(A), tf.float64)

        self.t = tf.cast(tf.convert_to_tensor(t), tf.float64)


    @tf.function
    def g_trial(self,x,t):
        """
        The trial function for the eigenvalue problem

        arguments:
            x(tensor): the x values
            t(tensor): the t values

        returns:
            g(tensor): the trial function
        """
        g = tf.exp(-t)*x + (1-tf.exp(-t))*self.model(t)
        return g

    @tf.function
    def loss(self):
        """
        The loss function for the eigenvalue problem

        returns:
            loss(tensor): the loss function
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.t)
            g=self.g_trial(self.x,self.t)
        
        dg_dt = tape.gradient(g, self.t)
        x=tf.transpose(g)
        dx_dt=tf.transpose(dg_dt)

        xx=tf.reduce_sum(tf.multiply(x,x))
        Ax=tf.matmul(self.A,x)
        xAx=tf.reduce_sum(tf.multiply(x,Ax))

        I= tf.eye(tf.shape(self.A)[0], dtype=tf.float64) 

        f=tf.matmul((xx*self.A + (1-xAx)*I),x)

        loss= tf.reduce_mean(tf.square(dx_dt-f+x))
         
        return loss

    @tf.function
    def compute_eig(self,x,t):
        """
        Compute the eigenvalue

        arguments:
            x(tensor): the x values
            t(tensor): the t values

        returns:
            eigenvale(tensor): the eigenvalue
            eigenvector(tensor): the eigenvector
        """
        
        v=self.g_trial(x,t)
        v=tf.transpose(v)
        vv=tf.matmul(tf.transpose(v),v)
        Av=tf.matmul(self.A,v)
        vAv=tf.matmul(tf.transpose(v),Av)

        
        eigenvalue=tf.divide(vAv,vv)

        #Normalsed eigenvector
        eigenvec=tf.divide(v,tf.sqrt(vv))
         
        return eigenvalue,eigenvec
    
    @tf.function
    def grad(self):
        """
        Compute the gradient of the loss function

        arguments:
            t(tensor): the t values

        returns:
            loss(tensor): the loss function
            gradients(tensor): the gradients of the loss function
        """
        with tf.GradientTape() as tape:
            loss = self.loss()
        gradients = tape.gradient(loss, self.model.trainable_variables)
        return loss, gradients


    def set_data(self, x, t):
        
        
        self.x0 = x 
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        t = tf.convert_to_tensor(t, dtype=tf.float64)
       
        x, t = tf.meshgrid(x, t)
        x, t = tf.reshape(x, [-1, 1]), tf.reshape(t, [-1, 1])
        self.x, self.t = x, t

    
    def predict(self, epochs,x,t,lr):
        """
        Fit the neural network

        arguments:
            epochs(int): the number of epochs
            x(numpy array): the x values
            t(numpy array): the t values
            lr(float): the learning rate
            

        returns:
            eigenvals(numpy array): the eigenvalues as a function of epochs
            eigenvecs(numpy array): the eigenvectors as a function of epochs
            
        """
        
        #Set the data
        self.set_data(x,t)
        
        #Arrays to store predictions as function of epochs
        
        eigenvals = np.zeros(epochs)
        eigenvecs = np.zeros([epochs, self.n])
        t_max = tf.reshape(self.t[-1], [-1,1])

        #Fit the model and compute the eigenvalues and eigenvectors
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        for epoch in range(epochs):
            
            loss, gradients = self.grad()
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            eigenval, eigenvec  = self.compute_eig(self.x0, t_max)
            eigenvals[epoch] = eigenval.numpy()
            eigenvecs[epoch,:] = eigenvec.numpy().T[:]
            
        
        return eigenvals, eigenvecs


if __name__ == "__main__":
    seed = 5
    np.random.seed(seed)
    tf.random.set_seed(seed)

    #Define the matrix A
    n=6
    Q = np.random.rand(n, n)
    A = 1/2*(Q + Q.T)

    #Define the x and t values
    t_max=1e3
    x = np.random.normal(0, 1, n)
    t = np.linspace(0, t_max, 100)

    #Define the neural network architecture
    n_hidden = 1000
    layers = [{"nodes": n_hidden, "activation": "relu"}

            ]

    ########## Max eigenvalue ##########

    NN = NN_eigen(layers, A, t)

    model = NN.model
    
    epochs = 5000 #Number of epochs
    epochs_array = np.arange(epochs)

    
    lr = 1e-3 #Learning rate
    #Fit the model
    eigvals, eigvecs = NN.predict(epochs, x, t,lr)

    #Compute the eigenvalues and eigenvectors using numpy
    numpy_eigvals, numpy_eigvecs = np.linalg.eig(A)
    max_idx = np.argmax(numpy_eigvals)
    max_eigval = numpy_eigvals[max_idx]
    max_eigvec=numpy_eigvecs.T[max_idx]



    #Plot the eigenvalues
    
    plt.plot(epochs_array, eigvals, label='NN')
    plt.hlines(numpy_eigvals, 0, epochs, colors='r', linestyles='dashed', label='numpy')
    plt.legend()
    plt.show()

    #Plot the eigenvector components
    for i in range(n):
        plt.plot(epochs_array, eigvecs[:,i],label= f'NN component {i}')
        plt.hlines(max_eigvec[i], 0, epochs, colors='r', linestyles='dashed')

    plt.ylim(min(max_eigvec)-0.1, max(max_eigvec)+0.1) 
    plt.show()

    #plot mse eigenvalue
    mse_eigval = np.square(eigvals - max_eigval)
    plt.plot(epochs_array, mse_eigval, label='MSE')
    plt.legend()
    plt.show()




    ######### MIN EIGENVALUE #########
    
    NN = NN_eigen(layers, A, t, max=False)

    model = NN.model
    
    epochs = 5000 #Number of epochs
    epochs_array = np.arange(epochs)

    lr = 1e-3 #Learning rate

    #Fit the model
    eigvals, eigvecs = NN.predict(epochs, x, t,lr)

    #Compute the eigenvalues and eigenvectors using numpy
    numpy_eigvals, numpy_eigvecs = np.linalg.eig(A)
    min_idx = np.argmin(numpy_eigvals)
    min_eigval = numpy_eigvals[min_idx]
    min_eigvec=numpy_eigvecs.T[min_idx]



    #Plot the eigenvalues
    plt.plot(epochs_array, eigvals)
    plt.hlines(numpy_eigvals, 0, epochs, colors='r', linestyles='dashed')
    plt.show()
    #Plot the eigenvector components
    for i in range(n):
        plt.plot(epochs_array, eigvecs[:,i])
        plt.hlines(min_eigvec[i], 0, epochs, colors='r', linestyles='dashed')

    plt.ylim(min(min_eigvec)-0.1, max(min_eigvec)+0.1) 
    plt.show()

    mse_eigval = np.square(eigvals - min_eigval)
    plt.plot(epochs_array, mse_eigval,label='MSE')
    plt.legend()
    plt.show()




        

    
