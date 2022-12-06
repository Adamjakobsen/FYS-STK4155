
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.keras.utils.set_random_seed(42)
tf.keras.backend.set_floatx("float64")




class NN():
    """
    Neural Network class for solving PDE"s. Specificly, the loss function is designed for solving
    the diffusion equation.

    """

    def __init__(self,layers,x,t):
        """
        Initialize the neural network

        arguments:

                    x(numpy array): the x values
                    t(numpy array): the t values

                    layers(dict): dictionary of the layers in the neural network.
        """
        self.x,self.t = self.get_data(x,t)

        layer_list=[]
        for layer in layers:
            layer_list.append(tf.keras.layers.Dense(layer["nodes"], activation=layer["activation"]))

        layer_list.append(tf.keras.layers.Dense(1, name="output"))
        self.model = tf.keras.Sequential(layer_list)
        

    def get_data(self,x,t):
        """
        Transform the data into tensorflow tensors

        arguments:
            x(numpy array): the x values
            t(numpy array): the t values

        returns:
            x(tensor): the x values
            t(tensor): the t values
        """
       
        X,T= tf.meshgrid(tf.cast(tf.convert_to_tensor(x), tf.float64),
                        tf.cast(tf.convert_to_tensor(t), tf.float64))

        x = tf.reshape(X,[-1,1])
        t = tf.reshape(T,[-1,1])  

        return x,t
    
    @tf.function
    def initial_condition(self,x):
        """
        Initial condition

        arguments:
            x(tensor): the x values

        """

        I=tf.sin(np.pi * x)
        return I
    
    @tf.function
    def g(self,model,x,t):
        """
        Trial function solution

        arguments:
            model(tensorflow object): the neural network model
            x(tensor): the x values
            t(tensor): the t values

        returns:
            Trial_solution(tensor): the trial solution
        """
        data_points = tf.concat([x, t], axis=1)

        trial_solution=(1 - t) * self.initial_condition(x) + x * (1 - x) * t * model(data_points)

        return trial_solution
    
    

    @tf.function
    def grad(self,model):
        """
        Compute loss and gradient of loss with respect to the model parameters
        
        arguments:
            model(tensorflow object): the neural network model


        returns:
            loss(tensor): the loss
            gradients(tensor): the gradients of the loss with respect to the model parameters
        """
        
        with tf.GradientTape() as tape:
            
            loss_value = self.loss(model)
            tape.watch(model.trainable_variables)

        return loss_value, tape.gradient(loss_value, model.trainable_variables)
    
    @tf.function
    def loss(self,model):
        """
        Loss/cost function

        arguments:
            model(tensorflow object): the neural network model


        returns:
            MSE(tensor): the mean squared error
        """
        with tf.GradientTape(persistent=True) as tape:
            #Record the operations for automatic differentiation
            tape.watch([self.x, self.t])
            with tf.GradientTape(persistent=True) as tape_2:
                tape_2.watch([self.x, self.t])

                trial = self.g(model,self.x,self.t)
            
            #Compute the derivative of the trial function
            d_g_dx = tape_2.gradient(trial, self.x)
            d_g_dt = tape_2.gradient(trial, self.t)

        d2_g_d2x = tape.gradient(d_g_dx, self.x)

        del tape_2
        del tape

        MSE = tf.reduce_mean(tf.square( d2_g_d2x - d_g_dt))

        return MSE
    
    
    def train(self,model,epochs,optimizer):
        """
        Train the model

        arguments:
            model(tensorflow object): the neural network model
            epochs(int): number of epochs to train
            optimizer(tensorflow object): the optimizer
        
        """
        for epoch in range(epochs):
            # Apply gradients in optimizer
            cost, gradients = self.grad(model)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            
            if epoch % 100 == 0:
                print("Epoch: {}, Loss: {}".format(epoch, cost))

def g_analytic(x, t):
    """
    Analytic solution
    """

    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)            

if __name__ == "__main__":
    
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
    network = NN(layers,x,t)
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

    sol_t0 = g_nn[int(num_points*0.1),:]
    plt.plot(np.linspace(0, 1, num_points), sol_t0,'-', label="NN")
    plt.plot(np.linspace(0, 1, num_points), g_analytic(np.linspace(0, 1, num_points), 0.1), label="Analytic")
    plt.legend()
    plt.savefig("sol_t0.png")
    plt.close()

    sol_t1 = g_nn[int(0.8*num_points),:]
    plt.plot(np.linspace(0, 1, num_points), sol_t1, '-', label="t=0.8")
    plt.plot(np.linspace(0, 1, num_points), g_analytic(np.linspace(0, 1, num_points), 0.8), label="Analytic")
    plt.legend()
    plt.savefig("sol_t1.png")
    plt.close()

    """ 
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(X_test, T_test, g_nn, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u")
    ax.set_title("Neural Network Solution")
    plt.savefig("Neural Network Solution.png")
    plt.show()
    """

