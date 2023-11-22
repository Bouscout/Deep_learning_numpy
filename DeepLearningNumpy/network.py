import numpy as np
import enum
import pickle
from .activations import sigmoid, linear, softmax, tanh, Relu
from .layer import layer_layout
from .optimizers import SGD, RMSprop, Adam
from .loss_functions import *
from .utils import shuffler, divide_batch


# this file will assemble all the components in order to make network easy to use

class network():
    def __init__(self, neurons:list = None, activation:str ='tanh', loss_func:str = 'MSE', *,learning_rate:float =0.01, optimizer:str="SGD") -> None:
        """
        Class to create and manage a full model with the forward, backward and training process being instantiable with simple methods\n
        
        the layers are accessible in self.layers attributes, the activation layers in self.activ_layers
        """
        self.learning_rate = learning_rate
        self.first_feed = False
        self.neurons = neurons # list of neurons at each layer

        self.learning_rate = learning_rate

        # declaring the loss function
        loss_functions_choices = {
            'MSE' : [MSE, derivative_MSE] ,
            'cross-entropy': [cross_entropy_non_one_hot, derivate_cross_entropy_non_one_hot],
        }
        self.loss_function = loss_functions_choices[loss_func]

        self.error = None # the overall error of the network

        #optimizers choises
        optimizer_choices = {
            'SGD' : SGD,
            'RMSPROP' : RMSprop,
            "ADAM" : Adam,
        }

        self.optimizer_chosen = optimizer_choices[optimizer.upper()]

        # preparing the layers
        self.layers = []
        self.activ_layers = []

        # if a quick creation layout has been provided
        if neurons :
             # choices for the activations layers
            activations_choice = {
                'sigmoid' : sigmoid,
                'linear' : linear,
                'tanh' : tanh,
                'relu' : Relu,
            }
            activ_function = activations_choice[activation] # defining global activation layer

            for num in range(len(neurons) - 1) :
                self.layers.append(layer_layout(input_size=neurons[num], output_size=neurons[num+1], optimizer=self.optimizer_chosen, l_r=learning_rate))

                # initializing the weight differently if relu
                if activation == 'relu' :
                    layer = self.layers[num]
                    layer.custom_weights_initialization()  
            
                # activation layer
                self.activ_layers.append(activ_function())

    def create_model(self, layers:list, l_r:float, optimizer:str) :
        """
        Create a model from a list of layers and activations layers\n
        All layer informations must be provided at their creation
        """
        optimizer_choices = {
            'SGD' : SGD,
            'RMSPROP' : RMSprop,
            "ADAM" : Adam,
        }

        optim = optimizer_choices[optimizer.upper()]
        self.optimizer_chosen = optim

        self.layers = []
        self.activ_layers = []
        for i in range(0, len(layers), 2) :
            layer = layers[i]
            
            layer.weights_optimizer = optim(layer.weights.shape)
            layer.bias_optimizer = optim(layer.bias.shape)
            layer.alpha = l_r

            self.layers.append(layers[i])
            self.activ_layers.append(layers[i+1])

        

    def feed(self, input: np.ndarray) -> np.ndarray:
        # checking if first layer created, if not create the neurons according to input shape
        if self.first_feed == False:
            output_size = self.layers[0].output_size
            
            self.layers[0] = layer_layout(
                input_size=input.shape[-1],
                output_size=output_size,
                optimizer=self.optimizer_chosen,
                l_r=self.learning_rate
            )

            self.first_feed = True

        output = input
        for lay, activ in zip(self.layers, self.activ_layers) :
            non_active_output = lay.forward_propagation(output)
            output = activ.forward_propagation(non_active_output)
            
        return output
    
    # function to propagate the error backwards and adjust weight and bias
    def adjust(self, output_gradient:np.ndarray, average=True) -> None:
        """Backpropagate the gradient"""
        gradient = output_gradient
        for activ, lay in zip(reversed(self.activ_layers) ,reversed(self.layers)):
            gradient_activ = activ.backward_propagation(gradient, self.learning_rate)
            gradient = lay.backward_propagation(gradient_activ, self.learning_rate, average)


    # function to train the network
    def train(self, X:np.ndarray, Y:np.ndarray, epochs:int = 5, batch_size: int = 1, shuffle:bool=False) -> None :

        loss_function, loss_function_derivative = self.loss_function

        if shuffle :
            X, Y = shuffler(X, Y) # shuffling the datapoint whiling maintaining the relationship
       

        for _ in range(epochs):
            for x_batch, y_batch in divide_batch(X, Y, batch_size) :
                output = x_batch

                output = self.feed(input=output)

                #error and backprog
                error = loss_function(output, y_batch)
                gradients = loss_function_derivative(output, y_batch)

                self.adjust(gradients)

        self.error = error

        return output

    def predict(self, input: np.ndarray) -> np.ndarray:
        output = self.feed(input=input)
        return output
    
    # overiding the call function
    def __call__(self, input:np.ndarray) -> np.ndarray:
        return self.predict(input=input)

    # get models weights and bias
    def get_params(self):
        """
        Create an array of tuple containing the weights and bias values :\n
        params = [(weights_layer_1, bias_layer_1), (weights_layer_2, bias_layer_2), ...]
        """
        parameters = [0 for _ in range(len(self.layers))]
        for index, layer in enumerate(self.layers) :
            parameters[index] = (np.copy(layer.weights), np.copy(layer.bias))

        return parameters
    
    def reset_optimizers(self):
        """Reset the inertie and momemtum of the optimizers"""
        for layer in self.layers :
            layer.weights_optimizer.reset_optimizer()
            layer.bias_optimizer.reset_optimizer()

    # set the weights and bias from given numpy arrays of correct shape
    # should be in a format with list of tuples containing weights and bias
    def set_params(self, params):
        if len(params) != len(self.layers) :
            raise ValueError('mismatch num of layers')
        
        for index, variables in enumerate(params):
            weights, bias = variables
            self.layers[index].weights = weights
            self.layers[index].bias = bias
            
        self.first_feed = True   

    # ----------------- Trained Models ---------------------


     # create a model with the received list of tuple of weights and bias as layers
    def load_model_from_paramsList(self, params:list, activations:list, torch:bool = False):
        """
        Load a model from tensorflow or pytorch using their respective function to access the model parameters.\n

        the params need to come in the form params = [weights_l1, bias_l1, weights_l2, bias_l2, ...]\n
        
        the number of activations needs to match the structure of the model \n

        the params need to be converted to numpy before being passed to this method.\n

        the model learning rate and optimizer need to be specified before calling the method
        """
        layer = []
        activ_layer = activations

        for i in range(0, len(params), 2) :
            weights = params[i]
            bias = params[i + 1]

            # pytorch inverse the order in the matrix
            if torch :
                layer_init = layer_layout(*reversed(weights.shape), optimizer=self.optimizer_chosen, l_r=self.learning_rate)
                layer_init.weights = weights.T
            else :
                layer_init = layer_layout(*weights.shape, optimizer=self.optimizer_chosen, l_r=self.learning_rate)
                layer_init.weights = weights
            
            layer_init.bias = bias.reshape(1, -1)

            layer.append(layer_init)

        self.layers = layer
        self.activ_layers = activ_layer
        self.first_feed = True


    def save_model(self, file_path:str):
        params = []
        for layer in self.layers :
            params.append(layer.weights)
            params.append(layer.bias)

        activations = [repr(activ) for activ in self.activ_layers]

        with open(file_path, "wb") as f :
            pickle.dump([params, activations], f)
        print("model saved at ", file_path)

    
    def load_model(self, file_path) :
        """
        Load the model from a file containing :\n
        >>> (list[numpyWeights, bias, weights ...], list["tanh", "tanh", "sigmoid"])
        """
        activations = {
            "sigmoid" : sigmoid,
            "tanh" : tanh,
            "relu" : Relu,
            "softmax" : softmax,
            "linear" : linear,
        }
        with open(file_path, "rb") as f :
            params_activ = pickle.load(f)
        
        self.load_model_from_paramsList(
            params_activ[0] ,# weights and bias
            [activations[x.lower()]() for x in params_activ[1]] # activations layers
        )

        print("model loaded from ", file_path)

        