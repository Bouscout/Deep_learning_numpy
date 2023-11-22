from typing import Any
import numpy as np
from .errors import ValueOverflowingError


# this function will contain the logic for defining a layer of the network
# we won't need to represent the neuron individually, they will just be considered by their indexes in the matrices


# the layers informations needs to be provided

class layer_layout:
    def __init__(self, input_size: int, output_size: int, optimizer=None, l_r:float = 0.001):
        # we will represent the neurons in a matrix with the num of rows representing the num of neurons
        # on the next layer and the columns representing the weights for each neurons
        self.input_size = input_size # num of neurons in layer
        self.output_size = output_size # num of weights per neuron for next layer
        self.alpha = l_r # learning rate

        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(1, output_size)

        self.input = None # input received through propagation

        self.level = 0

        # initializing optimizers for each parameter
        if optimizer :
            self.weights_optimizer = optimizer(layer_shape=[self.input_size, self.output_size])
            self.bias_optimizer = optimizer(layer_shape=[1, self.output_size])

    def __call__(self, data:np.ndarray) -> np.ndarray:
        """
        perform the operations : \n
        >>> z = matmul(data, weights)
        >>> output = z + bias
        """
        return self.forward_propagation(data=data)

    def forward_propagation(self, data:np.array ) :
        # we assume input will arrive in the correct form from prv layer
        # we perform a matrix multiplication with the weights on this layer
        
        self.input = data
        # z = np.dot(self.weights, data)  #old version
        z = np.matmul(data, self.weights)
        # output = z 
        output = z + self.bias
        
        # useful for catching unstable values but can be commented out
        if np.any(np.isnan(output)) or np.any(np.isinf(output)):
           raise ValueOverflowingError('one of the value is overflowing')
       
        return output

    def backward_propagation(self, gradient, learning_rate, average):
        # we will receive a matrix of gradient corresponding to each weight
        # in each neuron, so we just perform a gradient descent to adjust each weight 
        
        # we need to find the gradient of the input which will correspond to the ouput of the previous layer
        
        # derv_input_x1 = sum(w*derv_loss for every w and derv_loss tied to x1)
        input_gradient_set = np.matmul(gradient, self.weights.T) # new version

        # weights update
        # the gradient of the weights from the derivation is found through the operation between the input and the loss gradient
        inputs = self.input[:, :, None]
        gradient_column = gradient[:, None, :]

        weight_gradients = np.matmul(inputs, gradient_column)

        if average :
            weight_gradients = np.mean(weight_gradients, axis=0)
        else :
            weight_gradients = np.sum(weight_gradients, axis=0)

        if self.weights_optimizer :
            step_size = self.weights_optimizer.find_step_size(weight_gradients, self.alpha)
            self.weights -= step_size
        else :
            # SGD
            self.weights -= (learning_rate * weight_gradients) 
        

        # the bias gradient is only the loss gradient so we sum the gradient options on the row axis
        if average :
            bias_gradient = np.mean(gradient, axis=0)
        else :
            bias_gradient = np.sum(gradient, axis=0)
            
        step_size = self.bias_optimizer.find_step_size(bias_gradient, self.alpha)
        # self.bias -= (learning_rate * bias_gradient)
        self.bias -= step_size

        return input_gradient_set
            
    def custom_weights_initialization(self, stddev=None) -> np.ndarray:
        """
        Initialize the weights using a normal distribution around 0 with a certain standard deviation
        """
        # Calculate the standard deviation for the weights
        stddev = np.sqrt(2.0 / self.input_size) if not stddev else stddev

        # Initialize the weights with random values from a Gaussian distribution
        weights = np.random.normal(0, stddev, self.weights.shape)
        self.weights = weights
        self.bias = np.random.normal(0, stddev, self.bias.shape)
    


