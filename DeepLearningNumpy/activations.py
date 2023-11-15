import numpy as np

# this file will contain all the activation function possible that we 
# will place between the layers

class activation:
    def __init__(self) :
        self.input = None
        self.output = None

# all values between -1 and 1
class tanh(activation):
    def __call__(self, data) -> np.ndarray:
        return self.forward_propagation(data)
    
    def __repr__(self) -> str:
        return "tanh"
    
    def forward_propagation(self, data):
        self.input = data
        output = np.tanh(self.input)
        return output
    
    def backward_propagation(self, output_gradient, learning_gradient):
        drv_output = 1 - np.tanh(self.input) ** 2
        output = np.multiply(output_gradient, drv_output)
        return output
    
# all values between 0 and 1
class sigmoid(activation):
    def __call__(self, data) -> np.ndarray:
        return self.forward_propagation(data)
    
    def __repr__(self) -> str:
        return "sigmoid"

    def forward_propagation(self, data):
            self.input = data
            output = 1 / (1 + np.exp(-data))
            return output
    
    def backward_propagation(self, output_gradient, learning_gradient):
        s = 1 / (1 + np.exp(-self.input))
        drv_output =  s * (1 - s)    

        output = np.multiply(output_gradient, drv_output)
        return output

   
# return zero if value < 0 otherwise return the actual value
class Relu(activation) :
    def __call__(self, data) -> np.ndarray:
        return self.forward_propagation(data)
    
    def __repr__(self) -> str:
        return "Relu"

    def forward_propagation(self, data) :
        self.input = data
        return np.maximum(0, data)

    def backward_propagation(self, output_gradient, learning_rate) :
        drv_output = np.where(self.input > 0, 1, 0) # filter values
        output = drv_output * output_gradient
        return output
    
#TO DO
class leaking_Relu(activation):
    pass
    
# return actual value
class linear(activation):
    def __call__(self, data:np.ndarray) -> np.ndarray:
        return self.forward_propagation(data)
    
    def __repr__(self) -> str:
        return "linear"

    def forward_propagation(self, data):
        return data
    
    def backward_propagation(self, output_gradient, learning_rate):
        # just return the gradient times one
        return output_gradient



class softmax(activation):
    def __init__(self):
        super().__init__()

    def __call__(self, logits:np.ndarray) -> np.ndarray:
        """Create a probability distribution of shape (batch_size, num_logits)"""
        self.forward_propagation(logits)

    def __repr__(self) -> str:
        return "softmax"  

    def forward_propagation(self, logits:np.ndarray):
        self.input = logits

        # Compute softmax values for each batch
        exp_data = np.exp(logits)
        softmax_output = exp_data / np.sum(exp_data, axis=1, keepdims=True)

        self.output = softmax_output
        return softmax_output

    def backward_propagation(self, output_gradient, learning_rate):
        # Compute the Jacobian matrix for softmax
        identity = np.eye(self.output.shape[-1])

        # we know that the softmax derivative differs for indices i,j : 
        #dsI / dsJ = softmax(x) * (1 - softmax(x)) for i == j
        # dsI / dsJ = softmax(xI) * -softmax(xJ) for i != j
        # using the identity matrix will have the cases where i == j be represented by 1 and the rest 0

        # we expand dimensions in order to performs operations at batch level
        jacobian = self.output[:, :, None] * (identity - self.output[:, None, :])

        # Compute the gradient for each batch
        batch_gradients = np.matmul(jacobian, output_gradient[:, :, None])

        # reducing dimensions
        all_gradients = batch_gradients[:, :, 0]

        return all_gradients