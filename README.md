# Deep Learning Module with Numpy

This Python module provides a lightweight deep learning framework implemented using only NumPy. It includes support for various activation functions, loss functions, and optimizers.<br>

All files in the master branch.

## Requirements
python (3.11.0 version tested)<br>
numpy (1.24.2 version tested)

To install in your environment run the command:
```
pip install .\dist\DeepLearningNumpy-0.0.1.tar.gz
```

## Usage

### Dense Layer
The layer_layout class contains the neurons and bias of the network in a matrix form and their respectives optimizers.
The layer_layout requires the number of inputs from the previous layer and the number of outputs to the next layer

the learning rate and the optimizer can be provided if necessary
ex : 
```python
from DeepLearningNumpy.layer import layer_layout
dense_1 = layer_layout(32, 32)
```

### Activation Layers
The module includes the following activation functions:
Tanh, Relu, Sigmoid, Softmax

ex :
```python
from DeepLearningNumpy.activations import tanh, sigmoid, linear, Relu, softmax
x = np.random.randn(5, 10)
activ_layer = tanh()
print(activ_layer(x))
```
all activations layer have a forward method for activating the logits and a backward method for computing the gradient

### optimizers
the optimizers class includes support for :
SGD(stochaistic gradient descent), RMsprop(Root Mean squared propagation) and Adam(Adaptive Moment Estimation)

for a specific layer, one optimzer object handle the weights's step size and another one handle the bias's step sizes

the optimizer requires the shape of the variables to train and a learning rate, those values can be directly provided if necessary
```python
from DeepLearningNumpy.optimizers import Adam
optimizer = Adam((10 ,5))
step_size = optimizer.find_step_size(gradient, learning_rate)

# gradient descent
weights -= step_size
```

### model
The network class handles the creation and management of fully connected layers.
the model creation handles the setup of the optimizers inside the layers objects.

you can create a model in these two ways :
```python
import numpy as np
from DeepLearningNumpy.activations import tanh, linear, layer_layout
from DeepLearningNumpy.layer import layer_layout
from DeepLearningNumpy.network import network

# Define the network structure
structure = [10, 10, 1]

from DeepLearningNumpy.network import network

# Method 1: Define the network structure
structure = [10, 10, 1]
model = network(neurons=structure, activation="tanh", loss_func="MSE", learning_rate=0.005, optimizer="SGD")

# Method 2: Create a model using layer_layout instances
model = network()
model.create_model([
    layer_layout(10, 10),
    tanh(),
    layer_layout(10, 1),
    tanh()
], l_r=0.005, optimizer="Adam")
```
the input layer is created at the first feed_forward method or to prevent that behavior you can set
```python
model.first_feed = True
```
You can train the model using the training method 
ex :
```python
def deriv_mse(y_pred, y_true) :
    return 2(y_pred - y_true)
def mse(y_pred, y_true) :
    return (y_pred - y_true)**2

model.loss_functions = (mse, deriv_mse)

model.train(x, y, epochs=100, batch_size=16, shuffle=True)

loss = model.error
print(loss)
```

you can also direcly backprograte the gradient into the network using the adjust method
but the shape of gradient should match the shape of the last output
```python
gradient = np.random.randn(10, 5)
model.adjust(gradient, average=True)
```

### Loss functions
the loss functions file contains a series of helper function to compute the loss and loss derivative of some popular loss function
ex :
```python
from DeepLearningNumpy.loss_functions import MSE, derivative_MSE
from DeepLearningNumpy.network import network
import numpy as np

x = np.random.randn(5, 10)
y = np.random.randn(5, 1)

epochs = 100

structure = [10, 10, 1]
model = network(structure, "relu")

for _ in range(epochs) :
    prediction = model(x)
    loss = MSE(prediction, y)
    gradient = derivative_MSE(prediction, y)

    model.adjust(gradient, average=True)
```

## Documentations
Additional implementations details are provided in the code documentations (comments)

