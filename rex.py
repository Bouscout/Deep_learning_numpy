import numpy as np
from network import network
from activations import tanh, sigmoid, linear
from optimizers import Adam
from layer import layer_layout

x = np.random.randn(5, 10)
y = np.random.randn(5, 1)

model = network()

model.create_model([
    layer_layout(10, 10),
    tanh(),
    layer_layout(10, 1),
    linear()
], l_r=0.005, optimizer="Adam")

model.train(x, y, epochs=1000)

print(model(x))
print("=======")
print(y)

