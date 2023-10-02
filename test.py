import unittest
import numpy as np
from network import network
from activations import tanh, sigmoid, linear
from optimizers import Adam
from layer import layer_layout

class TestYourModel(unittest.TestCase):

    def test_model_training(self):
        # Generate random data for testing
        x = np.random.randn(5, 10)
        y = np.random.randn(5, 1)

        # Create the model
        model = network()
        model.create_model([
            layer_layout(10, 10),
            tanh(),
            layer_layout(10, 1),
            linear()
        ], l_r=0.005, optimizer="Adam")

        # Train the model
        model.train(x, y, epochs=1000)

        # Make predictions
        predictions = model(x)

        # Check if the predictions are of the expected shape
        self.assertEqual(predictions.shape, y.shape)

        # Check if the values of predictions and y are close
        self.assertTrue(np.allclose(predictions, y, rtol=1e-3, atol=1e-3))

if __name__ == '__main__':
    unittest.main()
