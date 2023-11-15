import unittest
import numpy as np
from DeepLearningNumpy.network import network
from DeepLearningNumpy.activations import tanh, sigmoid, linear
from DeepLearningNumpy.optimizers import Adam
from DeepLearningNumpy.layer import layer_layout

class TestYourModel(unittest.TestCase):
    def test_model_creation_1(self):
        structure = [10, 10, 1]
        model = network(structure, "relu", "MSE", learning_rate=0.001, optimizer="Adam")

    def test_model_creation_2(self):
        # Create the model
        model = network()
        model.create_model([
            layer_layout(10, 10),
            tanh(),
            layer_layout(10, 1),
            linear()
        ], l_r=0.005, optimizer="Adam")

    def test_model_prediction(self):
        # Generate random data for testing
        output_size = 3
        batch_size = 10

        x = np.random.randn(batch_size, 10)

        # Create the model
        model = network()
        model.create_model([
            layer_layout(10, 10),
            tanh(),
            layer_layout(10, output_size),
            linear()
        ], l_r=0.005, optimizer="Adam")

        predictions = model(x)
        self.assertEqual(predictions.shape, (batch_size, output_size))

    def test_training(self):
        model = network()
        model.create_model([
            layer_layout(10, 10),
            tanh(),
            layer_layout(10, 1),
            linear()
        ], l_r=0.005, optimizer="Adam")
        # Train the model
        x = np.random.randn(10, 20)
        y = np.random.randn(10, 1)

        model.train(x, y, epochs=5000, batch_size=1, shuffle=True)

        # Make predictions
        predictions = model(x)

        # Check if the values of predictions and y are close
        self.assertTrue(np.allclose(predictions, y, rtol=0.1, atol=0.1))

if __name__ == '__main__':
    unittest.main()
