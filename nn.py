import random
from socket import TIPC_DEST_DROPPABLE

import numpy as np


def sigmoid(z, der=False):
    if der:
        return np.exp(-z) / (1 + np.exp(-z))
    else:
        return 1 / (1 + np.exp(-z))


def softmax(z, der=False):
    # Shift the z values so the highest is 0 (to avoid calculating very large numbers).
    z -= np.max(z)

    if der:
        return np.exp(z) / np.sum(np.exp(z)) * (1 - np.exp(z) / np.sum(np.exp(z)))
    else:
        return np.exp(z) / np.sum(np.exp(z))


class Network:
    def __init__(self, layers):
        """Initialize the neural network, based on the layers array, passed to it.
        The length of the array represents the number of layers (including the input and output layers.
        The elements of the array represent the number of neurons in the individual layers.
        The weights and biases are initialized to random values from the standard normal distribution (stdev=1, mean=0)"""

        self.num_layers = len(layers)
        self.layers = layers
        self.weights = []
        self.biases = []
        for layer_no in range(1, self.num_layers):
            self.weights.append(np.random.randn(layers[layer_no], layers[layer_no - 1]))
            self.biases.append(
                np.random.randn(
                    layers[layer_no],
                )
            )

        for weight in self.weights:
            print(f"Weights: {weight}\n{weight.shape}")
        for bias in self.biases:
            print(f"Biases: {bias}\n{bias.shape}")

    def forward(self, x):
        """Compute the output of the network, based on an input x. The function goes through
        all the layers and uses the weights and biases of each layer to compute the output of a layer
        until the final layer is reached. The sigmoid is used as the activation function."""

        for layer_weights, layer_biases in zip(self.weights, self.biases):
            x = sigmoid(np.dot(layer_weights, x) + layer_biases)
        return x

    def SGD(
        self, training_data, epochs, mini_batch_size, learning_rate, test_data=None
    ):
        """Perform stochastic gradient descent by taking the training data and then for each
        learning epoch, shuffle it, separate into mini batches and for each mini batch call the
        update_mini_batch function (which updates the weights based on the learning rate).
        If test data is passed to the function, print the accuracy of the nn on the test data
        at the end of each epoch.

        Training data and test data are list of tuples, with the input and label values. """

        for epoch in range(epochs):
            print(f"----- Training epoch {epoch}/{epochs} -----")

            # Shuffle and create mini batches.
            random.shuffle(training_data)
            mini_batches = [
                training_data[i : i + mini_batch_size]
                for i in range(0, len(training_data), mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            if test_data:
                print(f"Accuracy: {self.evaluate(test_data) / len(test_data)}%")

    def update_mini_batch(self, mini_batch, eta):
        pass

    def backprop(self, x, y):
        pass

    def evaluate(self, test_data):
        """Return the number of correct outputs from the neural network, based on test_data,
        a list of tuples, containing the inputs with matchind labels."""
        
        count_correct = 0
        for (input, label) in test_data:
            if np.argmax(self.forward(input)) == label:
                count_correct += 1

        return count_correct

    def cost_derivative(self, output_activations, y):
        pass


nn = Network([2, 4, 1])
print(nn.forward(np.array([1, 2])))
