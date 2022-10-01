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
            self.biases.append(np.random.randn(layers[layer_no], 1))

        for weight in self.weights:
            print(f"Weights: {weight}\n{weight.shape}")
        for bias in self.biases:
            print(f"Biases: {bias}\n{bias.shape}")

    def forward(self, x):
        for layer_weights, layer_biases in zip(self.weights, self.biases):
            x = sigmoid(np.dot(layer_weights, x) + layer_biases)
        return x

nn = Network([2, 4, 1])
