import random

import numpy as np


class Sigmoid:
    @staticmethod
    def activation(z, der=False):
        """Return the sigmoid or sigmoid derivative of value z, based on the der value."""
        if der:
            return Sigmoid.activation(z) * (1 - Sigmoid.activation(z))
        else:
            return 1.0 / (1.0 + np.exp(-z))


class Softmax:
    @staticmethod
    def activation(z, der=False):
        """Return the softmax or softmax derivative of value z, based on the der value."""
        # Shift the z values so the highest is 0 (to avoid calculating very large numbers).
        z -= np.max(z)

        if der:
            return np.exp(z) / np.sum(np.exp(z)) * (1 - np.exp(z) / np.sum(np.exp(z)))
        else:
            return np.exp(z) / np.sum(np.exp(z))


class QuadraticCost:
    @staticmethod
    def error(a, y, z):
        """Return the error delta from the output layer."""
        return np.multiply((a - y), Sigmoid.activation(z, der=True))


class CrossEntropyCost:
    @staticmethod
    def error(a, y, z):
        """Return the error delta from the output layer."""
        return a - y


class LogLikelihoodCost:
    @staticmethod
    def error(a, y, z):
        """Return the error delta from the output layer."""
        return a - y


class Network:
    def __init__(self, layers, cost_fn=QuadraticCost, output_fn=Sigmoid):
        """Initialize the neural network, based on the layers array, passed to it.
        The length of the array represents the number of layers (including the input and output layers.
        The elements of the array represent the number of neurons in the individual layers.
        The weights are initialized to random values with mean = 0 and stdev=1/sqrt(no_neurons)
        The biases are initialized to random values from the standard normal distribution (stdev=1, mean=0)"""

        self.cost_fn = cost_fn
        self.output_fn = output_fn
        self.layers = layers
        self.weights = []
        self.biases = []
        self.max_test_accuracy = 0

        for i in range(1, len(self.layers)):
            # self.weights.append(np.random.randn(layers[i], layers[i - 1]) / np.sqrt[layers[i - 1]])
            self.weights.append(np.random.randn(layers[i], layers[i - 1]))
            self.biases.append(np.random.randn(layers[i], 1))

    def forward(self, x):
        """Compute the output of the network, based on an input x. The function goes through
        all the layers and uses the weights and biases of each layer to compute the output of a layer
        until the final layer is reached. The sigmoid is used as the activation function."""

        # Use the input as the activation of the first layer
        a = x

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, a) + b
            if i < len(self.weights) - 1:
                # Up to the output layer, use the sigmoid activation fn.
                a = Sigmoid.activation(z)
            else:
                # For the output layer, use the defined activation fn.
                a = self.output_fn.activation(z)

        return a

    def SGD(
        self,
        training_data,
        epochs,
        mini_batch_size,
        learning_rate,
        test_data=None,
        reg=""
        lmbda=0,
    ):
        """Perform stochastic gradient descent by taking the training data and then for each
        learning epoch, shuffle it, separate into mini batches and for each mini batch call the
        update_mini_batch function (which updates the weights based on the learning rate).
        If test data is passed to the function, print the accuracy of the nn on the test data
        at the end of each epoch.

        Training data and test data are list of tuples, with the input and label values."""

        for epoch in range(epochs):
            print(f" Training epoch {epoch+1} ".center(30, "-"))

            # Shuffle and create mini batches
            random.shuffle(training_data)
            for i in range(0, len(training_data), mini_batch_size):
                mini_batch = training_data[i : i + mini_batch_size]
                self.update_mini_batch(
                    mini_batch, learning_rate, lmbda, len(training_data)
                )

            if test_data:
                test_accuracy = self.evaluate(test_data) / len(test_data) * 100
                if test_accuracy > self.max_test_accuracy:
                    self.max_test_accuracy = test_accuracy
                print(f"Test Accuracy: {test_accuracy}%")

        print(f"NN training finished. Max test accuracy: {self.max_test_accuracy}%")

    def update_mini_batch(self, mini_batch, learning_rate, lmbda, len_training):
        """Update the weights and biases, based on a single mini batch.
        The errors are calculated using the backprop algorigthm, summed and averaged.
        The weights and biases are updated using a learning rate, passed to the function."""

        errors_weights = [np.zeros(w.shape) for w in self.weights]
        errors_biases = [np.zeros(b.shape) for b in self.biases]

        # Use the backprop algorithm and sum the weights and biases from all the samples
        # in the mini batch.
        for x, y in mini_batch:
            sample_errors_weights, sample_errors_biases = self.backprop(x, y)
            errors_weights = [
                ew + sew for ew, sew in zip(errors_weights, sample_errors_weights)
            ]
            errors_biases = [
                eb + seb for eb, seb in zip(errors_biases, sample_errors_biases)
            ]

        # Average the sum by dividing by the number of samples and update the existing
        # weights and biases by subtracting the error, multiplied by the learning rate.
        # If lambda is non-zero, L2 regularization is active.
        self.weights = [
            (1 - learning_rate * lmbda / len_training) * w
            - (learning_rate / len(mini_batch)) * ew
            for w, ew in zip(self.weights, errors_weights)
        ]
        self.biases = [
            b - (learning_rate / len(mini_batch)) * eb
            for b, eb in zip(self.biases, errors_biases)
        ]

    def backprop(self, x, y):
        """Perform the backpropagation algorithm, by taking a single input, feeding it through
        the network, calculating the output error and propagating it through all the layers.
        return the errors of weights and biases for each layer, as a list of np arrays."""

        errors_weights = [np.zeros(w.shape) for w in self.weights]
        errors_biases = [np.zeros(b.shape) for b in self.biases]

        # Move forward through the network, storing the weighted inputs (z) activations (a) of each neuron.
        weighted_inputs = []
        activations = [x]

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, activations[-1]) + b
            if i < len(self.weights) - 1:
                # Up to the output layer, use the sigmoid activation fn.
                a = Sigmoid.activation(z)
            else:
                # For the output layer, use the defined activation fn.
                a = self.output_fn.activation(z)

            weighted_inputs.append(z)
            activations.append(a)

        # Calculate the output error, giving us also the errors of weights and biases for the output layer.
        output_error = self.cost_fn.error(activations[-1], y, weighted_inputs[-1])

        errors_weights[-1] = np.dot(output_error, activations[-2].transpose())
        errors_biases[-1] = output_error

        # Backpropagate the output error, calculating the errors of weights and biases for each layer.
        layer_error = output_error
        for layer_no in range(2, len(self.layers)):
            layer_error = np.multiply(
                np.dot(self.weights[-layer_no + 1].transpose(), layer_error),
                Sigmoid.activation(weighted_inputs[-layer_no], der=True),
            )

            errors_weights[-layer_no] = np.dot(
                layer_error, activations[-layer_no - 1].transpose()
            )
            errors_biases[-layer_no] = layer_error

        return errors_weights, errors_biases

    def evaluate(self, test_data):
        """Return the number of correct outputs from the neural network, based on test_data,
        a list of tuples, containing the inputs with matchind labels."""

        count_correct = 0
        for (x, y) in test_data:
            if np.argmax(self.forward(x)) == y:
                count_correct += 1

        return count_correct

    def cost_derivative(self, output_activations, y):
        """Return the vector or partial derivatives of the quadratic cost function."""
        return output_activations - y
