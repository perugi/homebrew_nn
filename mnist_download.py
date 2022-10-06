import pickle

import numpy as np
from keras.datasets import mnist


def download():
    """Download the MNIST datase and store it to disk."""
    (training_inputs, training_labels), (test_inputs, test_labels) = mnist.load_data()

    # Scale the images to range 0-1
    training_inputs = training_inputs.astype("float32") / 255
    test_inputs = test_inputs.astype("float32") / 255

    training_data = reshape(training_inputs, training_labels)
    test_data = reshape(test_inputs, test_labels)

    f = open("./data/mnist.pickle", "wb")
    pickle.dump((training_data, test_data), f)
    f.close()


def reshape(inputs, labels):
    """Reshape the incoming data, structured as (N, pixel_x, pixel_y) and (N,) np nd-arrays
    into a list of tuples, the first tuple being a flattened image and the second the label value."""
    data = []
    for input, label in zip(inputs, labels):
        data.append(
            (
                input.reshape(input.shape[0] * inputs.shape[1], 1),
                label,
            )
        )
    return data
