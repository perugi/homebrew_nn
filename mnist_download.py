import pickle

import numpy as np
from keras.datasets import mnist


def download():
    """Download the MNIST datase and store it to disk."""
    (training_inputs, training_labels), (
        validation_inputs,
        validation_labels,
    ) = mnist.load_data()

    # Scale the images to range 0-1
    training_inputs = training_inputs.astype("float32") / 255
    validation_inputs = validation_inputs.astype("float32") / 255

    training_data = reshape(training_inputs, training_labels)
    validation_data = reshape(validation_inputs, validation_labels)

    # split the training data into training (50k) and test (10k)
    test_data = training_data[-10000:]
    training_data = training_data[:-10000]

    print(f"training data: {len(training_data)}")
    print(f"test data: {len(test_data)}")
    print(f"validation data: {len(validation_data)}")

    f = open("./data/mnist.pickle", "wb")
    pickle.dump((training_data, test_data, validation_data), f)
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
