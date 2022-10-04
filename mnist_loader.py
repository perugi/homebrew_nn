"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data, based on the implementation 
by mnielsen:
https://github.com/mnielsen/neural-networks-and-deep-learning

"""

#### Libraries
# Standard library
import pickle

# Third-party libraries
import numpy as np


def load_data():
    """Return the MNIST data as a tuple containing the training data
    and the test data from the data, stored on the disk (downloaded using
    the mnist_download.download function).

    ``training_data`` is a list containing 60,000 2-tuples ``(x, y)``.
    ``x`` is a 784-dimensional numpy.ndarray containing the input image.
    ``y`` is a 10-dimensional numpy.ndarray representing the unit vector
    corresponding to the correct digit for ``x``.

    ``test_data`` is a list containing 10,000 2-tuples ``(x, y)``.
    ``x`` is a 784-dimensional numpy.ndarry containing the input image.
    ``y`` is the corresponding classification, i.e., the digit values
    (integers) corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the test data.  These formats turn out to
    be the most convenient for use in our neural network code.
    """
    f = open("./data/mnist.pickle", "rb")
    training_data, test_data = pickle.load(f)
    f.close()

    training_data = [
        (input, vectorized_result(label)) for (input, label) in training_data
    ]
    return (training_data, test_data)


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
