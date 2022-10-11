import mnist_loader
import nn
import numpy as np

training_data, test_data = mnist_loader.load_data()

# print(type(training_data))
# print(len(training_data))
# print(type(training_data[0]))
# print(len(training_data[0]))
# print(training_data[0][0].shape)
# print(training_data[0][0])
# print(training_data[0][1])

# for i in range(1, 4):
#     print("")
#     print(f"##### NN no. {i} #####")
#     print("")
#     net = nn.Network([784, 30, 10])
#     net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
# print(training_data[0][0].shape)

print("Training with Quadratic CF")
for i in range(1, 4):
    print("")
    print(f" NN no. {i} ".center(30, "#"))
    print("")
    net = nn.Network([784, 30, 10], nn.QuadraticCost)
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

print("Training with Cross Entropy CF")
for i in range(1, 4):
    print("")
    print(f"##### NN no. {i} #####")
    print("")
    net = nn.Network([784, 30, 10], nn.CrossEntropyCost)
    net.SGD(training_data, 30, 10, 0.5, test_data=test_data)
