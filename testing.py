import mnist_loader
import nn
import numpy as np

training_data, test_data, validation_data = mnist_loader.load_data()

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

# net = nn.Network([784, 30, 10], nn.CrossEntropyCost)
# output = net.forward(training_data[0][0])
# print(output, np.sum(output))

# net = nn.Network([784, 30, 10], nn.LogLikelihoodCost, output_fn=nn.Softmax)
# output = net.forward(training_data[0][0])
# print(output, np.sum(output))

# print("")
# print("Training with Quadratic CF")
# for i in range(1, 4):
#     print("")
#     print(f" NN no. {i} ".center(30, "#"))
#     print("")
#     net = nn.Network([784, 30, 10], nn.QuadraticCost)
#     net.SGD(training_data, 30, 10, 3.0, test_data=test_data, reg="L2", lmbda=5.0)

print("")
print("Training with Cross Entropy CF")
for i in range(1, 4):
    print("")
    print(f" NN no. {i} ".center(30, "#"))
    print("")
    net = nn.Network([784, 30, 10], nn.CrossEntropyCost)
    net.SGD(
        training_data,
        100,
        10,
        0.5,
        reg="",
        lmbda=0,
        monitor_training_cost=True,
        monitor_training_accuracy=True,
        monitor_test_cost=True,
        monitor_test_accuracy=True,
        test_data=test_data,
        no_improvement_in_n=3,
    )

# print("")
# print("Training with a Softmax output activation fn and Log-likelihood cost")
# for i in range(1, 4):
#     print("")
#     print(f" NN no. {i} ".center(30, "#"))
#     print("")
#     net = nn.Network([784, 30, 10], nn.LogLikelihoodCost, output_fn=nn.Softmax)
#     net.SGD(training_data, 30, 10, 0.5, test_data=test_data, reg="L2", lmbda=5.0)

# print("Quadratic CF  with 100 hidden neurons and L1 regularization")
# net = nn.Network([784, 100, 10], nn.QuadraticCost)
# net.SGD(training_data, 60, 10, 3.0, test_data=test_data, reg="L1", lmbda=5.0)

# print("Cross Entropy CF with 100 hidden neurons and L1 regularization")
# net = nn.Network([784, 100, 10], nn.CrossEntropyCost)
# net.SGD(training_data, 60, 10, 0.5, test_data=test_data, reg="L1", lmbda=5.0)

# print("Log-likelihood CF + Softmax with 100 hidden neurons and L1 regularization")
# net = nn.Network([784, 100, 10], nn.LogLikelihoodCost, output_fn=nn.Softmax)
# net.SGD(training_data, 60, 10, 0.5, test_data=test_data, reg="L1", lmbda=5.0)
