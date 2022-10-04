import mnist_loader
import nn

training_data, test_data = mnist_loader.load_data()
net = nn.Network([784, 30, 10])

for i in range(1, 6):
    print("")
    print(f"##### NN no. {i} #####")
    print("")
    net.SGD(training_data, 30, 10, 0.1, test_data=test_data)
# print(training_data[0][0].shape)
