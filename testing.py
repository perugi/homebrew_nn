from nn import *

nn = Network([2, 4, 1])
print(nn.forward(np.array([1, 2])))
