import numpy as np
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data

nnfs.init()

# X = [
#     [1, 2, 3, 2.5],
#     [2.0, 5.0, -1.0, 2.0],
#     [-1.5, 2.7, 3.3, -0.8]
# ]

X, y = spiral_data(100, 3)


class LayerDense:

    def __init__(self, nInput, nNeurons):
        # Get a matrix of size nInput * nNeurons filled with random values between -1 and 1
        self.weights = 0.10 * np.random.randn(nInput, nNeurons)

        # Get a vector of size nNeurons filled with 0
        self.biases = np.zeros((1, nNeurons))
        self.output = 0

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:

    def __init__(self):
        self.output = 0

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Layer1 = first layer (take the input)
layer1 = LayerDense(2, 5)
activation1 = ActivationReLU()

# Forward propagation
layer1.forward(X)
output = layer1.output

plt.scatter(output[:, 0], output[:, 1])
plt.show()

# Rectified Linear Activation
activation1.forward(output)
rectifiedOutput = activation1.output
plt.scatter(rectifiedOutput[:, 0], rectifiedOutput[:, 1])
plt.show()
