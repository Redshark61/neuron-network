# pylint: disable=attribute-defined-outside-init

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


class LayerDense:

    def __init__(self, nInput, nNeurons):
        # Get a matrix of size nInput * nNeurons filled with random values between -1 and 1
        self.weights = 0.10 * np.random.randn(nInput, nNeurons)

        # Get a vector of size nNeurons filled with 0
        self.biases = np.zeros((1, nNeurons))
        self.output = 0

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        # Subtract the max value from each row not to get overflow
        expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = expValues / np.sum(expValues, axis=1, keepdims=True)
        self.output = probabilities


X, y = spiral_data(samples=100, classes=3)

dense1 = LayerDense(2, 3)
activation1 = Activation_ReLU()
dense2 = LayerDense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])
