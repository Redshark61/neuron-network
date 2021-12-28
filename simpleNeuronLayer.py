import numpy as np
np.random.seed(0)

X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]


class LayerDense:

    def __init__(self, nInput, nNeurons):
        # Get a matrix of size nInput * nNeurons filled with random values between -1 and 1
        self.weights = 0.10 * np.random.randn(nInput, nNeurons)

        # Get a vector of size nNeurons filled with 0
        self.biases = np.zeros((1, nNeurons))
        self.output = 0

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# Layer1 = first layer (take the input)
layer1 = LayerDense(4, 5)

# Layer2 = second layer (take the output of layer1) and end with the output
# so the first param (nInput) is the number of neurons in the first layer
layer2 = LayerDense(5, 2)

# Forward propagation
layer1.forward(X)

# Forward propagation
layer2.forward(layer1.output)

print(layer1.output)
print(layer2.output)
