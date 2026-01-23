# only standard Python libraries
import math
import random

# Layer object to be used when building NN models
class Layer:
    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def update(self, lr):
        pass

# fully connected (FC) module
class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = [
            [random.uniform(-1, 1) for _ in range(input_size)]
            for _ in range(output_size)
        ]
        self.biases = [0.0 for _ in range(output_size)]

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = []

        for w, b in zip(self.weights, self.biases):
            z = sum(w[i] * inputs[i] for i in range(len(inputs))) + b
            self.outputs.append(z)

        return self.outputs

    def backward(self, grad_output):
        self.grad_w = [
            [grad_output[i] * self.inputs[j] for j in range(len(self.inputs))]
            for i in range(len(grad_output))
        ]
        self.grad_b = grad_output[:]

        grad_input = [0.0 for _ in range(len(self.inputs))]
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                grad_input[j] += self.weights[i][j] * grad_output[i]

        return grad_input

    def update(self, lr):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j] -= lr * self.grad_w[i][j]
            self.biases[i] -= lr * self.grad_b[i]


#
