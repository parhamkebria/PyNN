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


# ReLU activation function
class ReLU(Layer):
    def forward(self, inputs):
        self.inputs = inputs
        return [max(0.0, x) for x in inputs]

    def backward(self, grad_output):
        return [
            grad_output[i] if self.inputs[i] > 0 else 0.0
            for i in range(len(grad_output))
        ]


# Softmax 
class Softmax(Layer):
    def forward(self, inputs):
        max_val = max(inputs)
        exp_vals = [math.exp(x - max_val) for x in inputs]
        total = sum(exp_vals)
        self.outputs = [x / total for x in exp_vals]
        return self.outputs

    def backward(self, grad_output):
        return grad_output  # simplified (paired with CE loss)


# cross-entropy for softmax and classification tasks
class CrossEntropyLoss:
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        loss = 0.0

        for i in range(len(targets)):
            loss -= targets[i] * math.log(predictions[i] + 1e-9)

        return loss

    def backward(self):
        return [
            self.predictions[i] - self.targets[i]
            for i in range(len(self.targets))
        ]


# the complete NN class
class NeuralNetwork:
    def __init__(self, layers, lr=0.1):
        self.layers = layers
        self.lr = lr
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self):
        for layer in self.layers:
            layer.update(self.lr)

    def train(self, X, Y, epochs=1000, batch_size=1):
        for epoch in range(epochs):
            total_loss = 0.0

            for i in range(0, len(X), batch_size):
                batch_X = X[i:i + batch_size]
                batch_Y = Y[i:i + batch_size]

                for x, y in zip(batch_X, batch_Y):
                    preds = self.forward([x] if isinstance(x, (int, float)) else x)
                    loss = self.loss_fn.forward([preds] if isinstance(preds, (int, float)) else preds,
                                                [y] if isinstance(y, (int, float)) else y)
                    total_loss += loss

                    grad = self.loss_fn.backward()
                    self.backward(grad)
                    self.update()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
