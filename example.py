from pynn import *
import random

random.seed(42)

model = NeuralNetwork(
    layers=[
        Dense(2, 8),
        ReLU(),
        Dense(8, 8),
        ReLU(),
        Dense(8, 3),
        Softmax()
    ],
    lr=0.1
)

model.train(X, Y, epochs=2000, batch_size=2)
