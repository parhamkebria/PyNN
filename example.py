from pynn import *
import random

random.seed(42)

# build the model
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

# train the model
model.train(X, Y, epochs=2000, batch_size=2)

# test the model
print("\nPredictions:")
for x in X:
    print(x, model.forward(x))
