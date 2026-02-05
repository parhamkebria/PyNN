from pynn import *
import random
import math

random.seed(42)

# build the model
model = NeuralNetwork(
    layers=[
        Dense(1, 8),
        ReLU(),
        Dense(8, 8),
        ReLU(),
        Dense(8, 3),
        Softmax()
    ],
    lr=0.1
)

X = []
Y = []

# XOR sample data
for i in range(60):
    x = random.uniform(0, 3)
    X.append([x])
    
    if math.sin(x) > 0.5:
        Y.append([1, 0, 0])
    elif math.sin(x) > -0.5:
        Y.append([0, 1, 0])
    else:
        Y.append([0, 0, 1])

# train the model
model.train(X, Y, epochs=100, batch_size=2) # adjust the batch size and epochs as you wish

# test the model
print("\nPredictions:")
for x in X:
    print(x, model.forward(x))
