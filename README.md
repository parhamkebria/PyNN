# PyNN — A Neural Network Framework Built from Scratch!

> **No NumPy. No PyTorch. No TensorFlow. No shortcuts.**  
> Just **pure standard Python**, raw math, and a deep respect for how neural networks *actually* work.

**PurePyNN** is a minimalist yet fully-functional neural network framework implemented using **only built-in Python modules** (`math`, `random`). It’s designed for learning, teaching, and understanding neural networks at their core — not for hiding them behind abstractions.

If you’ve ever wondered *“What is PyTorch really doing under the hood?”* — This repo is your answer.

---

## Features

**Fully-connected (Dense) layers**  
**ReLU activation**  
**Softmax output**  
**Cross-Entropy loss**  
**Backpropagation from scratch**  
**Mini-batch gradient descent**  
**Arbitrary network depth**  
**Object-oriented layer design**  
**Zero third-party dependencies**  

Everything is implemented using:
- Python lists
- Loops
- Basic math

Nothing else. Ever.

---

## Architecture Overview

PyNN follows a clean, modular design:

```text
NeuralNetwork
 ├── Dense
 ├── ReLU
 ├── Dense
 ├── ReLU
 ├── Dense
 └── Softmax

