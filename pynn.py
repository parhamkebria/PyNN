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

