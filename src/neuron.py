"""Core implementation of a single artifical neuron."""

import math

def sigmoid(z:float)-> float:
    """Return the sigmoid activation of z."""
    return 1.0 / (1.0+math.exp(-z))

def weighted_sum(
    x1: float,
    x2: float,
    w1: float,
    w2: float,
    bias: float,
    )-> float:
    """Calculate the weighted sum plus bias."""
    return (x1 * w1) + (x2 * w2) + bias

def neuron(
    x1: float,
    x2: float,
    w1: float,
    w2: float,
    bias: float,
) -> float:
    """Calculate the output of a two-input neuron."""
    z = weighted_sum(x1,x2,w1,w2,bias)
    return sigmoid(z)

