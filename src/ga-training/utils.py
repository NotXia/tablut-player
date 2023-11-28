import numpy as np


def normalize(x):
    min_x = min(x)
    shift = abs(min_x) if min_x < 0 else 0
    x = [i + shift for i in x]
    sum_x = sum(x)
    return [i / sum_x for i in x]


def softmax(x):
    return np.exp(x) / sum(np.exp(x))