from numpy import ndarray
import numpy as np
from scipy.special import logsumexp


def assert_same_shape(a: ndarray, b: ndarray):
    assert a.shape == b.shape,\
            '''
		Two ndarrays should have the same shape;
		instead, first ndarray's shape is {0}
		and second ndarray's shape is {1}
		'''.format(tuple(a.shape), tuple(b.shape))


def to_2d_np(a: ndarray, type: str = "col") -> ndarray:
    assert a.ndim == 1
    if type == "col":
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)


def softmax(x, axis=None):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


def onehot_encode(t: ndarray, num_class: int) -> ndarray:
    num_labels = len(t)
    train_labels = np.zeros((num_labels, num_class))
    for i in range(num_labels):
        train_labels[i][t[i]] = 1
    return train_labels
