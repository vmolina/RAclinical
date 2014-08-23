from __future__ import division
import numpy as np
from sklearn.metrics.pairwise import linear_kernel, manhattan_distances

__author__ = 'victor'

def min_max(x,y=None):

    if y is None:
        y = x

    if len(x.shape) == 1:
        x.shape = (len(x), 1)
    if len(y.shape) == 1:
        y.shape = (len(y), 1)


    kernel = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            m = np.sum(np.minimum(x[i, :], y[j, :]), axis=None)

            M = np.sum(np.maximum(x[i, :], y[j, :]), axis=None)
            if 0 == M :
                kernel[i, j] = 1.
            else:
                kernel[i, j] = m / M
    return kernel

def DiracKernel(x, y=None):
    if y is None:
        y = x

    if len(x.shape) == 1:
        x.shape = (len(x), 1)
    if len(y.shape) == 1:
        y.shape = (len(y), 1)

    kernel = np.dot(x, y.T) + np.dot(x-1, y.T-1)

    return kernel

def IdentityorZero(x,y=None):
    if y is None or (x==y).all():
        return np.eye(x.shape[0])
    else:
        return np.zeros((x.shape[0], y.shape[0]))

def equal(x,y=None):
    if y is None:
        y = x
    if len(x.shape) == 1:
        x.shape = (len(x), 1)
    if len(y.shape) == 1:
        y.shape = (len(y), 1)
    kernel = np.equal(x,y.T)
    return kernel


def cosine_similarity(x, y=None):
    if y is None:
        y = x
    if len(x.shape) == 1:
        x.shape = (len(x), 1)
    if len(y.shape) == 1:
        y.shape = (len(y), 1)
    kernel = linear_kernel(x,y)
    x_norm = np.sum(x * x, axis=1)
    x_norm.shape = (len(x_norm), 1)
    y_norm = np.sum(y*y, axis=1)
    y_norm.shape = (len(y_norm), 1)


    return kernel / np.dot(x_norm, y_norm.T)
