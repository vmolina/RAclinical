from __future__ import division
from sklearn.svm import SVR
import numpy as np

__author__ = 'victor'

class sumSVR(object):

    def __init__(self, dim=None,  *args, **kwargs):
        self.dim = dim if dim is not None else 1

        w = kwargs.pop("w", None)

        self.kernel_functions = kwargs.pop("kernel_functions", [])
        if self.kernel_functions is not None:
            self.kernel_kwargs = kwargs.pop("kernel_kwargs", [{} for i in self.kernel_functions])
        else:
            self.kernel_kwargs = []

        kwargs["kernel"] = "precomputed"
        if w is None:
            w = np.ones(dim)

        self.w = w / np.linalg.norm(w)
        self.x = kwargs.pop('x', None)



        self.SVR = SVR(*args, **kwargs)

    def fit(self, x, y):
        self.x = x
        kernel_train = np.zeros((x.shape[0], x.shape[0]))
        for i in range(self.dim):
            x_i = x[:,i]
            kernel_i = self.kernel_functions[i](x_i, **self.kernel_kwargs[i])
            kernel_train += self.w[i] * kernel_i

        self.SVR.fit(kernel_train,y)

    def predict(self, x):
        kernel_test = np.zeros((x.shape[0], self.x.shape[0]))
        for i in range(self.dim):
            x_i = x[:,i]
            tr_i = self.x[:,i]
            kernel_i = self.kernel_functions[i](x_i, tr_i, **self.kernel_kwargs[i])
            kernel_test += self.w[i] * kernel_i

        return self.SVR.predict(kernel_test)

    def get_params(self, deep=False):
        params = self.SVR.get_params()
        params['dim'] = self.dim
        params['w'] = self.w
        params['kernel_functions'] = self.kernel_functions
        params['kernel_kwargs'] = self.kernel_kwargs
        params['x'] = self.x
        return params

    def set_params(self, **params):
        self.__init__(**params)
        return self