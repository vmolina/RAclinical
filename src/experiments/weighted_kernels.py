__author__ = 'victor'
import os
from base.wrapper import MethodComparison
from base.data import load_data
from base.kernels import DiracKernel, cosine_similarity
from base.sumSVR import sumSVR
from sklearn.metrics import make_scorer
import numpy as np
import scipy.stats as sps
import pickle
from itertools import product

PROJECT = os.path.dirname(__file__)
DATA = os.path.join(PROJECT, '../../data')
FOLDS = os.path.join(DATA, 'folds.pickle')
TRAIN_FOLDS = os.path.join(DATA, 'train_folds.pickle')
OUTPUT = os.path.join(PROJECT,'../../results')

def scorer(x, y):
    return sps.pearsonr(x, y)[0]


def main():
    #Load the data. A pandas DataFrame is returned.Only the training data is selected
    data = load_data()[0]
    selected_columns = ["baselineDAS", "Age", "Gender"]
    y = np.array(data["Response.deltaDAS"])
    x = np.array(data[selected_columns])
    #Build a method with the dictionary and another one with the grid of parameters
    methods = {
        'sumSVR': sumSVR
    }

    params_grid = {
        'sumSVR': {
            'dim': [3],
            'C': np.arange(0.1, 2, 0.4),
            'epsilon': np.arange(0.01, 0.1, 0.02),
            #'degree': np.arange(1, 10),
            'kernel_functions':[[cosine_similarity, cosine_similarity, DiracKernel]],
            'w': list(product(range(1,6), repeat=3))
        }
    }
    #Build and run the comparison. tr_scoring has to be constructed like it is shown here.
    comp = MethodComparison(methods, params_grid=params_grid)
    scores = comp.process(x, y, scorer, repeats=10, train_cv=3,
                          tr_scoring=make_scorer(scorer, greater_is_better=True), n_jobs=8)

    return scores

if __name__ == "__main__":
    scores = main()
    with open(os.path.join(OUTPUT,'results.pickle'), 'w') as f:
        pickle.dump(scores, f)