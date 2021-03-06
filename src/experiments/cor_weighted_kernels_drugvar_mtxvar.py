__author__ = 'victor'
import os
from base.wrapper import MethodsPrediction
from base.data import load_data, update_results, write_results
from base.kernels import DiracKernel, cosine_similarity, equal
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
    train, test = load_data(corona=True)
    selected_columns = ["baselineDAS", "Age", "Gender", "Drug", "Mtx"]
    y = np.array(train["Response.deltaDAS"])
    x = np.array(train[selected_columns])

    x_test = np.array(test[selected_columns])
    print x_test.shape
    ids = np.array(test["ID"])
    #Build a method with the dictionary and another one with the grid of parameters
    methods = {
        'sumSVR': sumSVR
    }

    params_grid = {
        'sumSVR': {
            'dim': [5],
            'C': np.arange(0.1, 2, 0.4),
            'epsilon': np.arange(0.01, 0.1, 0.02),
            #'degree': np.arange(1, 10),
            'kernel_functions':[[cosine_similarity, cosine_similarity, DiracKernel, equal, DiracKernel]],
            'w': list(product(range(1,5), repeat=5))
        }
    }
    #Build and run the comparison. tr_scoring has to be constructed like it is shown here.
    comp = MethodsPrediction(methods, params_grid=params_grid)
    predictions = comp.process(x, y, x_test, train_cv=10,
                          tr_scoring=make_scorer(scorer, greater_is_better=True), n_jobs=8)
    final_results = {}
    for method in predictions:
        final_results[method] = update_results(ids, predictions, {})
    return final_results, ids

if __name__ == "__main__":
    results, ids = main()
    for model in results:
        write_results(os.path.join(OUTPUT,'results__drugVar_mtxVar_%s.csv'%model),results[model],ids)