__author__ = 'victor'
import os
from base.wrapper import MethodComparison
from base.data import load_data
from sklearn.svm import SVR
from sklearn.metrics import make_scorer
import numpy as np
import scipy.stats as sps
import pickle

DATA = os.path.join(os.path.dirname(__file__), '../../data')
FOLDS = os.path.join(DATA, 'folds.pickle')
TRAIN_FOLDS = os.path.join(DATA, 'train_folds.pickle')


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
        'SVR': SVR
    }

    params_grid = {
        'SVR': {
            'C': np.arange(0.1, 2, 0.1),
            'epsilon': np.arange(0.01, 0.2, 0.02),
            #'degree': np.arange(1, 10),
            'gamma': np.arange(0, 1, 0.1)

        }
    }
    #Build and run the comparison. tr_scoring has to be constructed like it is shown here.
    comp = MethodComparison(methods, params_grid=params_grid)
    scores = comp.process(x, y, scorer, repeats=10, train_cv=10,
                          tr_scoring=make_scorer(scorer, greater_is_better=True), n_jobs=3)

    return scores

if __name__ == "__main__":
    scores = main()
    with open('results.pickle', 'w') as f:
        pickle.dump(scores, f)