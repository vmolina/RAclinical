__author__ = 'victor'
import os
from base.wrapper import MethodsPrediction
from base.data import load_data, update_results, write_results
from base.kernels import DiracKernel, cosine_similarity, IdentityorZero
from base.sumSVR import sumSVR
from sklearn.metrics import make_scorer
from sklearn.cross_validation import LeaveOneOut
import numpy as np
import scipy.stats as sps
import pickle
from itertools import product

PROJECT = os.path.dirname(__file__)
DATA = os.path.join(PROJECT, '../../data')
FOLDS = os.path.join(DATA, 'folds.pickle')
TRAIN_FOLDS = os.path.join(DATA, 'train_folds.pickle')
OUTPUT = os.path.join(PROJECT,'../../../results')

def scorer(x, y):
    return sps.pearsonr(x, y)[0]


def main():
    #Load the data. A pandas DataFrame is returned.Only the training data is selected
    train, test = load_data(corona=True)
    selected_columns = ["baselineDAS", "Age", "Gender","ID"]
    predictions = {}

    for i in range(test.shape[0]):
        print i
        x_test = test.iloc[i]
        if x_test['Drug'] is np.nan:
            if x_test['Mtx'] == -1:
                x_train = train[np.logical_and(train.Drug is not np.nan, train.Mtx != -1)]
            else:
                x_train = train[np.logical_and(train.Drug is not np.nan, train.Mtx != x_test.Mtx)]
        else:
            if x_test['Mtx'] == -1:
                x_train = train[np.logical_and(train.Drug == x_test.Drug, train.Mtx != -1)]
            else:
                x_train = train[np.logical_and(train.Drug == x_test.Drug, train.Mtx == x_test.Mtx)]
            if x_train.shape[0] == 0:
                valid_drugs = ["infliximab", "adalimumab"]
                if x_test['Mtx'] == -1:
                    x_train = train[np.logical_and(np.logical_or(train.Drug == valid_drugs[0],
                                                                train.Drug == valid_drugs[1]), train.Mtx != -1)]
                else:
                    x_train = train[np.logical_and(np.logical_or(train.Drug == valid_drugs[0],
                                                                train.Drug == valid_drugs[1]), train.Mtx == x_test.Mtx)]


        id = x_test["ID"]
        x_test = np.array(x_test[selected_columns])
        x_test.shape=(1,4)
        train_groups = x_train.groupby(["Drug","Mtx"],).groups
        predictions[id] = []

        for group in train_groups:

            x_train = np.array(train.iloc[train_groups[group]][selected_columns])
            y_train = np.array(train.iloc[train_groups[group]]["Response.deltaDAS"])
            method = sumSVR(dim=4, w=[1,1,1,1],kernel_functions=[cosine_similarity, cosine_similarity, DiracKernel,IdentityorZero])
            method.fit(x_train, y_train)
            predictions[id].append(method.predict(x_test))

    return predictions, test["ID"]

if __name__ == "__main__":
    results, ids = main()
    write_results(os.path.join(OUTPUT,'q1_corona.csv'),results,ids)