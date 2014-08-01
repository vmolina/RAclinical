__author__ = 'victor'

from sklearn.cross_validation import KFold
import pandas as pd
import numpy as np
import scipy.stats as sps
import os.path as path
import csv
import pickle

CLINICAL_FILE = path.join(path.dirname(__file__), "../../data/TrainingData_PhenoCov_Release.txt" )
TRAIN_FILE = path.join(path.dirname(__file__), "train.txt")
TEST_FILE = path.join(path.dirname(__file__), "test.txt")

def load_data():
    clinical_data = pd.read_csv(CLINICAL_FILE, delim_whitespace=True)

    lost_gender = np.isnan(clinical_data["Gender"])
    clinical_data["Gender"][lost_gender] = sps.mode(clinical_data["Gender"][map(lambda x: not x, lost_gender)])[0]

    lost_age = np.isnan(clinical_data["Age"])
    clinical_data["Age"][lost_age] = np.mean(clinical_data["Age"][map(lambda x: not x, lost_gender)])


    train = clinical_data[pd.isnull(clinical_data['Response.deltaDAS']) == False]
    test = clinical_data[pd.isnull(clinical_data['Response.deltaDAS'])]

    return train, test


def load_folds(fold_path,train_fold_path):
    f = open(fold_path)
    folds = pickle.load(f)
    f.close()
    f = open(train_fold_path)
    train_folds = pickle.load(f)
    f.close()
    return folds, train_folds


def create_folds_and_save(n_folds,n_train_folds, output_dir=None):
    n_train = load_data()[0].shape[0]
    folds_ = KFold(n_train, n_folds=n_folds, shuffle=True)
    train_folds = []
    folds = []
    for train, test in folds_:
        folds.append((train, test))
        train_folds.append( [(train,test) for train, test in KFold(len(train), n_folds=n_train_folds, shuffle=True)])

    output_dir = path.dirname(__file__) if output_dir is None else output_dir

    with open(path.join(output_dir,"folds.pickle"),"w") as folds_out:
        pickle.dump(folds, folds_out)

    with open(path.join(output_dir, "train_folds.pickle"),"w") as train_folds_out:
        pickle.dump(train_folds, train_folds_out)

if __name__ == "__main__":
    create_folds_and_save(10, 10, path.join(path.dirname(__file__),"../../data/"))