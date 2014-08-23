from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
import numpy as np

__author__ = 'victor'


class MethodComparison(object):

    def __init__(self,methods, params_grid, groups=None):
        self.methods = methods
        self.params_grid = params_grid

    def process(self,x,y, score, tr_scoring=None,repeats=1, cv=10, train_cv=3, n_jobs= 1):

        n_samples =x.shape[0]

        scores = {}
        for method in self.methods:
            scores[method] = np.zeros((repeats,cv))

        for i in range(repeats):
            print "Repetition %i" % i

            for j, (train,test) in enumerate(KFold(n_samples,n_folds=cv, indices=False)):
                print "Fold %i" % j
                x_train = x[train,:]
                x_test = x[test,:]
                y_train = y[train]
                y_test = y[test]

                for method in self.methods:
                    print "Method: " + method
                    pred_method = GridSearchCV(self.methods[method](), self.params_grid[method], scoring=tr_scoring,
                                               cv=train_cv, n_jobs=n_jobs, refit=True, verbose=True)
                    pred_method.fit(x_train, y_train)
                    y_pred = pred_method.predict(x_test)
                    scores[method][i, j] = score(y_pred, y_test)

        return scores

class MethodsPrediction(object):
    def __init__(self, methods, params_grid, only_train=False):
        self.methods = methods
        self.params_grid = params_grid
        self.only_train = only_train

    def process(self, x_train, y_train, x_test, tr_scoring=None, train_cv=3, n_jobs= 1):
        predictions = {}
        for method in self.methods:
            print "Method: " + method
            pred_method = GridSearchCV(self.methods[method](), self.params_grid[method], scoring=tr_scoring,
                                               cv=train_cv, n_jobs=n_jobs, refit=True, verbose=True)
            pred_method.fit(x_train, y_train)
            if self.only_train:
                predictions[method] = pred_method
            else:
                predictions[method] = pred_method.predict(x_test)

        return predictions