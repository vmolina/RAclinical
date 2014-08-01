from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
import numpy as np

__author__ = 'victor'


class MethodComparison(object):

    def __init__(self,methods, params_grid):
        self.methods = methods
        self.params_grids = params_grid

    def process(self,x,y, score, tr_scoring=None,repeats=1, cv=10, train_cv=3, n_jobs= 1):
        scores = {}
        for method in self.methods:
            scores[method] = np.zeros((repeats,cv))
        n_samples = x.shape[0]
        for i in range(repeats):
            print "Repetition %i" % i
            for j, (train,test) in enumerate(KFold(n_samples,n_folds=cv)):
                print "Fold %i" % j
                x_train = x[train,:]
                x_test = x[test,:]
                y_train = y[train]
                y_test = y[test]
                for method in self.methods:
                    print "Method: " + method
                    pred_method = GridSearchCV(self.methods[method](), self.params_grids[method], scoring=tr_scoring,
                                               cv=train_cv, n_jobs=n_jobs, refit=True, verbose=True)
                    pred_method.fit(x_train, y_train)
                    y_pred = pred_method.predict(x_test)
                    scores[method][i, j] = score(y_pred, y_test)

        return scores