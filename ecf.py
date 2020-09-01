# Author: Pierre-François Gimenez <pierre-francois.gimenez@laas.fr>
# License: MIT

import numpy as np
import math
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class EmpiricalChristoffelFunction(BaseEstimator, OutlierMixin):
    """Unsupervised Outlier Detection using the empirical Christoffel function

    Fitting complexity: O(n*p^d+p^(3d))
    Prediction complexity: O()
    where n is the number of examples, p is the number of features and d is the degree of the polynomial.

    This package follows the scikit-learn objects convention.

    Parameters
    ----------
    degree : int, default=4
        The degree of the polynomial. Higher the degree, more complex the model.

    Attributes
    ----------
    score_ : ndarray of shape (n_samples,)
        The density of the training samples. The higher, the more normal.

    References
    ----------
    Lasserre, J. B., & Pauwels, E. (2019). The empirical Christoffel function with applications in data analysis. Advances in Computational Mathematics, 45(3), 1439-1468.
    arXiv version: https://arxiv.org/pdf/1701.02886.pdf

    Examples
    --------
    >>> import ecf
    >>> import numpy as np
    >>> c = ecf.EmpiricalChristoffelFunction()
    >>> X = np.array([[0,2],[1,1.5],[0.2,1.9],[100,1.2]])
    >>> c.fit_predict(X)
    array([ 1,  1,  1, -1])
    >>> c.score_
    array([ 3.99998702,  4.00000255,  3.99997548, -8.04537834])
    """
    monpowers = None
    score_ = None
    model_ = None
    degree = None

    def __init__(self, degree=4):
        self.degree = degree

    def get_params(self, deep=True):
        return {"degree": self.degree}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _compute_mat(self, X):
        nb_mon = self.monpowers.shape[0]
        mat = np.empty((0,nb_mon))
        for x in X:
            x = np.tile([x], (nb_mon,1))
            x = np.power(x, self.monpowers)
            x = np.prod(x,axis=1)
            mat = np.concatenate((mat,[x]))
        return mat

    def fit(self, X, y=None):
        """Fit the model using X as training data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
        """
        X = check_array(X)
        n,p = X.shape
        if self.degree == 0:
            self.monpowers = np.zeros((1,p))
        else:
            # create the monome powers
            self.monpowers = np.identity(p)
            self.monpowers = np.flip(self.monpowers,axis=1) # flip LR
            last = np.copy(self.monpowers)
            for i in range(1,self.degree): # between 1 and degree-1
                new_last = np.empty((0,p))
                for j in range(p):
                    tmp = np.copy(last)
                    tmp[:,j] += 1
                    new_last = np.concatenate((new_last, tmp))
                # remove duplicated rows
                tmp = np.ascontiguousarray(new_last).view(np.dtype((np.void, new_last.dtype.itemsize * new_last.shape[1])))
                _, idx = np.unique(tmp, return_index=True)
                last = new_last[idx]

                self.monpowers = np.concatenate((self.monpowers, last))
            self.monpowers = np.concatenate((np.zeros((1,p)),self.monpowers))

        # create the model
        nb_mon = self.monpowers.shape[0]
        mat = self._compute_mat(X)
        self.model_ = np.linalg.inv(np.dot(np.transpose(mat),mat)/n+np.identity(nb_mon)*0.000001)
        return self

    def predict(self, X):
        """Predict the labels (1 inlier, -1 outlier) of X according to ECF.
        This method allows to generalize prediction to *new observations* (not in the training set).
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The query samples.
        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            Returns -1 for anomalies/outliers and +1 for inliers.
        """

        check_is_fitted(self)
        X = check_array(X)

        _,p = X.shape
        self.score_samples(X)
        # level = math.factorial(p + self.degree) / (math.factorial(p) * math.factorial(self.degree))
        level = 0
        return np.array([-1 if s <= level else 1 for s in self.score_])

    def score_samples(self, X):
        X = check_array(X)
        assert self.monpowers is not None
        mat = self._compute_mat(X)
        self.score_ = np.sum(mat*np.dot(mat,self.model_),axis=1)
        return self.score_

    def fit_predict(self, X, y=None):
        """Fits the model to the training set X and returns the labels.
        Label is 1 for an inlier and -1 for an outlier according to the ECF score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The query samples.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            Returns -1 for anomalies/outliers and 1 for inliers.
        """
        super().fit_predict(X)

