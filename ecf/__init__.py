# Author: Pierre-François Gimenez <pierre-francois.gimenez@laas.fr>
# License: MIT

import numpy as np
import math
from random import shuffle
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class EmpiricalChristoffelFunction(BaseEstimator, OutlierMixin):
    """Unsupervised outlier and novelty detection using the empirical Christoffel function

    This model is suited for moderate dimensions and potentially very large number of observations.

    Fitting complexity: O(n*p^d+p^(3d))
    Prediction complexity: O(n*p^(2d))
    where n is the number of examples, p is the number of features and d is the degree of the polynomial. This complexity assumes d is constant. See [1] for more details.

    This package follows the scikit-learn objects convention.

    Parameters
    ----------
    degree : int, default=4
        The degree of the polynomial. Higher the degree, more complex the model.
    n_components : int, default=4
        The maximal number of components.
    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion of outliers in the data set. When fitting this is used to define the threshold on the scores of the samples.
        - if 'auto', the threshold is determined as in the
          original paper [1],
        - if a float, the contamination should be in the range [0, 0.5].

    Attributes
    ----------
    score_ : ndarray of shape (n_samples,)
        The score of the training samples. The lower, the more normal.

    References
    ----------
    [1] Pauwels, E., & Lasserre, J. B. (2016). Sorting out typicality with the inverse moment matrix SOS polynomial. In Advances in Neural Information Processing Systems (pp. 190-198).
    [2] Lasserre, J. B., & Pauwels, E. (2019). The empirical Christoffel function with applications in data analysis. Advances in Computational Mathematics, 45(3), 1439-1468.

    """
    monpowers = None # the monomials of degree less of equal to d
    score_ = None # score of the last predict data
    predict_ = None
    level_set_ = None
    model_ = None
    robust_scaler_ = None
    pca_ = None
    decision_scores_ = None # score of the training data, pyod-compliant
    labels_ = None # labels of the training data, pyod-compliant

    def __init__(self, degree=4, n_components=4, contamination="auto"):
        self.degree = degree
        self.n_components = n_components
        self.contamination = contamination

    def _process_data(self, X):
        # if no need to remove components
        # verify if not already processed
        if self.pca_ is None or np.array(X).shape[1] <= self.n_components:
            return X
        else:
            return self.pca_.transform(self.robust_scaler_.transform(X))

    def get_params(self, deep=True):
        return {"degree": self.degree, "n_components": self.n_components, "contamination": self.contamination}

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
        # mat is denoted v_d(x) in [1]
        # mat size: O(n*p^d)
        return mat

    def fit(self, X, y=None):
        # self._fit_one_iter(X)
        # if self.iterative:
        #     p = np.percentile(self.decision_scores_, 80)
        #     X = X[self.decision_scores_ < p]
        #     self._fit_one_iter(X)
        # return self

    # def _fit_one_iter(self, X, y=None):
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

        if self.contamination != 'auto' and not(0. < self.contamination <= .5):
            raise ValueError("contamination must be in (0, 0.5], got: %f" % self.contamination)

        if self.n_components is not None and np.array(X).shape[1] > self.n_components:
        # learn the new robust scaler and PCA
            self.robust_scaler_ = RobustScaler()
            self.pca_ = PCA(n_components=self.n_components)
            self.pca_.fit(self.robust_scaler_.fit_transform(X))
        else:
            self.robust_scaler_ = None
            self.pca_ = None

        X = self._process_data(X)
        n,p = X.shape
        # monome powers, denoted v_d(X) in [1]
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

        nb_mon = self.monpowers.shape[0]
        # in fact, level_set == nb_mon
        mat = self._compute_mat(X)
        md = np.dot(np.transpose(mat),mat)
        # md is denoted M_d(mu) in [1]. It is the moment matrix.
        # cf. the last equation of Section 2.2 in [1]
        self.model_ = np.linalg.inv(md/n+np.identity(nb_mon)*0.000001)
        # add a small value on the diagonal to avoid numerical problems
        # model is M_d(mu)^-1 in [1]
        self.decision_scores_ = self.decision_function(X)

        # level set proposed in [1]
        if self.contamination == "auto":
            self.level_set_ = math.factorial(p + self.degree) / (math.factorial(p) * math.factorial(self.degree))
        else:
            self.level_set_ = np.percentile(self.decision_scores_, 100. * (1 - self.contamination))

        self.labels_ = self.predict(X)
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
        X = self._process_data(X)

        n,p = X.shape
        self.decision_function(X)
        self.predict_ = np.ones(n, dtype=int)
        self.predict_[self.score_ >= self.level_set_] = -1
        return self.predict_

    def decision_function(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X = self._process_data(X)
        assert self.monpowers is not None

        mat = self._compute_mat(X)
        # cf. Eq. (2) in [1]
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
        return super().fit_predict(X)

class BaggedECF(BaseEstimator, OutlierMixin):

    def __init__(self, degree=3, n_models=50, contamination="auto"):
        self.degree = degree
        self.n_models = n_models
        self.contamination = contamination

    def fit(self, X, y=None):
        X = check_array(X)
        n,p = X.shape
        self.n_components = max(2,math.floor(math.sqrt(p)))
        self.n_components = min(30,max(2,math.floor(p/3)))
        # can't bag with p <= n_components, i.e. if p=1 or p=2
        if p <= self.n_components:
            self.models_ = [EmpiricalChristoffelFunction(self.degree, None, self.conmatimation)]
            self.features_ = [[True]*p]
            self.n_models = 1

        self.models_ = []
        self.features_ = []
        for i in range(self.n_models):
            f = np.array([True]*self.n_components + [False]*(p-self.n_components))
            shuffle(f)
            self.features_.append(f)
            m = EmpiricalChristoffelFunction(self.degree, None, self.contamination)
            train_set = X[np.random.choice(X.shape[0], X.shape[0], replace=True),:]
            m.fit(train_set[:,f])
            self.models_.append(m)

        self.decision_scores_ = self.decision_function(X)

    def decision_function(self, X):
        # check_is_fitted(self) # TODO
        X = check_array(X)
        if False:
            pred = np.array([self.models_[i].predict(X[:,self.features_[i]]) for i in range(self.n_models)])
            pred[pred==1] = 0
            pred[pred==-1] = 1
            self.score_ = np.sum(pred,axis=0)
            return self.score_
        if True:
            pred = np.array([self.models_[i].decision_function(X[:,self.features_[i]]) for i in range(self.n_models)])
            self.score_ = np.sum(pred,axis=0)
            return self.score_
