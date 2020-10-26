# Author: Pierre-François Gimenez <pierre-francois.gimenez@laas.fr>
# License: MIT

import numpy as np
import math
from random import shuffle
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, sigmoid_kernel, linear_kernel

class EmpiricalChristoffelFunction(BaseEstimator, OutlierMixin):
    """Unsupervised outlier and novelty detection using the empirical Christoffel function

    This model is suited for moderate dimensions and potentially very large number of observations.

    Fitting complexity: O(n*p^d+p^(3d))
    Prediction complexity: O(n*p^(2d))
    where n is the number of examples, p is the number of features and d is the degree of the polynomial. This complexity assumes d is constant. See [1] for more details.

    This package follows the scikit-learn objects convention.

    Parameters
    ----------
    degree : int, default=3
        The degree of the polynomial. Higher the degree, more complex the model. It should be at least 2.
    n_components : int, default=4
        The maximal number of components.
    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion of outliers in the data set. When fitting this is used to define the threshold on the scores of the samples.
        - if 'auto', the threshold is determined as in the
          original paper [1],
        - if a float, the contamination should be in the range [0, 0.5].
    filtering_frac : float, default=1.0
        Learn with only the lowest 'filtering_frac' fraction of the training set in terms of outlier score (the most normal instances). Double the learning time if filtering_frac < 1.0.

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
    # post_robust_scaler_ = None
    pca_ = None
    decision_scores_ = None # score of the training data, pyod-compliant
    labels_ = None # labels of the training data, pyod-compliant

    def __init__(self, degree=3, n_components=4, contamination="auto", filtering_frac=1.0):
        self.degree = degree
        self.n_components = n_components
        self.contamination = contamination
        self.filtering_frac = filtering_frac

    def _process_data(self, X):
        # if no need to remove components
        # verify if not already processed
        if self.pca_ is None or np.array(X).shape[1] <= self.n_components:
            return X
            # return self.post_robust_scaler_.transform(X)
        else:
            return self.pca_.transform(self.robust_scaler_.transform(X))
            # return self.post_robust_scaler_.transform(self.pca_.transform(self.robust_scaler_.transform(X)))

    def get_params(self, deep=True):
        return {"degree": self.degree, "n_components": self.n_components, "contamination": self.contamination, "filtering_frac": self.filtering_frac}

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
        self._fit_once(X)
        if self.filtering_frac < 1.0:
            p = np.percentile(self.decision_scores_, self.filtering_frac * 100)
            X = X[self.decision_scores_ < p]
            self._fit_once(X)
        return self

    def _fit_once(self, X, y=None):
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
            data = self.pca_.fit_transform(self.robust_scaler_.fit_transform(X))
            # self.post_robust_scaler_ = RobustScaler().fit_transform(data)
        else:
            self.robust_scaler_ = None
            # self.post_robust_scaler_ = RobustScaler().fit(X)
            self.pca_ = None

        unprocessed_X = np.copy(X)
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

        self.labels_ = self.predict(unprocessed_X)
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

    # Reference: Kroó, A., & Lubinsky, D. S. (2013). Christoffel functions and universality in the bulk for multivariate orthogonal polynomials. Canadian Journal of Mathematics, 65(3), 600-620.
    def density(self, X):
        self.decision_function(X)
        X = self._process_data(X)
        n,p = X.shape
        self.density_ = np.zeros((n))
        a = (self.degree+1)/2
        factor = math.gamma(a) / (math.pi**a) * math.factorial(p + self.degree) / (math.factorial(p) * math.factorial(self.degree))
        for i in range(n):
            self.density_[i] = factor / self.score_[i] * math.sqrt(max(0,1 - np.linalg.norm(X[i])*1.35/2))
            # 1.35/2 comes from the fact that we consider the euclidian ball of radius 2 sigma (so 95% of the points fall inside). After the robust standardization, the radius of the ball is about 1.35 sigma (under gaussian assumption).
        return self.density_


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

class KECF(BaseEstimator, OutlierMixin):
    """Unsupervised outlier and novelty detection using the kernelized inverse Christoffel function

    This package follows the scikit-learn objects convention.

    Parameters
    ----------
    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion of outliers in the data set. When fitting this is used to define the threshold on the scores of the samples.
        - if 'auto', the threshold is determined as in the
          original paper [1],
        - if a float, the contamination should be in the range [0, 0.5].
    filtering_frac : float, default=1.0
        Learn with only the lowest 'filtering_frac' fraction of the training set in terms of outlier score (the most normal instances). Double the learning time if filtering_frac < 1.0.

    Attributes
    ----------
    score_ : ndarray of shape (n_samples,)
        The score of the training samples. The lower, the more normal.

    References
    ----------
    [3] Askari, A., Yang, F., & Ghaoui, L. E. (2018). Kernel-based outlier detection using the inverse christoffel function. arXiv preprint arXiv:1806.06775.
    """
    def __init__(self, kernel="rbf", gamma="scale", C=5000, degree=3, coef0=1, contamination="auto"):
        # normalized dataset
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.degree = degree
        self.coef0 = coef0
        self.contamination = contamination

    def get_params(self, deep=True):
        return {"kernel": self.kernel, "gamma": self.gamma, "C": self.C, "degree": self.degree, "coef0": self.coef0, "contamination": self.contamination}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


    def _apply_kernel(self, X, Y):
        if self.kernel == "rbf":
            return rbf_kernel(X, Y, self.gamma)
        elif self.kernel == "sigmoid":
            return sigmoid_kernel(X, Y, self.gamma, self.coef0)
        elif self.kernel == "poly":
            return polynomial_kernel(X, Y, self.degree, self.gamma, self.coef0)
        elif self.kernel == "linear":
            return linear_kernel(X, Y)
        elif callable(self.kernel):
            return self.kernel(X, Y)
        else:
            raise ValueError("Unknown kernel: "+str(self.kernel))

    def fit(self, X, y=None):
        self.X_train = check_array(X)
        n,p = X.shape
        # default value

        self.post_robust_scaler_ = RobustScaler().fit(self.X_train)
        if self.gamma == "auto":
            self.gamma = 1.0 / p
        elif self.gamma == "scale":
            X_var = X.var()
            if X_var != 0:
                self.gamma = 1.0 / (p * X_var)
            else:
                self.gamma = 1.0

        self.X_train = self.post_robust_scaler_.transform(self.X_train)
        Q = np.zeros((n,n))
        # compute phi * phi.T from [3]
        Q = self._apply_kernel(self.X_train, self.X_train)
        # rho as proposed by [3]
        self.rho = np.linalg.norm(Q)/(self.C*math.sqrt(n))
        self.model = np.linalg.inv(np.identity(n) + Q/self.rho)
        if self.contamination == "auto":
            if self.kernel == "poly":
                self.level_set_ = self.rho * math.factorial(p + self.degree) / (math.factorial(p) * math.factorial(self.degree))
            else:
                self.level_set_ = 100 # TODO
        else:
            self.level_set_ = np.percentile(self.decision_scores_, 100. * (1 - self.contamination))
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X = self.post_robust_scaler_.transform(X)
        phi = self._apply_kernel(X,self.X_train)
        tmp = np.dot(phi,self.model)
        self.score_ = np.zeros((len(X)))
        for i in range(tmp.shape[0]):
            self.score_[i] = np.dot(tmp[i,:],phi[i,:])
        return self.score_

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

        n,p = X.shape
        self.decision_function(X)
        self.predict_ = np.ones(n, dtype=int)
        self.predict_[self.score_ >= self.level_set_] = -1
        return self.predict_

