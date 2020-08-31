import numpy as np
import math

class Christoffel():
    """Unsupervised Outlier Detection using the empirical Christoffel function

    Fitting complexity: O(n*p^d+p^(3d))
    Prediction complexity: O()
    where n is the number of examples, p is the number of features and d is the degree of the polynomial.

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
    >>> import christoffel
    >>> c = christoffel.Christoffel()
    >>> X = np.array([[0,2],[1,1.5],[0.2,1.9],[100,1.2]])
    >>> c.fit_predict(X)
    [ 1  1  1 -1]
    >>> c.score_
    [   3.99998999    4.00015445    3.99999215 -151.32839602]
    """
    monpowers = None
    score_ = None
    degree = None

    def __init__(self, degree=4):
        self.degree = degree

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
        n,p = X.shape
        if self.degree==0:
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
        self.model = np.linalg.inv(np.dot(np.transpose(mat),mat)/n+np.identity(nb_mon)*0.000001)

    def predict(self, X):
        _,p = X.shape
        self.score_samples(X)
        # level = math.factorial(p + self.degree) / (math.factorial(p) * math.factorial(self.degree))
        level = 0
        return np.array([-1 if s <= level else 1 for s in self.score_])

    def score_samples(self, X):
        assert self.monpowers is not None
        mat = self._compute_mat(X)
        self.score_ = np.sum(mat*np.dot(mat,self.model),axis=1)
        return self.score_

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)

