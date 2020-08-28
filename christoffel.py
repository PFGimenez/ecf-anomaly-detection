import numpy as np

class Christoffel():
    monpowers = None

    def __init__(self, degree=2):
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

    def fit(self, X=None, y=None):
        n,p = X.shape
        if self.degree==0:
            self.monpowers=np.zeros((1,p))
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
        assert self.monpowers is not None
        mat = self._compute_mat(X)
        X = np.sum(mat*np.dot(mat,self.model),axis=1)
        return X

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

c = Christoffel(3)
X = np.random.random((6,3))
print(c.fit_predict(X))
