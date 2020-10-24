from abc import ABC, abstractmethod

import numpy as np
import scipy as sc

class SearchModel(ABC):

    @abstractmethod
    def sample(self, n_sample: int, seed: int = None):
        raise NotImplementedError

    @abstractmethod
    def fit(self, X_nxp: np.array, weights: np.array = None):
        raise NotImplementedError

    @abstractmethod
    def loglikelihood(self, X_nxp: np.array) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self):
        raise NotImplementedError

    @abstractmethod
    def set_parameters(self, value: tuple):
        raise NotImplementedError

    parameters = property(get_parameters, set_parameters)


class MultivariateGaussian(SearchModel):

    def __init__(self, dim: int = 10):
        self.dim = dim
        self.set_parameters((np.zeros([dim]), np.eye(dim)))

    def get_initialization_kwargs(self):
        kwargs = {
            "dim": self.dim,
        }
        return kwargs

    def sample(self, n_sample: int, seed: int = None) -> np.array:
        np.random.seed(seed)
        X_nxp = np.random.multivariate_normal(self._parameters[0], self._parameters[1], size=n_sample)
        return X_nxp

    def fit(self, X_nxp: np.array, weights: np.array = None):
        if weights is None:
            weights = np.ones([X_nxp.shape[0]])
        weights_nx1 = np.reshape(weights, (X_nxp.shape[0], 1))
        Xweighted_nxp = weights_nx1 * X_nxp
        mean_p = np.sum(Xweighted_nxp, axis=0, keepdims=False) / np.sum(weights_nx1)
        Xcentered_nxp = X_nxp - mean_p[None, :]

        cov_pxp = np.dot(Xcentered_nxp.T, weights_nx1 * Xcentered_nxp) / np.sum(weights_nx1)
        self._parameters = (mean_p, cov_pxp)

    def loglikelihood(self, X_nxp: np.array) -> np.array:

        ll = lambda X_p: sc.stats.multivariate_normal.logpdf(X_p, mean=self._parameters[0], cov=self._parameters[1])
        try:
            ll_n = np.array([ll(X_p) for X_p in X_nxp])
        except np.linalg.LinAlgError:
            print("Singular covariance matrix. Cannot evaluate log-likelihood.")
            raise np.linalg.LinAlgError
        return ll_n

    def get_parameters(self):
        return self._parameters[0], self._parameters[1]

    def set_parameters(self, value: tuple):
        if len(value) != 2:
            raise ValueError("Need to supply both the mean and covariance parameters.")
        if value[1].shape != (value[0].size, value[0].size):
            raise ValueError("Shapes of mean and covariance parameters do not match.")
        self._parameters = value

    parameters = property(get_parameters, set_parameters)

    def save(self, filename: str):
        print("Saving to {} using np.savez.".format(filename))
        parameters = self.get_parameters()
        np.savez(filename, mean_d=parameters[0], cov_dxd=parameters[1])

    def load(self, filename: str):
        d = np.load(filename)
        if 'mean_d' not in d or 'cov_dxd' not in d:
            raise ValueError('File {} is missing either the mean or covariance parameter.'.format(filename))
        self.set_parameters((d["mean_d"], d["cov_dxd"]))
