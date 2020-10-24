from abc import ABC, abstractmethod
import os
from copy import deepcopy

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold

from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.utils.generic_utils import get_custom_objects

TOL = 1e-8

def neg_log_likelihood(truth_n, pred_nx2):
    truth_n = truth_n[:, 0]
    mean_n = pred_nx2[:, 0]
    var_n = K.softplus(pred_nx2[:, 1]) + 1e-6
    logvar_n = K.log(var_n)
    nll_n = 0.5 * K.mean(logvar_n, axis=-1) + 0.5 * K.mean(K.square(truth_n - mean_n) / var_n, axis=-1) + \
            0.5 * K.log(2 * np.pi)
    return nll_n
get_custom_objects().update({"neg_log_likelihood": neg_log_likelihood})

class Oracle(ABC):
    @abstractmethod
    def predict(self, X_nxp: np.array):
        raise NotImplementedError

    @abstractmethod
    def fit(self, X_nxp: np.array, gt_n: np.array, weights_n: np.array = None):
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self):
        raise NotImplementedError

    @abstractmethod
    def set_parameters(self, value):
        raise NotImplementedError

    parameters = property(get_parameters, set_parameters)

    @abstractmethod
    def get_initialization_kwargs(self):
        raise NotImplementedError

    @abstractmethod
    def save(self, savepath: str):
        raise NotImplementedError

class KernelRidgeRegression(Oracle):
    def __init__(self, kernel = 'rbf'):
        self.kernel = kernel
        self.model = KernelRidge(alpha=1, kernel=kernel, gamma=None, degree=5, coef0=1, kernel_params=None)

    def predict(self, X_nxp: np.array):
        return self.model.predict(X_nxp), self.oracle_std * np.ones((X_nxp.shape[0]))

    def fit(self, X_nxp: np.array, gt_n: np.array, weights_n: np.array = None, k_estimate_var: int = 4,
            epochs: int = None, seed: int = None, verbose: bool = False):
        if weights_n is None:
            weights_n = np.ones([gt_n.size])

        # ------ fit oracle variance -----
        kf = KFold(n_splits=k_estimate_var, shuffle=True)
        kf.get_n_splits(X_nxp)
        oracle_var = 0.0
        for k, idx in enumerate(kf.split(X_nxp)):
            train_idx, val_idx = idx
            xtr_nx1, xval_nx1 = X_nxp[train_idx], X_nxp[val_idx]
            ytr_n, yval_n = gt_n[train_idx], gt_n[val_idx]
            wtr_n, wval_n = weights_n[train_idx], weights_n[val_idx]
            self.model.fit(xtr_nx1, ytr_n, sample_weight=wtr_n)
            oracle_var += np.mean(wval_n * np.square(self.model.predict(xval_nx1) - yval_n))
        oracle_var /= float(k_estimate_var)
        oracle_std = np.sqrt(oracle_var)
        self.oracle_std = oracle_std
        self.model.fit(X_nxp, gt_n, sample_weight=weights_n)

    def get_parameters(self):
        return self.model, self.oracle_std
    def set_parameters(self, value):
        self.model = deepcopy(value[0])
        self.oracle_std = value[1]

    parameters = property(get_parameters, set_parameters)

    def get_initialization_kwargs(self):
        return {'kernel': self.kernel}
    def save(self, savepath: str):
        print("Not saving KernelRidgeRegression.")


def build_nn(input_dim: int = 10, hidden_units: tuple = None, act='elu'):
    x = Input(shape=(input_dim,))
    y = Dense(hidden_units[0], activation=act)(x)
    for n_units in hidden_units[1:]:
        y = Dense(n_units, activation=act)(y)
    y = Dense(2, activation='linear')(y)
    model = Model(inputs=x, outputs=y)
    return model

class DeepEnsemble(Oracle):
    # from Lakshiminarayanan et al., "Simple and Scalable Predictive Uncertainty Estimation Using
    # Deep Ensembles", NeurIPS 2017
    def __init__(self, input_dim: int = 10, hidden_units: tuple = None, n_nn: int = 3):
        if hidden_units is None:
            hidden_units = (100, 10)
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.n_nn = n_nn
        self.oracle = [build_nn(input_dim, hidden_units) for _ in range(n_nn)]

    def get_initialization_kwargs(self):
        kwargs = {
            'input_dim': self.input_dim,
            'hidden_units': self.hidden_units,
            'n_nn': self.n_nn,
        }
        return kwargs


    def predict(self, X_nxp: np.array):
        if len(X_nxp.shape) == 3:
            n, seq_len, alphabet_size = X_nxp.shape
            X_nxp = np.reshape(X_nxp, [n, seq_len * alphabet_size], order='C')
        # using notation from Lakshminarayanan et al. (2017)
        n = X_nxp.shape[0]
        M = self.n_nn
        mean_oxn = np.zeros((M, n))
        var_oxn = np.zeros((M, n))
        for o_idx in range(M):
            pred_nx2 = self.oracle[o_idx].predict(X_nxp)
            mean_oxn[o_idx, :] = pred_nx2[:, 0]
            var_n = np.log(1 + np.exp(np.fmin(pred_nx2[:, 1], 10000), dtype=np.float128), dtype=np.float128) + 1e-6
            var_oxn[o_idx, :] = var_n
        mu_star_n = np.mean(mean_oxn, axis=0)
        var_star_n = (1.0 / M) * (np.sum(var_oxn, axis=0) + np.sum(np.square(mean_oxn), axis=0)) - np.square(mu_star_n)
        return mu_star_n, var_star_n

    def fit(self, X_nxp: np.array, gt_n: np.array, weights_n: np.array = None,
            lr: float = 5e-4, n_epochs: int = 2000, batch_size: int = 100, early_stop_patience: int = 2,
            seed: int = 0, verbose: int = 2):
        if len(X_nxp.shape) == 3:
            n, seq_len, alphabet_size = X_nxp.shape
            X_nxp = np.reshape(X_nxp, [n, seq_len * alphabet_size], order='C')

        if weights_n is None:
            weights_n = np.ones([gt_n.size])
        early_stop = EarlyStopping(monitor='val_loss', patience=early_stop_patience, verbose=verbose)
        optimizer = Adam(lr=lr)

        n_train = int(0.8 * gt_n.size)
        n_val = gt_n.size - n_train

        for o_idx, o in enumerate(self.oracle):
            np.random.seed(seed + o_idx)
            idx = np.random.permutation(gt_n.size)
            Xshuf_nxp = X_nxp[idx]
            gtshuf_n = gt_n[idx]
            wshuf_n = weights_n[idx]
            Xtrain_nxp, gttrain_n, wtrain_n = Xshuf_nxp[: n_train], gtshuf_n[: n_train], wshuf_n[: n_train]
            Xval_nxp, gtval_n, wval_n = Xshuf_nxp[n_train :], gtshuf_n[n_train :], wshuf_n[n_train :]

            if np.sum(np.square(wtrain_n)) and np.sum(np.square(wval_n)) >= TOL:
                o.compile(optimizer=optimizer, loss=neg_log_likelihood)
                o.fit(Xtrain_nxp, gttrain_n,
                      sample_weight=n_train * wtrain_n / np.sum(wtrain_n),
                      epochs=n_epochs, batch_size=batch_size,
                      validation_data=(Xval_nxp, gtval_n, n_val * wval_n / np.sum(wval_n)),
                      callbacks=[early_stop],
                      verbose=verbose
                      )
            else:
                print("Train or validation importance weights near zero. Not training model {}.".format(o_idx))

    def get_parameters(self):
        return [o.get_weights() for o in self.oracle]

    def set_parameters(self, value):
        if len(value) != self.n_nn:
            raise ValueError("Need to supply parameters for all {} neural networks.".format(self.n_nn))
        for o_idx, o in enumerate(self.oracle):
            o.set_weights(value[o_idx])

    parameters = property(get_parameters, set_parameters)

    def save(self, savepath: str, prefix: str = None):
        if prefix is None:
            for o_idx, o in enumerate(self.oracle):
                o.save(os.path.join(savepath, 'model{}.h5'.format(o_idx)))
        else:
            for o_idx, o in enumerate(self.oracle):
                o.save(os.path.join(savepath, '{}_model{}.h5'.format(prefix, o_idx)))

    def load(self, loadpath: str, prefix: str = None):
        if prefix is None:
            self.oracle = [load_model(os.path.join(loadpath, 'model{}.h5'.format(i))) for i in range(self.n_nn)]
        else:
            self.oracle = [load_model(os.path.join(loadpath, '{}_model{}.h5'.format(prefix, i))) for i in range(self.n_nn)]

