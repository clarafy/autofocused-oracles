import numpy as np
import scipy as sc
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold

def ground_truth_fn(x_n: np.array):
    return sc.stats.norm.pdf(x_n, loc=5, scale=1) + sc.stats.norm.pdf(x_n, loc=7, scale=0.5)

DEFAULT_X = np.arange(0, 10, 0.1)
GT_X = ground_truth_fn(DEFAULT_X)
DEFAULT_THRESHOLDS = np.arange(1, 101, 1)

def get_training_data(n, mean: float = 3, train_std: float = 1.6, label_std: float = 0.04, seed: int = 0):
    np.random.seed(seed)
    x_nx1 = sc.stats.norm.rvs(loc=mean, scale=train_std, size=n)[:, None]
    gt_n = ground_truth_fn(x_nx1[:, 0]).flatten()
    np.random.seed(seed + 1)
    noise_n = label_std * np.random.randn(n)
    labels_n = gt_n + noise_n
    return x_nx1, labels_n


def p0x_cond_s(x_n: np.array, oracle: KernelRidge, max_oracle_val: float, homo_std: float,
               mean: float = 3, train_std: float = 1.6):

    def p0xs(x):  # compute p_0(x, y \in S) = p_0(y \in S | x) p_0(x)
        # p_0(y \in S | x)
        p0s_cond_x = sc.stats.norm.sf(max_oracle_val, loc=oracle.predict(np.array([[x]])), scale=homo_std)
        # p_0(x)
        p0x = sc.stats.norm.pdf(x, loc=mean, scale=train_std)
        return p0s_cond_x * p0x

    result = sc.integrate.quad(p0xs, 0, 10)  # p_0(y \in S)  = \int p_0(x, y \in S) dx
    p0s = result[0]
    p0xs_n = np.array([p0xs(x) for x in x_n])
    return (p0xs_n / p0s).flatten()  # p_0(x | y \in S) = p_0(x, y \in S) / p_0(y \in S)


def fit_oracle(x_nx1: np.array, labels_n: np.array, w_n: np.array = None, k_iwcv: int = 4):
    if w_n is None:
        w_n = np.ones([labels_n.size])

    # === estimate oracle variance \sigma^2_\beta(x) via importance-weighted CV ==
    oracle = KernelRidge(kernel='rbf')
    kf = KFold(n_splits=k_iwcv, shuffle=True)
    kf.get_n_splits(x_nx1)
    oracle_var = 0.0
    for k, idx in enumerate(kf.split(x_nx1)):
        train_idx, val_idx = idx
        xtr_nx1, xval_nx1 = x_nx1[train_idx], x_nx1[val_idx]
        ytr_n, yval_n = labels_n[train_idx], labels_n[val_idx]
        wtr_n, wval_n = w_n[train_idx], w_n[val_idx]
        oracle.fit(xtr_nx1, ytr_n, sample_weight=wtr_n)
        iw_holdout_var = np.mean(wval_n * np.square(oracle.predict(xval_nx1) - yval_n))
        oracle_var += iw_holdout_var
    oracle_var /= float(k_iwcv)
    oracle_std = np.sqrt(oracle_var)

    # === estimate oracle expectation \mu_\beta(x) = E_\beta[y | x] ===
    oracle.fit(x_nx1, labels_n, sample_weight=w_n)
    return oracle, oracle_std


def evaluate_mbd_objective(searchmod_n: np.array):
    return sc.integrate.simps(searchmod_n * GT_X, DEFAULT_X)


def iterate_no_af(x_nx1: np.array, labels_n: np.array, thresholds: np.array = DEFAULT_THRESHOLDS,
                  k_iwcv: int = 5, train_std: float = 1.6, mean: float = 3):
    n_it = thresholds.size

    # === fit oracle expectation and variance ===
    oracle, oracle_std = fit_oracle(x_nx1, labels_n, w_n=None, k_iwcv=k_iwcv)
    oracle_vals_m = oracle.predict(DEFAULT_X[:, None])

    # === compute sequence of updated search models ===
    searchmod_txm = np.zeros([n_it + 1, DEFAULT_X.size])
    searchmod_txm[0] = sc.stats.norm.pdf(DEFAULT_X, loc=mean, scale=train_std)
    mbd_objective_t = np.zeros([n_it + 1])
    mbd_objective_t[0] = evaluate_mbd_objective(searchmod_txm[0])
    for t in range(n_it):
        thresh = np.percentile(oracle_vals_m, thresholds[t])
        p0x_cond_s_m = p0x_cond_s(DEFAULT_X, oracle, thresh, oracle_std, mean=mean, train_std=train_std)
        searchmod_txm[t + 1] = p0x_cond_s_m
        mbd_objective_t[t + 1] = evaluate_mbd_objective(p0x_cond_s_m)
    return searchmod_txm, mbd_objective_t


def iterate_af(x_nx1: np.array, labels_n: np.array, thresholds: np.array = DEFAULT_THRESHOLDS,
               k_iwcv: int = 5, train_std: float = 1.6, mean: float = 3):
    n_it = thresholds.size

    # === fit initial oracle expectation and variance ===
    oracle, oracle_std = fit_oracle(x_nx1, labels_n, w_n=None, k_iwcv=k_iwcv)
    oracle_vals_m = oracle.predict(DEFAULT_X[:, None])

    # === compute sequence of updated search models and oracles ===
    searchmod_txm = np.zeros([n_it + 1, DEFAULT_X.size])
    searchmod_txm[0] = sc.stats.norm.pdf(DEFAULT_X, loc=mean, scale=train_std)
    mbd_objective_t = np.zeros([n_it + 1])
    mbd_objective_t[0] = evaluate_mbd_objective(searchmod_txm[0])
    oracle_txm = np.zeros([n_it + 1, DEFAULT_X.size])
    oracle_std_t = np.zeros([n_it + 1])
    oracle_txm[0] = oracle_vals_m
    oracle_std_t[0] = oracle_std
    iw_txn = np.zeros([n_it, x_nx1.shape[0]])
    p0_n = sc.stats.norm.pdf(x_nx1.flatten(), loc=mean, scale=train_std)


    for t in range(n_it):
        # update search model
        thresh = np.percentile(oracle_vals_m, thresholds[t])
        searchmod_txm[t + 1] = p0x_cond_s(DEFAULT_X, oracle, thresh, oracle_std, mean=mean, train_std=train_std)
        p0x_cond_s_n = p0x_cond_s(x_nx1.flatten(), oracle, thresh, oracle_std, mean=mean, train_std=train_std)

        # update oracle
        w_n = p0x_cond_s_n / p0_n
        iw_txn[t] = w_n
        oracle, oracle_std = fit_oracle(x_nx1, labels_n, w_n=w_n, k_iwcv=k_iwcv)
        oracle_vals_m = oracle.predict(DEFAULT_X[:, None])
        oracle_txm[t + 1] = oracle_vals_m
        oracle_std_t[t + 1] = oracle_std
        mbd_objective_t[t + 1] = evaluate_mbd_objective(searchmod_txm[t + 1])
    return searchmod_txm, oracle_txm, oracle_std_t, iw_txn, mbd_objective_t
