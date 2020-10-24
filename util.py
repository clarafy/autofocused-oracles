"""
General utilities.
"""
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

# tuned xgboost hyperparameters from Hamidieh (2018)
XGB_PARAMS = {
    "n_estimators": 200,
    'eta': 0.02,
    'max_depth': 16,
    "colsample_bytree": 0.5,
    "subsample": 0.5,
    "objective": "reg:squarederror",
    "eval_metric": ["rmse", "mae"],
    "random_state": 0,
    "verbose": 2
}

# ===== general utilities =====

def ess(lrs_xn):
    # lrs_xn = lrs_xn / np.sum(lrs_xn, axis=-1, keepdims=True)
    denom = np.sum(np.square(lrs_xn), axis=-1)
    numer = np.square(np.sum(lrs_xn, axis=-1))
    result = np.zeros(numer.shape)
    if denom.size == 1:
        print("Computing ESS: {} / {}".format(numer, denom))
    return 0 if denom == 0  else numer / denom
    result[denom > 0] = numer[denom > 0] / denom[denom > 0]
    return result

def get_data_below_percentile(X_nxp: np.array, y_n: np.array, percentile: float, n_sample: int = None, seed: int = None):
    perc = np.percentile(y_n, percentile)
    idx = np.where(y_n <= perc)[0]
    print("Max label in training data: {:.1f}. 80-th percentile label: {:.1f}".format(np.max(y_n), perc))
    if n_sample is not None and n_sample < idx.size:
        np.random.seed(seed)
        idx = np.random.choice(idx, size=n_sample, replace=False)
    Xbelow_nxm = X_nxp[idx]evonia
    ybelow_n = y_n[idx]
    return Xbelow_nxm, ybelow_n, idx

def get_promising_candidates(oracle_m: np.array, gt_m: np.array, percentile: float = 80):
    oracle_percentile = np.percentile(oracle_m, percentile)
    candidate_idx = np.where(oracle_m >= oracle_percentile)[0]
    return oracle_m[candidate_idx], gt_m[candidate_idx], oracle_percentile

def evaluate_top_candidates(o_m: np.array, gt_m: np.array, gt0_n: np.array, percentile: float):
    o_cand, gt_cand, o_perc = get_promising_candidates(o_m, gt_m, percentile=percentile)
    rho_and_p = sc.stats.spearmanr(gt_m, o_m)
    rmse = np.sqrt(np.mean(np.square(gt_m - o_m)))
    pci = 100 * np.sum(gt_cand > np.max(gt0_n)) / float(gt_cand.size)
    return o_perc, np.median(gt_cand), np.max(gt_cand), pci, rho_and_p, rmse

def score_top_candidates(o_txm: np.array, gt_txm: np.array, gt0_n: np.array,
                         oaf_txm: np.array, gtaf_txm: np.array, gt0af_n: np.array, percentile: float):
    scores_tx = np.zeros([o_txm.shape[0], 5])
    scoresaf_tx = np.zeros([oaf_txm.shape[0], 5])
    operc_t = np.zeros([o_txm.shape[0]])
    oafperc_t = np.zeros([o_txm.shape[0]])
    for t in range(o_txm.shape[0]):
        o_perc, gt_med, gt_max, pci, rho_and_p, rmse = evaluate_top_candidates(o_txm[t], gt_txm[t], gt0_n, percentile)
        scores_tx[t] = np.array([gt_med, gt_max, pci, rho_and_p[0], rmse])
        operc_t[t] = o_perc
    for t in range(oaf_txm.shape[0]):
        oaf_perc, gtaf_med, gtaf_max, pciaf, rhoaf_and_p, rmseaf = evaluate_top_candidates(
            oaf_txm[t], gtaf_txm[t], gt0af_n, percentile)
        scoresaf_tx[t] = np.array([gtaf_med, gtaf_max, pciaf, rhoaf_and_p[0], rmseaf])
        oafperc_t[t] = oaf_perc
    t_max = np.argmax(operc_t)
    taf_max = np.argmax(oafperc_t)
    return scores_tx[t_max], scoresaf_tx[taf_max], t_max, taf_max

def compare_af(scores_trx: np.array, scoresaf_trx: np.array):
    mean_diffs = np.mean(scoresaf_trx - scores_trx, axis=0)
    p_values = [sc.stats.wilcoxon(scoresaf_trx[:, i], scores_trx[:, i])[1] for i in range(scores_trx.shape[1])]
    formatted_scores = ["{:.2f}".format(val) for val in np.mean(scores_trx, axis=0)]
    formatted_scoresaf = ["{:.2f}".format(val) for val in np.mean(scoresaf_trx, axis=0)]
    formatted_diffs = ["{:.2f}".format(val) for val in mean_diffs]
    formatted_p = ["{:.4f}".format(val) for val in p_values]
    print("            GT Median  |  GT Max  |  PCI  |  rho  |  RMSE")
    print("Original    {:<14}{:<11}{:<8}{:<8}{:<6}".format(*formatted_scores))
    print("Autofocused {:<14}{:<11}{:<8}{:<8}{:<6}".format(*formatted_scoresaf))
    print("Mean Diff.  {:<14}{:<11}{:<8}{:<8}{:<6}".format(*formatted_diffs))
    print("p-value     {:<14}{:<11}{:<8}{:<8}{:<6}".format(*formatted_p))

def iw_rmse(y1_n: np.array, y2_n: np.array, w_n: np.array = None, self_normalize=False):
    if w_n is None:
        w_n = np.ones((y1_n.size))
    if self_normalize:
        rmse = np.sqrt(np.sum(w_n * np.square(y1_n - y2_n)) / np.sum(w_n))
    else:
        rmse = np.sqrt(np.mean(w_n * np.square(y1_n - y2_n)))
    return rmse

def rmse(x1_n: np.array, x2_n: np.array):
    return np.sqrt(np.mean(np.square(x1_n - x2_n)))


# ===== plotting =====

def plot_xy(x_n: np.array, y_n: np.array, color: str = "orange", alpha: float = 0.9):
    vmin = np.min([np.min(x_n), np.min(y_n)])
    vmax = np.max([np.max(x_n), np.max(y_n)])
    plt.plot([vmin, vmax], [vmin, vmax], "--", c=color, alpha=alpha);
