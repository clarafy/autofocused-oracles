import time
from abc import ABC, abstractmethod

import numpy as np
import scipy as sc

from oracles import Oracle
from searchmodels import SearchModel, MultivariateGaussian
from util import get_promising_candidates

TOL = 1e-8

def get_importance_weights(X_nxp: np.array, init_searchmodel: SearchModel, searchmodel: SearchModel,
                           z_nxl: np.array = None, iw_alpha: float = 0.2):
    """
    Get importance weights of a dataset.
    :param X_nxp: np.array of data (covariates only)
    :param init_searchmodel: SearchModel, initial search model
    :param searchmodel: SearchModel, other search model
    :param z_nxl: np.array of latent z ~ p(z | x), if the searchmodels are VAEs
    :param iw_alpha: float, hyperparameter for flattening importance weights to control variance
    :return weights_n: np.array of importance weights
    """
    if z_nxl is None:
        try:
            ll0_n = init_searchmodel.loglikelihood(X_nxp)
            llt_n = searchmodel.loglikelihood(X_nxp)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError
    else:
        ll0_n = init_searchmodel.logpxcondz(X_nxp, z_nxl)
        llt_n = searchmodel.logpxcondz(X_nxp, z_nxl)
    iw_n = np.exp(llt_n - ll0_n)
    weights_n = np.power(iw_n, iw_alpha)
    return weights_n


class MBOAlgorithmTrajectory(object):

    def __init__(self, n_iter: int, n_sample: int, n_train: int, update_strf: str = "{:.1f}"):
        self.oracle_txm = np.zeros([n_iter, n_sample])
        self.gt_txm = np.zeros([n_iter, n_sample])
        self.iw_txn = np.zeros([n_iter, n_train])
        self.mboweights_txm = np.zeros([n_iter, n_sample])
        self.se_txm = np.zeros([n_iter, n_sample])
        self.selected_alpha_t = -1 * np.ones(n_iter)
        self.oracle0_n = None
        self.gt0_n = None
        self.last_iter = 0
        self.update_strf = update_strf

    def record_initialization(self, X_nxp: np.array, gt_n: np.array, oracle: Oracle, verbose: bool = False):
        self.oracle0_n = oracle.predict(X_nxp)[0]
        self.gt0_n = gt_n
        max_idx = np.argmax(gt_n)
        oracle_gt_max = oracle.predict(X_nxp[max_idx, :][None, :])[0][0]
        if verbose:
            print('Initialization. Oracle 80-th percentile: {:.2f}. Ground truth median/max in training data: {:.2f}, {:.2f}'.format(
                np.percentile(self.oracle0_n, 80), np.median(self.gt0_n), np.max(self.gt0_n)
            ))
            print('Oracle value of ground-truth max: {:.2f}'.format(oracle_gt_max))

    def final_candidates(self, percentile: int = 80):
        tmp_t = [get_promising_candidates(o_m, gt_m, percentile=percentile)
                 for o_m, gt_m in zip(self.oracle_txm, self.gt_txm)]
        ocand_t, gtcand_t, operc_t = list(zip(*tmp_t))
        return ocand_t, gtcand_t, operc_t

    def update(self, t: int, o_m: np.array, gt_m: np.array, iw_n: np.array, mboweights_m: np.array,
               se_m: np.array, selected_alpha: float, percentile: float = 80, verbose: bool = False):
        self.oracle_txm[t] = o_m
        self.gt_txm[t] = gt_m
        self.iw_txn[t] = iw_n
        self.mboweights_txm[t] = mboweights_m
        self.se_txm[t] = se_m
        self.last_iter = t + 1
        # self.selected_alpha_t[t] = selected_alpha

        if verbose:
            update_str = 'Iter {{}}. Oracle {{}}-th percentile: {}. Ground-truth median/max of top candidates: {}, {}'.format(
                self.update_strf, self.update_strf, self.update_strf)
            o_cand, gt_cand, o_perc = get_promising_candidates(o_m, gt_m, percentile=percentile)
            print(update_str.format(t, percentile, o_perc, np.median(gt_cand), np.max(gt_cand)))

    def save(self, filename: str):
        print("Saving trajectory data to {}".format(filename))
        np.savez(filename, oracle_txm=self.oracle_txm, gt_txm=self.gt_txm, iw_txn=self.iw_txn,
                 mboweights_txm=self.mboweights_txm, se_txm=self.se_txm, selected_alpha_t=self.selected_alpha_t,
                 oracle0_n=self.oracle0_n, gt0_n=self.gt0_n, last_iter=self.last_iter)

class ModelBasedOptimizationAlgorithm(ABC):

    @abstractmethod
    def get_mle_weights(self, Xt_mxp: np.array, ot_m: np.array, ovart_m: np.array, init_searchmodel: SearchModel,
                        searchmodel: SearchModel, aux = None):
        raise NotImplementedError

    def run(self, X_nxp: np.array, gt_n: np.array, ground_truth_fn, init_oracle: Oracle, init_searchmodel: SearchModel,
            autofocus: bool = False, iw_alpha: float = 0.2, n_iter: int = 20, n_sample: int = 10000,
            ztr_nxl: np.array = None, update_epochs: int = 10, k_iwcv: int = 1,
            update_strf: str = "{:.1f}", verbose: bool = True, autofocus_verbose: bool = False):

        if len(X_nxp.shape) == 2:
            n_train, n_feat = X_nxp.shape
        elif len(X_nxp.shape) == 3:
            n_train, seq_len, alphabet_size = X_nxp.shape
            n_feat = seq_len * alphabet_size

        trajectory = MBOAlgorithmTrajectory(n_iter=n_iter, n_sample=n_sample, n_train=n_train, update_strf=update_strf)
        trajectory.record_initialization(X_nxp, gt_n, init_oracle, verbose=verbose)
        searchmodel = init_searchmodel.__class__(**init_searchmodel.get_initialization_kwargs())  # TODO: same init kwargs?
        searchmodel.set_parameters(init_searchmodel.get_parameters())
        oracle = init_oracle.__class__(**init_oracle.get_initialization_kwargs())  # TODO: same init kwargs?
        oracle.set_parameters(init_oracle.get_parameters())

        for t in range(n_iter):
            iter_start_time = time.time()

            # === sample the current search model and evaluate ground truth ===
            Xt_mxp, auxiliary_var = searchmodel.sample(n_sample)
            gtt_m = ground_truth_fn(Xt_mxp)

            # === retrain oracle, if autofocusing ===
            iw_n = np.ones([n_train])
            selected_alpha = iw_alpha
            if t > 0 and autofocus:
                # TODO: double-check integrated ztr_nxl correctly throughout run()
                try:
                    iw_n = get_importance_weights(X_nxp, init_searchmodel, searchmodel, iw_alpha=selected_alpha, z_nxl=ztr_nxl)
                    keep_idx = np.where((~np.isnan(iw_n)) & (~np.isinf(iw_n)))[0]
                    # print(keep_idx.size)
                    ss = np.sum(np.square(iw_n[keep_idx]))
                    if ss >= TOL:
                        oracle.fit(X_nxp[keep_idx], gt_n[keep_idx], iw_n[keep_idx], seed=t, verbose=2 if autofocus_verbose else 0)
                    else:
                        print("Importance weights near zero (SS = {}). Not retraining oracle.".format(ss))
                except np.linalg.LinAlgError:
                    print("Singular matrix while computing importance weights. Not retraining oracle.".format(ss))

            # === evaluate oracle ===
            # oracle predicts both mean E[y | x] and variance Var[y | x]
            ot_m, ovart_m = oracle.predict(Xt_mxp)
            set_m = np.square(ot_m - gtt_m)

            # === get MLE/re-fitting weights for updating search model ===
            try:
                mboweights_m = self.get_mle_weights(Xt_mxp, ot_m, ovart_m, init_searchmodel, searchmodel, aux=auxiliary_var)
            except np.linalg.LinAlgError:
                print("Iter {}. Could not compute log-likelihood due to LinAlgError.".format(t))
                trajectory.last_iter = t
                return trajectory

            # === record progress ===
            trajectory.update(t, ot_m, gtt_m, iw_n, mboweights_m, set_m, selected_alpha, verbose=verbose)

            # === update search model ===
            # discard weights that are unreliably small
            keep_idx = np.where(mboweights_m >= TOL)[0]
            if keep_idx.size == 0:
                print("Iter {}. No sample weights surpassed {}.".format(t, TOL))
                trajectory.last_iter = t
                return trajectory
            # if verbose:
            #     print("{} samples with non-negligible weights.".format(keep_idx.size))
            if t < n_iter - 1:
                self.update_searchmodel(searchmodel, Xt_mxp[keep_idx], mboweights_m[keep_idx], update_epochs=update_epochs)

        return trajectory

    def update_searchmodel(self, searchmodel: SearchModel, Xt_mxp: np.array, mboweights_m: np.array,
                           update_epochs: int = 10):
        searchmodel.fit(Xt_mxp, weights=mboweights_m)


class ConditioningByAdaptiveSampling(ModelBasedOptimizationAlgorithm):
    def __init__(self, quantile: float = 0.9):
        self.s_threshold = -np.inf
        self.quantile = quantile

    def get_mle_weights(self, Xt_mxp: np.array, ot_m: np.array, ovart_m: np.array, init_searchmodel: SearchModel,
                        searchmodel: SearchModel, aux: np.array):
        if aux is None:
            try:
                llt_m = searchmodel.loglikelihood(Xt_mxp)
                ll0_m = init_searchmodel.loglikelihood(Xt_mxp)
            except np.linalg.LinAlgError:
                raise np.linalg.LinAlgError
        else:
            zt_mxp = np.copy(aux)
            ll0_m = init_searchmodel.logpxcondz(Xt_mxp, zt_mxp)
            llt_m = searchmodel.logpxcondz(Xt_mxp, zt_mxp)
        weights1_n = np.exp(ll0_m - llt_m)
        o_percentile = np.percentile(ot_m, self.quantile * 100)
        if o_percentile > self.s_threshold:
            self.s_threshold = o_percentile
        weights2_n = sc.stats.norm.sf(self.s_threshold, loc=ot_m, scale=np.sqrt(ovart_m))
        cbas_weights_n = weights1_n * weights2_n
        return cbas_weights_n


class DesignByAdaptiveSampling(ModelBasedOptimizationAlgorithm):
    def __init__(self, quantile: float = 0.9):
        self.s_threshold = -np.inf
        self.quantile = quantile

    def get_mle_weights(self, Xt_mxp: np.array, ot_m: np.array, ovart_m: np.array, init_searchmodel: SearchModel,
                        searchmodel: SearchModel, aux = None):
        o_percentile = np.percentile(ot_m, self.quantile * 100)
        if o_percentile > self.s_threshold:
            self.s_threshold = o_percentile
        dbas_weights_n = sc.stats.norm.sf(self.s_threshold, loc=ot_m, scale=np.sqrt(ovart_m))
        return dbas_weights_n


class RewardWeightedRegression(ModelBasedOptimizationAlgorithm):
    def __init__(self, gamma: float = 0.01):
        self.gamma = gamma

    def get_mle_weights(self, Xt_mxp: np.array, ot_m: np.array, ovart_m: np.array, init_searchmodel: SearchModel,
                        searchmodel: SearchModel, aux = None):
        rwr_weights_n = np.exp(self.gamma * ot_m)
        rwr_weights_n /= np.sum(rwr_weights_n)
        return rwr_weights_n


class FeedbackMechanism(ModelBasedOptimizationAlgorithm):
    def __init__(self, X_nxp: np.array, gt_n: np.array, quantile: float = 0.9):
        self.quantile = quantile
        self.fb_threshold = np.percentile(gt_n, quantile * 100)
        self.Xfb_nxp = np.copy(X_nxp)
        self.n = X_nxp.shape[0]

    def get_mle_weights(self, Xt_mxp: np.array, ot_m: np.array, ovart_m: np.array, init_searchmodel: SearchModel,
                        searchmodel: SearchModel, aux = None):
        fb_weights_m = (ot_m >= self.fb_threshold).astype(int)
        return fb_weights_m

    def update_searchmodel(self, searchmodel: SearchModel, Xt_mxp: np.array, mboweights_m: np.array,
                           update_epochs: int = 10):
        Xkeep_mxp = Xt_mxp[mboweights_m > 0]
        n_keep = Xkeep_mxp.shape[0]
        if n_keep < self.n:
            self.Xfb_nxp = np.vstack([Xkeep_mxp, self.Xfb_nxp[: -n_keep]])
        else:
            idx = np.random.permutation(n_keep)[: self.n]
            self.Xfb_nxp = Xkeep_mxp[idx]
        searchmodel.fit(self.Xfb_nxp)


class CrossEntropyMethodWithProbabilityOfImprovement(ModelBasedOptimizationAlgorithm):
    def __init__(self, max_gt_train: float, quantile: float = 0.9):
        self.quantile = quantile
        self.max_gt_train = max_gt_train

    def get_mle_weights(self, Xt_mxp: np.array, ot_m: np.array, ovart_m: np.array, init_searchmodel: SearchModel,
                        searchmodel: SearchModel, aux = None):
        probimprove_m = sc.stats.norm.sf(self.max_gt_train, loc=ot_m, scale=np.sqrt(ovart_m))
        threshold = np.percentile(probimprove_m, self.quantile * 100)
        cempi_weights_m = (probimprove_m >= threshold).astype(int)
        return cempi_weights_m
