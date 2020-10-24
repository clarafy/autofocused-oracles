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
               se_m: np.array, selected_alpha: float, elapsed: float = None, percentile: float = 80, verbose: bool = False):
        self.oracle_txm[t] = o_m
        self.gt_txm[t] = gt_m
        self.iw_txn[t] = iw_n
        self.mboweights_txm[t] = mboweights_m
        self.se_txm[t] = se_m
        self.last_iter = t + 1

        if verbose:
            update_str = 'Iter {{}}. Oracle {{}}-th percentile: {}. Ground-truth median/max of top candidates: {}, {}. {} s'.format(
                self.update_strf, self.update_strf, self.update_strf, self.update_strf)
            o_cand, gt_cand, o_perc = get_promising_candidates(o_m, gt_m, percentile=percentile)
            print(update_str.format(t, percentile, o_perc, np.median(gt_cand), np.max(gt_cand), elapsed))

    def save(self, filename: str):
        print("Saving trajectory data to {}".format(filename))
        np.savez(filename, oracle_txm=self.oracle_txm, gt_txm=self.gt_txm, iw_txn=self.iw_txn,
                 mboweights_txm=self.mboweights_txm, se_txm=self.se_txm, selected_alpha_t=self.selected_alpha_t,
                 oracle0_n=self.oracle0_n, gt0_n=self.gt0_n, last_iter=self.last_iter)

class ModelBasedOptimizationAlgorithm(ABC):

    @abstractmethod
    def get_mle_weights(self, Xt_mxp: np.array, ot_m: np.array, ovart_m: np.array, init_searchmodel: SearchModel,
                        searchmodel: SearchModel):
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
        searchmodel = init_searchmodel.__class__(**init_searchmodel.get_initialization_kwargs())
        searchmodel.set_parameters(init_searchmodel.get_parameters())
        oracle = init_oracle.__class__(**init_oracle.get_initialization_kwargs())
        oracle.set_parameters(init_oracle.get_parameters())

        for t in range(n_iter):
            iter_start_time = time.time()

            # === sample the current search model and evaluate ground truth ===
            Xt_mxp = searchmodel.sample(n_sample)
            gtt_m = ground_truth_fn(Xt_mxp)

            # === retrain oracle, if autofocusing ===
            iw_n = np.ones([n_train])
            selected_alpha = iw_alpha
            if t > 0 and autofocus:
                # TODO: double-check integrated ztr_nxl correctly throughout run()
                try:
                    iw_n = get_importance_weights(X_nxp, init_searchmodel, searchmodel, iw_alpha=selected_alpha, z_nxl=ztr_nxl)
                    keep_idx = np.where((~np.isnan(iw_n)) & (~np.isinf(iw_n)))[0]
                    ss = np.sum(np.square(iw_n[keep_idx]))
                    if ss >= TOL:
                        oracle.fit(X_nxp[keep_idx], gt_n[keep_idx], iw_n[keep_idx], seed=t, verbose=2 if autofocus_verbose else 0)
                    else:
                        print("Importance weights near zero (SS = {}). Not retraining oracle.".format(ss))
                except np.linalg.LinAlgError:
                    print("Singular matrix while computing importance weights. Not retraining oracle.")

            # === evaluate oracle ===
            # oracle predicts both mean E[y | x] and variance Var[y | x]
            ot_m, ovart_m = oracle.predict(Xt_mxp)
            set_m = np.square(ot_m - gtt_m)

            # === get MLE/re-fitting weights for updating search model ===
            try:
                mboweights_m = self.get_mle_weights(Xt_mxp, ot_m, ovart_m, init_searchmodel, searchmodel)
            except np.linalg.LinAlgError:
                print("Iter {}. Could not compute log-likelihood due to LinAlgError.".format(t))
                trajectory.last_iter = t
                return trajectory

            # === record progress ===
            trajectory.update(t, ot_m, gtt_m, iw_n, mboweights_m, set_m, selected_alpha,
                              elapsed=time.time() - iter_start_time, verbose=verbose)

            # === update search model ===
            # discard weights that are unreliably small
            keep_idx = np.where(mboweights_m >= TOL)[0]
            if keep_idx.size == 0:
                print("Iter {}. No sample weights surpassed {}.".format(t, TOL))
                trajectory.last_iter = t
                return trajectory
            if t < n_iter - 1:
                self.update_searchmodel(searchmodel, Xt_mxp[keep_idx], mboweights_m[keep_idx], update_epochs=update_epochs)

        return trajectory

    def update_searchmodel(self, searchmodel: SearchModel, Xt_mxp: np.array, mboweights_m: np.array,
                           update_epochs: int = 10):
        searchmodel.fit(Xt_mxp, weights=mboweights_m)


class CovarianceMatrixAdaptationEvolutionStrategy(object):
    # transcribed from the MATLAB source provided on pages 36-37 of Hansen's CMA-ES review.
    # wherever possible, same notation used.
    def __init__(self, max_gt_train: float, sigma: float = 0.5, use_probimp: bool = False):
        print("Reminder: CovarianceMatrixAdaptationEvolutionStrategy uses the MultivariateGaussian search model.")
        self.max_gt_train = max_gt_train
        self.sigma = sigma  # coordinate step size
        self.use_probimp = use_probimp

    def run(self, X_nxp: np.array, gt_n: np.array, ground_truth_fn, init_oracle: Oracle, init_searchmodel: SearchModel,
            autofocus: bool = False, iw_alpha: float = 0.2, n_iter: int = 20, n_sample: int = 10000,
            verbose: bool = True, autofocus_verbose: bool = False):
        if  init_searchmodel.__class__ is not MultivariateGaussian:
            raise ValueError("CMA-ES only uses a MultivariateGaussian search model.")

        n_train, n_feat = X_nxp.shape
        trajectory = MBOAlgorithmTrajectory(n_iter=n_iter, n_sample=n_sample, n_train=n_train)
        trajectory.record_initialization(X_nxp, gt_n, init_oracle, verbose=verbose)
        # meaning of subscript n changes from here on, dimension rather than n_train
        searchmodel = MultivariateGaussian(**init_searchmodel.get_initialization_kwargs())
        oracle = init_oracle.__class__(**init_oracle.get_initialization_kwargs())
        oracle.set_parameters(init_oracle.get_parameters())

        # selection hyperparameters
        lmbda = n_sample  # to keep notation below consistent with Hansen review
        mu = lmbda / 2.0
        weights_mu = np.log(mu + 0.5) - np.log(1 + np.arange(int(mu)))
        weights_mu = weights_mu / np.sum(weights_mu)
        mu = int(mu)  # number of parents for recombination
        mueff = np.square(np.sum(weights_mu)) / np.sum(np.square(weights_mu))  # variance-effective size of mu

        # adaptation hyperparameters
        cc = (4 + mueff / n_feat) / (n_feat + 4 + 2 * mueff / n_feat)  # time constant for C accumulation
        cs = (mueff + 2) / (n_feat + mueff + 5)  # time constant for sigma control
        c1 = 2 / (np.square(n_feat + 1.3) + mueff)  # learning rate for rank-one update of C
        cmu = 2 * (mueff - 2 + 1 / mueff) / (np.square(n_feat + 2) + 2 * mueff / 2)  # ditto for rank-mu update
        damps = 1 + 2 * np.fmax(0, np.sqrt((mueff - 1) / (n_feat + 1)) - 1) + cs  # damping for sigma

        # initialize dynamic (internal) strategy parameters
        pc_n = np.zeros((n_feat))  # evolution paths for C and sigma
        ps_n = np.zeros((n_feat))
        chiN = np.power(n_feat, 0.5) * (1 - 1 / (4 * n_feat) + 1 / (21 * np.square(n_feat)))

        xmean_n, C_nxn = init_searchmodel.get_parameters() # np.random.randn(n_feat)
        w_n, B_nxn = np.linalg.eigh(C_nxn)
        D_nxn = np.diag(np.sqrt(w_n))

        for t in range(n_iter):
            iter_start_time = time.time()

            # === sample the current search model and evaluate ground truth ===
            arz_nxl = np.random.randn(n_feat, lmbda)
            arx_nxl = xmean_n[:, None] + self.sigma * np.dot(np.dot(B_nxn, D_nxn), arz_nxl)
            gtt_l = ground_truth_fn(arx_nxl.T)

            # === retrain oracle, if autofocusing ===
            iw_n = np.ones([n_train])
            if t > 0 and autofocus:
                try:
                    iw_n = get_importance_weights(X_nxp, init_searchmodel, searchmodel, z_nxl=None, iw_alpha=iw_alpha)
                    keep_idx = np.where((~np.isnan(iw_n)) & (~np.isinf(iw_n)))[0]
                    ss = np.sum(np.square(iw_n[keep_idx]))
                    if ss >= TOL:
                        oracle.fit(X_nxp[keep_idx], gt_n[keep_idx], iw_n[keep_idx], seed=t, verbose=2 if autofocus_verbose else 0)
                    else:
                        print("Importance weights near zero (SS = {}). Not retraining oracle.".format(ss))
                except np.linalg.LinAlgError:
                    print("Singular matrix while computing importance weights. Not retraining oracle.")

            # === evaluate oracle ===
            # oracle predicts both mean E[y | x] and variance Var[y | x]
            ot_l, ovart_l = oracle.predict(arx_nxl.T)
            set_l = np.square(ot_l - gtt_l)

            # === record progress ===
            trajectory.update(t, ot_l, gtt_l, iw_n, np.zeros((lmbda)), set_l, iw_alpha,
                              elapsed=time.time() - iter_start_time, verbose=verbose)

            # === update mean ===
            if self.use_probimp:
                probimp_l = sc.stats.norm.sf(self.max_gt_train, loc=ot_l, scale=np.sqrt(ovart_l))
                argsort_l = np.argsort(probimp_l)[::-1]
            else:
                argsort_l = np.argsort(ot_l)[::-1]
            xmean_n = np.dot(arx_nxl[:, argsort_l[: mu]], weights_mu)
            zmean_n = np.dot(arz_nxl[:, argsort_l[: mu]], weights_mu)

            # === update evolution paths ===
            ps_n = (1 - cs) * ps_n + np.sqrt(cs * (2 - cs) * mueff) * np.dot(B_nxn, zmean_n)
            hsig = np.linalg.norm(ps_n) / np.sqrt(1 - np.power(1 - cs, 2 * (t + 1) / lmbda)) / chiN < 1.4 + 2 / (n_feat + 1)
            pc_n = (1 - cc) * pc_n + hsig * np.sqrt(cc * (2 - cc) * mueff) * np.dot(np.dot(B_nxn, D_nxn), zmean_n)

            # === adapt covariance matrix ===
            term1 = (1 - c1 - cmu) * C_nxn
            term2 = c1 * np.outer(pc_n, pc_n) + (1.0 - hsig) * cc * C_nxn
            bdz_nxmu = np.dot(np.dot(B_nxn, D_nxn), arz_nxl[:, argsort_l[: mu]])
            term3 = cmu * np.dot(np.dot(bdz_nxmu, np.diag(weights_mu)), bdz_nxmu.T)
            C_nxn = term1 + term2 + term3

            # === adapt step size ===
            self.sigma = self.sigma * np.exp((cs / damps) * (np.linalg.norm(ps_n) / chiN - 1))

            # === update B and D from C ===
            C_nxn = np.triu(C_nxn) + np.triu(C_nxn, k=1).T  # enforce symmetry
            w_n, B_nxn = np.linalg.eigh(C_nxn)
            D_nxn = np.diag(np.sqrt(w_n))
            B_nxn = B_nxn
            searchmodel.set_parameters((xmean_n, C_nxn))

        return trajectory


class ConditioningByAdaptiveSampling(ModelBasedOptimizationAlgorithm):
    def __init__(self, quantile: float = 0.9):
        self.s_threshold = -np.inf
        self.quantile = quantile

    def get_mle_weights(self, Xt_mxp: np.array, ot_m: np.array, ovart_m: np.array, init_searchmodel: SearchModel,
                        searchmodel: SearchModel):
        try:
            llt_m = searchmodel.loglikelihood(Xt_mxp)
            ll0_m = init_searchmodel.loglikelihood(Xt_mxp)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError
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
                        searchmodel: SearchModel):
        o_percentile = np.percentile(ot_m, self.quantile * 100)
        if o_percentile > self.s_threshold:
            self.s_threshold = o_percentile
        dbas_weights_n = sc.stats.norm.sf(self.s_threshold, loc=ot_m, scale=np.sqrt(ovart_m))
        return dbas_weights_n


class RewardWeightedRegression(ModelBasedOptimizationAlgorithm):
    def __init__(self, gamma: float = 0.01):
        self.gamma = gamma

    def get_mle_weights(self, Xt_mxp: np.array, ot_m: np.array, ovart_m: np.array, init_searchmodel: SearchModel,
                        searchmodel: SearchModel):
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
                        searchmodel: SearchModel):
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
                        searchmodel: SearchModel):
        probimprove_m = sc.stats.norm.sf(self.max_gt_train, loc=ot_m, scale=np.sqrt(ovart_m))
        threshold = np.percentile(probimprove_m, self.quantile * 100)
        cempi_weights_m = (probimprove_m >= threshold).astype(int)
        return cempi_weights_m


class RandomSearch(ModelBasedOptimizationAlgorithm):

    def __init__(self, max_gt_train: float):
        self.max_gt_train = max_gt_train
        print("Reminder: Random Search uses the MultivariateGaussian search model.")

    def get_mle_weights(self, Xt_nxp: np.array, ot_m: np.array, ovart_m: np.array, init_searchmodel: SearchModel,
                        searchmodel: SearchModel):
        probimprove_m = sc.stats.norm.sf(self.max_gt_train, loc=ot_m, scale=np.sqrt(ovart_m))
        rs_weights_m = np.zeros([ot_m.size])
        rs_weights_m[np.argmax(probimprove_m)] = 1.0
        return rs_weights_m

    def update_searchmodel(self, searchmodel: SearchModel, Xt_mxp: np.array, mboweights_m: np.array,
                           update_epochs: int = 0):
        if searchmodel.__class__ is not MultivariateGaussian:
            raise ValueError("Random search only uses a MultivariateGaussian search model.")
        bestsample_p = Xt_mxp[np.argmax(mboweights_m)]
        searchmodel.set_parameters((bestsample_p, np.eye(Xt_mxp.shape[1])))

