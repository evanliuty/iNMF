#   -*- coding: utf-8 -*-
#
#   factorization.py
#   
#   Developed by Tianyi Liu on 2020-11-24 as tianyi
#   Copyright (c) 2020. All Rights Reserved.

"""

"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class MovingAverage:
    def __init__(self, k, condition=1e-5):
        self.k = k
        self.condition = condition
        self.container = [np.inf * k]
        self.ma = np.inf
        self.ma_prev = np.inf

    def update_ma(self, obj_val):
        self.container.insert(0, obj_val)
        self.container.pop()
        self._cal_ma()

    def _cal_ma(self):
        if np.inf not in self.container:
            self.ma_prev = self.ma
            self.ma = sum(self.container) / len(self.container)

    def is_converge(self):
        if self.ma > self.ma_prev or (np.inf in self.container) or self.ma_prev == np.inf:
            return False
        else:
            return ((self.ma_prev - self.ma) / self.ma_prev) < self.condition


class iNMF:
    def __init__(self, x1, x2, k, lam, gam, penalty=True, metric="Frobenius", cvg_k=5, cvg_condition=1e-5):
        assert x1.shape[0] == x2.shape[0] if penalty is False else x1.shape == x2.shape
        assert metric in ["Frobenius", "kld"]
        self.x1 = x1
        self.x2 = x2
        self.k = k
        self.lam = lam
        self.gam = gam
        self.metric = metric
        self.penalty = penalty
        self.mat_w = np.abs(np.random.randn(x1.shape[0], k)) + 0.1
        self.mat_v1 = np.abs(np.random.randn(x1.shape[0], k)) + 0.1
        self.mat_v2 = np.abs(np.random.randn(x2.shape[0], k)) + 0.1
        self.mat_h1 = np.abs(np.random.randn(k, x1.shape[1])) + 0.1
        self.mat_h2 = np.abs(np.random.randn(k, x2.shape[1])) + 0.1
        self.obj = []
        self.cvg = MovingAverage(cvg_k, cvg_condition)
        self.embedding = None
        self.original = None

    def cal_objective(self):
        if self.metric == "Frobenius":
            objective = iNMF.frobenius_norm(self.x1 - np.dot(self.mat_w + self.mat_v1, self.mat_h1)) \
                        + iNMF.frobenius_norm(self.x2 - np.dot(self.mat_w + self.mat_v2, self.mat_h2)) \
                        + self.lam * iNMF.frobenius_norm(np.dot(self.mat_v1, self.mat_h1)) \
                        + self.lam * iNMF.frobenius_norm(np.dot(self.mat_v2, self.mat_h2))
            if self.penalty:
                objective += objective + self.gam * iNMF.frobenius_norm(self.mat_h1 - self.mat_h2)

        elif self.metric == "kld":
            objective = iNMF.kl_divergence(self.x1, np.dot(self.mat_w + self.mat_v1, self.mat_h1)) \
                        + iNMF.kl_divergence(self.x2, np.dot(self.mat_w + self.mat_v2, self.mat_h1)) \
                        + self.lam * iNMF.frobenius_norm(np.dot(self.mat_v1, self.mat_h1)) \
                        + self.lam * iNMF.frobenius_norm(np.dot(self.mat_v2, self.mat_h2))
            if self.penalty:
                objective += self.gam * iNMF.kl_divergence(self.mat_h1, self.mat_h2)
        else:
            raise ValueError("Invalid Metric {} Specified.".format(self.metric))

        self.obj.append(objective)
        return objective

    def cal_gradient(self):
        if self.metric == "Frobenius":

            _x1h1t = np.dot(self.x1, self.mat_h1.T)
            _x2h2t = np.dot(self.x2, self.mat_h2.T)
            _wv1h1h1t = np.linalg.multi_dot([self.mat_w + self.mat_v1, self.mat_h1, self.mat_h1.T])
            _wv2h2h2t = np.linalg.multi_dot([self.mat_w + self.mat_v2, self.mat_h2, self.mat_h2.T])
            _v1h1h1t = np.linalg.multi_dot([self.mat_v1, self.mat_h1, self.mat_h1.T])
            _v2h2h2t = np.linalg.multi_dot([self.mat_v2, self.mat_h2, self.mat_h2.T])
            _wv1tx1 = np.dot(self.mat_w.T + self.mat_v1.T, self.x1)
            _wv2tx2 = np.dot(self.mat_w.T + self.mat_v2.T, self.x2)
            _wv1twv1h1 = np.linalg.multi_dot([self.mat_w.T + self.mat_v1.T, self.mat_w + self.mat_v1, self.mat_h1])
            _wv2twv2h2 = np.linalg.multi_dot([self.mat_w.T + self.mat_v2.T, self.mat_w + self.mat_v2, self.mat_h2])
            _v1tv1h1 = np.linalg.multi_dot([self.mat_v1.T, self.mat_v1, self.mat_h1])
            _v2tv2h2 = np.linalg.multi_dot([self.mat_v2.T, self.mat_v2, self.mat_h2])

            grad_w = np.divide(_x1h1t + _x2h2t, _wv1h1h1t + _wv2h2h2t)
            grad_v1 = np.divide(2 * _x1h1t, 2 * _wv1h1h1t + self.lam * _v1h1h1t)
            grad_v2 = np.divide(2 * _x2h2t, 2 * _wv2h2h2t + self.lam * _v2h2h2t)
            grad_h1 = np.divide(_wv1tx1, _wv1twv1h1 + self.lam * _v1tv1h1)
            grad_h2 = np.divide(_wv2tx2, _wv2twv2h2 + self.lam * _v2tv2h2)
            grad_h1p = np.divide(_wv1tx1,
                                 2 * _wv1twv1h1 + 2 * self.lam * _v1tv1h1 + 2 * self.gam * (self.mat_h1 - self.mat_h2))
            grad_h2p = np.divide(_wv2tx2,
                                 2 * _wv2twv2h2 + 2 * self.lam * _v2tv2h2 + 2 * self.gam * (self.mat_h2 - self.mat_h1))

            if self.penalty:
                return [grad_w, grad_v1, grad_v2, grad_h1p, grad_h2p]
            else:
                return [grad_w, grad_v1, grad_v2, grad_h1, grad_h2]

        if self.metric == "kld":
            raise NotImplementedError

    def cal_grad_w(self):
        if self.metric == "Frobenius":
            grad_w = np.divide(self._cal_term("x1h1t") + self._cal_term("x2h2t"),
                               self._cal_term("wv1h1h1t") + self._cal_term("wv2h2h2t"))
            return grad_w
        elif self.metric == "kld":
            _numerator = np.dot(self._cal_term("x1dwv1h1", self._cal_term("wv1h1")), self.mat_h1.T) \
                         + np.dot(self._cal_term("x2dwv2h2", self._cal_term("wv2h2")), self.mat_h2.T)
            _denominator = np.array([np.sum(self.mat_h1, axis=1) for _ in range(self.mat_w.shape[0])]) \
                           + np.array([np.sum(self.mat_h2, axis=1) for _ in range(self.mat_w.shape[0])])
            return np.divide(_numerator, _denominator)

    def cal_grad_v(self):
        if self.metric == "Frobenius":
            grad_v1 = np.divide(self._cal_term("x1h1t"),
                                self._cal_term("wv1h1h1t") + self.lam * self._cal_term("v1h1h1t"))
            grad_v2 = np.divide(self._cal_term("x2h2t"),
                                self._cal_term("wv2h2h2t") + self.lam * self._cal_term("v2h2h2t"))
            return [grad_v1, grad_v2]
        elif self.metric == "kld":
            _numerator1 = np.dot(self._cal_term("x1dwv1h1", self._cal_term("wv1h1")), self.mat_h1.T)
            _numerator2 = np.dot(self._cal_term("x2dwv2h2", self._cal_term("wv2h2")), self.mat_h1.T)
            _denominator1 = np.array([np.sum(self.mat_h1, axis=1) for _ in range(self.mat_w.shape[0])]) \
                            + 2 * self.lam * self._cal_term("v1h1h1t")
            _denominator2 = np.array([np.sum(self.mat_h2, axis=1) for _ in range(self.mat_w.shape[0])]) \
                            + 2 * self.lam * self._cal_term("v2h2h2t")
            return [np.divide(_numerator1, _denominator1), np.divide(_numerator2, _denominator2)]

    def cal_grad_h(self):
        if self.metric == "Frobenius":
            _numerator1 = self._cal_term("wv1tx1")
            _numerator2 = self._cal_term("wv2tx2")
            _denominator1 = self._cal_term("wv1twv1h1") + self.lam * self._cal_term("v1tv1h1")
            _denominator2 = self._cal_term("wv2twv2h2") + self.lam * self._cal_term("v2tv2h2")
            return [np.divide(_numerator1, _denominator1), np.divide(_numerator2, _denominator2)] \
                if not self.penalty else \
                [np.divide(_numerator1, _denominator1 + self.gam * (self.mat_h1 - self.mat_h2)),
                 np.divide(_numerator2, _denominator2 + self.gam * (self.mat_h1 - self.mat_h2))]
        elif self.metric == "kld":
            _numerator1 = np.dot((self.mat_w + self.mat_v1).T, self._cal_term("x1dwv1h1", self._cal_term("wv1h1")))
            _numerator2 = np.dot((self.mat_w + self.mat_v2).T, self._cal_term("x2dwv2h2", self._cal_term("wv2h2")))
            _denominator1 = np.array([np.sum(self.mat_w + self.mat_v1, axis=0) for _ in
                                      range(self.mat_h1.shape[1])]).T + 2 * self.lam * self._cal_term("v1tv1h1")
            _denominator2 = np.array([np.sum(self.mat_w + self.mat_v2, axis=0) for _ in
                                      range(self.mat_h2.shape[1])]).T + 2 * self.lam * self._cal_term("v2tv2h2")
            return [np.divide(_numerator1, _denominator1), np.divide(_numerator2, _denominator2)] \
                if not self.penalty else \
                [np.divide(_numerator1, _denominator1 + self.gam * np.log(np.divide(self.mat_h1, self.mat_h2))),
                 np.divide(_numerator2, _denominator2 + self.gam * (1 - np.divide(self.mat_h1, self.mat_h2)))]

    def _cal_term(self, term, *argv):
        if term == "x1h1t":
            return np.dot(self.x1, self.mat_h1.T)
        elif term == "x2h2t":
            return np.dot(self.x2, self.mat_h2.T)
        elif term == "wv1h1h1t":
            return np.linalg.multi_dot([self.mat_w + self.mat_v1, self.mat_h1, self.mat_h1.T])
        elif term == "wv2h2h2t":
            return np.linalg.multi_dot([self.mat_w + self.mat_v2, self.mat_h2, self.mat_h2.T])
        elif term == "wv1h1":
            return np.dot(self.mat_w + self.mat_v1, self.mat_h1)
        elif term == "wv2h2":
            return np.dot(self.mat_w + self.mat_v2, self.mat_h2)
        elif term == "x1dwv1h1":
            return np.divide(self.x1, argv[0])
        elif term == "x2dwv2h2":
            return np.divide(self.x2, argv[0])
        elif term == "v1h1h1t":
            return np.linalg.multi_dot([self.mat_v1, self.mat_h1, self.mat_h1.T])
        elif term == "v2h2h2t":
            return np.linalg.multi_dot([self.mat_v2, self.mat_h2, self.mat_h2.T])
        elif term == "wv1":
            return
        elif term == "v1tv1h1":
            return np.linalg.multi_dot([self.mat_v1.T, self.mat_v1, self.mat_h1])
        elif term == "v2tv2h2":
            return np.linalg.multi_dot([self.mat_v2.T, self.mat_v2, self.mat_h2])
        elif term == "wv1tx1":
            return np.dot(self.mat_w.T + self.mat_v1.T, self.x1)
        elif term == "wv2tx2":
            return np.dot(self.mat_w.T + self.mat_v2.T, self.x2)
        elif term == "wv1twv1h1":
            return np.linalg.multi_dot([self.mat_w.T + self.mat_v1.T, self.mat_w + self.mat_v1, self.mat_h1])
        elif term == "wv2twv2h2":
            return np.linalg.multi_dot([self.mat_w.T + self.mat_v2.T, self.mat_w + self.mat_v2, self.mat_h2])
        elif term == "v1tv1h1":
            return np.linalg.multi_dot([self.mat_v1.T, self.mat_v1, self.mat_h1])
        elif term == "v2tv2h2":
            return np.linalg.multi_dot([self.mat_v2.T, self.mat_v2, self.mat_h2])
        else:
            raise NameError("Invalid Term: {}".format(term))

    def update_par(self):
        # Use latest pars for gradient
        grad_w = self.cal_grad_w()
        self.mat_w *= grad_w

        grad_v = self.cal_grad_v()
        self.mat_v1 *= grad_v[0]
        self.mat_v2 *= grad_v[1]

        grad_h = self.cal_grad_h()
        self.mat_h1 *= grad_h[0]
        self.mat_h2 *= grad_h[1]

    def current_par(self):
        return {"w": self.mat_w, "v1": self.mat_v1, "v2": self.mat_v2, "h1": self.mat_h1, "h2": self.mat_h2}

    def run_dr(self, dr_type, original=False):
        print(">>> Running " + dr_type.upper() + " Dimension Reduction.")
        data = np.concatenate((self.mat_h1.T, self.mat_h2.T), axis=0)
        if dr_type == "TSNE":
            self.embedding = TSNE(n_components=2).fit_transform(data)
            self.original = TSNE(n_components=2).fit_transform(
                np.concatenate((self.x1.T, self.x2.T))) if original else None
        elif dr_type == "PCA":
            self.embedding = PCA(n_components=2).fit_transform(data)
            self.original = PCA(n_components=2).fit_transform(
                np.concatenate((self.x1.T, self.x2.T))) if original else None

    def plot_embedding(self, batch_label, group_label, dr_type):
        def _2d_scatter(embedding, label, dr, title):
            unique_label = np.unique(label)
            for item in unique_label:
                plt.scatter(embedding[label == item, 0], embedding[label == item, 1], s=1, label=item)
            plt.xlabel(dr.upper() + str(1))
            plt.ylabel(dr.upper() + str(2))
            plt.title(title)

        if group_label is None and self.original is None:
            _2d_scatter(self.embedding, batch_label, dr_type, "Corrected Batches")
        elif group_label is None and self.original is not None:
            plt.subplot(121)
            _2d_scatter(self.original, batch_label, dr_type, "Original Batches")
            plt.subplot(122)
            _2d_scatter(self.embedding, batch_label, dr_type, "Corrected Batches")
        elif self.original is None:
            plt.subplot(121)
            _2d_scatter(self.embedding, batch_label, dr_type, "Corrected Batches")
            plt.subplot(122)
            _2d_scatter(self.embedding, group_label, dr_type, "Groups")
        else:
            plt.subplot(131)
            _2d_scatter(self.original, batch_label, dr_type, "Original Batches")
            plt.subplot(132)
            _2d_scatter(self.embedding, batch_label, dr_type, "Corrected Batches")
            plt.subplot(133)
            _2d_scatter(self.embedding, group_label, dr_type, "Groups")
        plt.savefig("./iNMF.pdf", dpi=400)

    def plot_obj(self):
        plt.clf()
        plt.plot(np.arange(len(self.obj[2:])), self.obj[2:])
        plt.title("Objective Function Value")
        plt.ylabel("Objective")
        plt.xlabel("Iteration")
        plt.savefig("./obj.pdf", dpi=400)

    @staticmethod
    def frobenius_norm(x):
        return np.linalg.norm(x)

    @staticmethod
    def kl_divergence(x1, x2):
        return np.sum(np.multiply(x1, np.log(np.divide(x1, x2))) - x1 + x2)


