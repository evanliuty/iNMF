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
from sklearn.neighbors import NearestNeighbors


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
    def __init__(self,
                 data_dict,
                 k,
                 lam,
                 gam,
                 penalty=True,
                 metric="Frobenius",
                 cvg_k=5,
                 cvg_condition=1e-5):
        assert metric in ["Frobenius", "kld"]
        self.x1 = data_dict["data"][0] + 1
        self.x2 = data_dict["data"][1] + 1
        self.batches = data_dict["batches"]
        self.groups = data_dict["groups"] if "groups" in data_dict.keys() else None
        assert self.x1.shape[0] == self.x2.shape[0] if penalty is False else self.x1.shape == self.x2.shape
        self.k = k
        self.lam = lam
        self.gam = gam
        self.metric = metric
        self.penalty = penalty
        self.obj = []
        self.cvg = MovingAverage(cvg_k, cvg_condition)
        self.embedding = None
        self.original = None
        self.dataset_alignment_h = 0
        self.group_alignment_h = []
        self.dataset_alignment = 0
        self.group_alignment = []
        self._init_weights()

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

    def update_par(self):
        # Use latest pars for gradient
        grad_w = self._cal_grad_w()
        self.mat_w *= grad_w

        grad_v = self._cal_grad_v()
        self.mat_v1 *= grad_v[0]
        self.mat_v2 *= grad_v[1]

        grad_h = self._cal_grad_h()
        self.mat_h1 *= grad_h[0]
        self.mat_h2 *= grad_h[1]

    def current_par(self):
        return {"w": self.mat_w, "v1": self.mat_v1, "v2": self.mat_v2, "h1": self.mat_h1, "h2": self.mat_h2}

    def _init_weights(self, method="abs_normal"):
        if method == "abs_normal":
            self.mat_w = np.abs(np.random.randn(self.x1.shape[0], self.k)) + 0.1
            self.mat_v1 = np.abs(np.random.randn(self.x2.shape[0], self.k)) + 0.1
            self.mat_v2 = np.abs(np.random.randn(self.x2.shape[0], self.k)) + 0.1
            self.mat_h1 = np.abs(np.random.randn(self.k, self.x1.shape[1])) + 0.1
            self.mat_h2 = np.abs(np.random.randn(self.k, self.x2.shape[1])) + 0.1
        elif method == "uniform":
            self.mat_w = np.random.uniform(0.1, 1, (self.x1.shape[0], self.k))
            self.mat_v1 = np.random.uniform(0.1, 1, (self.x2.shape[0], self.k))
            self.mat_v2 = np.random.uniform(0.1, 1, (self.x2.shape[0], self.k))
            self.mat_h1 = np.random.uniform(0.1, 1, (self.k, self.x1.shape[1]))
            self.mat_h2 = np.random.uniform(0.1, 1, (self.k, self.x2.shape[1]))

    def _cal_grad_w(self):
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

    def _cal_grad_v(self):
        if self.metric == "Frobenius":
            grad_v1 = np.divide(self._cal_term("x1h1t"),
                                self._cal_term("wv1h1h1t") + self.lam * self._cal_term("v1h1h1t"))
            grad_v2 = np.divide(self._cal_term("x2h2t"),
                                self._cal_term("wv2h2h2t") + self.lam * self._cal_term("v2h2h2t"))
            return [grad_v1, grad_v2]
        elif self.metric == "kld":
            _numerator1 = np.dot(self._cal_term("x1dwv1h1", self._cal_term("wv1h1")), self.mat_h1.T)
            _numerator2 = np.dot(self._cal_term("x2dwv2h2", self._cal_term("wv2h2")), self.mat_h2.T)
            _denominator1 = np.array([np.sum(self.mat_h1, axis=1) for _ in range(self.mat_w.shape[0])]) \
                            + 2 * self.lam * self._cal_term("v1h1h1t")
            _denominator2 = np.array([np.sum(self.mat_h2, axis=1) for _ in range(self.mat_w.shape[0])]) \
                            + 2 * self.lam * self._cal_term("v2h2h2t")
            return [np.divide(_numerator1, _denominator1), np.divide(_numerator2, _denominator2)]

    def _cal_grad_h(self):
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

    def run_dr(self, dr_type, original=False):
        print("\n>>> Running " + dr_type.upper() + " Dimension Reduction.")
        data = np.concatenate((self.mat_h1.T, self.mat_h2.T), axis=0)
        if dr_type == "TSNE":
            self.embedding = TSNE(n_components=2).fit_transform(data)
            self.original = TSNE(n_components=2).fit_transform(
                np.concatenate((self.x1.T, self.x2.T))) if original else None
        elif dr_type == "PCA":
            self.embedding = PCA(n_components=2).fit_transform(data)
            self.original = PCA(n_components=2).fit_transform(
                np.concatenate((self.x1.T, self.x2.T))) if original else None

    def plot_embedding(self, dr_type):
        def _2d_scatter(embedding, label, dr, title):
            unique_label = np.unique(label)
            for item in unique_label:
                plt.scatter(embedding[label == item, 0], embedding[label == item, 1], s=1, label=item)
            plt.legend()
            plt.xlabel(dr.upper() + str(1))
            plt.ylabel(dr.upper() + str(2))
            plt.title(title)

        if self.groups is None and self.original is None:
            _2d_scatter(self.embedding, self.batches, dr_type, "Corrected Batches")
        elif self.groups is None and self.original is not None:
            plt.subplot(121)
            _2d_scatter(self.original, self.batches, dr_type, "Original Batches")
            plt.subplot(122)
            _2d_scatter(self.embedding, self.batches, dr_type, "Corrected Batches")
        elif self.original is None:
            plt.subplot(121)
            _2d_scatter(self.embedding, self.batches, dr_type, "Corrected Batches")
            plt.subplot(122)
            _2d_scatter(self.embedding, self.groups, dr_type, "Groups")
        else:
            plt.subplot(131)
            _2d_scatter(self.original, self.batches, dr_type, "Original Batches")
            plt.subplot(132)
            _2d_scatter(self.embedding, self.batches, dr_type, "Corrected Batches")
            plt.subplot(133)
            _2d_scatter(self.embedding, self.groups, dr_type, "Groups")
        plt.savefig("./iNMF.pdf", dpi=400)

    def plot_obj(self):
        plt.clf()
        plt.plot(np.arange(len(self.obj[2:])), self.obj[2:])
        plt.title("Objective Function Value")
        plt.ylabel("Objective")
        plt.xlabel("Iteration")
        plt.savefig("./obj.pdf", dpi=400)

    def cal_alignment(self, space="hd", k_percent=0.01):
        # Calculate alignment score from Butler et. al. nbt, 2018.
        def _align(embedding, k_percent=0.01):
            group_alignment = []

            def _cal_ali(nbrs, embedding, label):
                _, index = nbrs.kneighbors(embedding)
                alignment = 0
                for i, cell in enumerate(index):
                    knn_label = label[cell]
                    cell_label = label[i]
                    alignment += 1 - (np.sum(knn_label == cell_label) - k / 2) / (k - k / 2)
                return alignment / embedding.shape[0]

            k = int(self.embedding.shape[0] * k_percent)

            # Dataset alignment
            nbrs = NearestNeighbors(n_neighbors=k).fit(embedding)
            dataset_alignment = _cal_ali(nbrs, embedding, self.batches)

            # Cell type specific alignment
            if self.groups is not None:
                unique_group = np.unique(self.groups)
                for item in unique_group:
                    group_emb = embedding[self.groups == item]
                    group_batch = self.batches[self.groups == item]
                    nbrs = NearestNeighbors(n_neighbors=k).fit(group_emb)
                    group_ali = _cal_ali(nbrs, group_emb, group_batch)
                    group_alignment.append([item, group_ali])
            return dataset_alignment, group_alignment

        if space == "h":
            self.dataset_alignment_h, self.group_alignment_h = _align(
                np.concatenate((self.mat_h1.T, self.mat_h2.T), axis=0), k_percent=k_percent)
        elif space == "hd":
            self.dataset_alignment, self.group_alignment = _align(self.embedding, k_percent=k_percent)
            self.dataset_alignment_h, self.group_alignment_h = _align(
                np.concatenate((self.mat_h1.T, self.mat_h2.T), axis=0), k_percent=k_percent)
        elif space == "d":
            self.dataset_alignment, self.group_alignment = _align(self.embedding, k_percent=k_percent)

        print("\nStatistics:")
        if self.dataset_alignment_h is not None:
            print("\tAlignment score in H space: {}".format(self.dataset_alignment_h))
        if self.dataset_alignment is not None:
            print("\tAlignment score: {}".format(self.dataset_alignment))
        if self.group_alignment_h is not []:
            print("\tGroup alignment score in H space:")
            for item in self.group_alignment_h:
                print("\t\t{}: {}".format(item[0], item[1]))
        if self.group_alignment_h is not []:
            print("Group alignment score:")
            for item in self.group_alignment:
                print("\t\t{}: {}".format(item[0], item[1]))

    @staticmethod
    def frobenius_norm(x):
        return np.linalg.norm(x)

    @staticmethod
    def kl_divergence(x1, x2):
        return np.sum(np.multiply(x1, np.log(np.divide(x1, x2))) - x1 + x2)

    def __str__(self):
        print("\niNMF Model Summary")
        print("\tData Shape:\t\t{}, {}, {}".format(self.x1.shape, self.x2.shape, "# Genes x # Cells"))
        print("\tGroup Avail:\t\t{}".format(str(self.groups is not None)))
        print("\tk:\t\t\t{}".format(self.k))
        print("\tMetric:\t\t\t{}".format(self.metric))
        print("\tLambda:\t\t\t{}".format(self.lam))
        print("\tPenalty:\t\t{}".format(self.penalty))
        print("\tGamma:\t\t\t{}".format(self.gam))
        return ""

