#   -*- coding: utf-8 -*-
#
#   factorization.py
#   
#   Developed by Tianyi Liu on 2020-11-24 as tianyi
#   Copyright (c) 2020. All Rights Reserved.

"""

"""


import numpy as np


class iNMF():
    def __init__(self, x1, x2, k, lam, gam, penalty=True, metric="Frobenius"):
        self.x1 = x1
        self.x2 = x2
        self.k = k
        self.lam = lam
        self.gam = gam
        self.metric = metric
        self.penalty = penalty

        assert x1.shape[0] == x2.shape[0]

        self.mat_w = np.abs(np.random.randn(x1.shape[0], k)) + 0.1
        self.mat_v1 = np.abs(np.random.randn(x1.shape[0], k)) + 0.1
        self.mat_v2 = np.abs(np.random.randn(x2.shape[0], k)) + 0.1
        self.mat_h1 = np.abs(np.random.randn(k, x1.shape[1])) + 0.1
        self.mat_h2 = np.abs(np.random.randn(k, x2.shape[1])) + 0.1

    @classmethod
    def frobenius_norm(cls, x):
        return np.linalg.norm(x)

    @classmethod
    def kl_divergence(cls, x1, x2):
        return np.sum(np.multiply(x1, np.log(np.divide(x1, x2))) - x1 + x2)

    def cal_objective(self):
        if self.metric == "Frobenius":
            objective = iNMF.frobenius_norm(self.x1 - np.dot(self.mat_w + self.mat_v1, self.mat_h1)) \
                        + iNMF.frobenius_norm(self.x2 - np.dot(self.mat_w + self.mat_v2, self.mat_h2)) \
                        + self.lam * iNMF.frobenius_norm(np.dot(self.mat_v1, self.mat_h1)) \
                        + self.lam * iNMF.frobenius_norm(np.dot(self.mat_v2, self.mat_h2))
            if self.penalty:
                return objective + self.gam * iNMF.frobenius_norm(self.mat_h1 - self.mat_h2)
            else:
                return objective

        elif self.metric == "kld":
            objective = iNMF.kl_divergence(self.x1, np.dot(self.mat_w + self.mat_v1, self.mat_h1)) \
                        + iNMF.kl_divergence(self.x2, np.dot(self.mat_w + self.mat_v2, self.mat_h1)) \
                        + self.lam * iNMF.frobenius_norm(np.dot(self.mat_v1, self.mat_h1)) \
                        + self.lam * iNMF.frobenius_norm(np.dot(self.mat_v2, self.mat_h2))
            if self.penalty:
                return objective + self.gam * iNMF.kl_divergence(self.mat_h1, self.mat_h2)
            else:
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
        _x1h1t = np.dot(self.x1, self.mat_h1.T)
        _x2h2t = np.dot(self.x2, self.mat_h2.T)
        _wv1h1h1t = np.linalg.multi_dot([self.mat_w + self.mat_v1, self.mat_h1, self.mat_h1.T])
        _wv2h2h2t = np.linalg.multi_dot([self.mat_w + self.mat_v2, self.mat_h2, self.mat_h2.T])
        grad_w = np.divide(_x1h1t + _x2h2t, _wv1h1h1t + _wv2h2h2t)
        return grad_w

    def cal_grad_v(self):
        _x1h1t = np.dot(self.x1, self.mat_h1.T)
        _x2h2t = np.dot(self.x2, self.mat_h2.T)
        _wv1h1h1t = np.linalg.multi_dot([self.mat_w + self.mat_v1, self.mat_h1, self.mat_h1.T])
        _wv2h2h2t = np.linalg.multi_dot([self.mat_w + self.mat_v2, self.mat_h2, self.mat_h2.T])
        _v1h1h1t = np.linalg.multi_dot([self.mat_v1, self.mat_h1, self.mat_h1.T])
        _v2h2h2t = np.linalg.multi_dot([self.mat_v2, self.mat_h2, self.mat_h2.T])

        grad_v1 = np.divide(2 * _x1h1t, 2 * _wv1h1h1t + self.lam * _v1h1h1t)
        grad_v2 = np.divide(2 * _x2h2t, 2 * _wv2h2h2t + self.lam * _v2h2h2t)
        return [grad_v1, grad_v2]

    def cal_grad_h(self):
        _wv1tx1 = np.dot(self.mat_w.T + self.mat_v1.T, self.x1)
        _wv2tx2 = np.dot(self.mat_w.T + self.mat_v2.T, self.x2)
        _wv1twv1h1 = np.linalg.multi_dot([self.mat_w.T + self.mat_v1.T, self.mat_w + self.mat_v1, self.mat_h1])
        _wv2twv2h2 = np.linalg.multi_dot([self.mat_w.T + self.mat_v2.T, self.mat_w + self.mat_v2, self.mat_h2])
        _v1tv1h1 = np.linalg.multi_dot([self.mat_v1.T, self.mat_v1, self.mat_h1])
        _v2tv2h2 = np.linalg.multi_dot([self.mat_v2.T, self.mat_v2, self.mat_h2])
        if self.penalty is False:
            grad_h1 = np.divide(_wv1tx1, _wv1twv1h1 + self.lam * _v1tv1h1)
            grad_h2 = np.divide(_wv2tx2, _wv2twv2h2 + self.lam * _v2tv2h2)
        else:
            grad_h1 = 2 * np.divide(_wv1tx1, 2 * _wv1twv1h1 + 2 * self.lam * _v1tv1h1 + 2 * self.gam * (
                        self.mat_h1 - self.mat_h2))
            grad_h2 = 2 * np.divide(_wv2tx2, 2 * _wv2twv2h2 + 2 * self.lam * _v2tv2h2 + 2 * self.gam * (
                        self.mat_h2 - self.mat_h1))
        return [grad_h1, grad_h2]

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
        return [self.mat_w, self.mat_v1, self.mat_v2, self.mat_h1, self.mat_h2]


