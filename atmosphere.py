from math import *
import numpy as np
import scipy.interpolate as interp


class Atmosphere_GOST_4401_81:
    def __init__(self):
        self.g = 9.80665
        self.r = 6356767
        self.R = 8314.32 / 28.96442
        self.beta_S = 1.458e-6
        self.S = 110.4
        self.k = 1.4
        self.c_p = self.R / (1 - 1 / self.k)
        self.data_H = np.array([0, 11, 20, 32, 47, 51, 71, 85]) * 1e3
        self.data_T = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.65])
        self.data_B = np.array([-6.5, 0, 1, 2.8, 0, -2.8, -2]) * 1e-3
        self.T_H = interp.interp1d(self.data_H, self.data_T, kind='linear')
        self.data_p = np.ones(len(self.data_T)) * 101325
        for i in range(1, len(self.data_T)):
            if self.data_B[i - 1] != 0:
                self.data_p[i] = self.data_p[i - 1] * \
                                 (self.data_T[i] / self.data_T[i - 1]) ** (-self.g / (self.data_B[i - 1] * self.R))
            else:
                self.data_p[i] = self.data_p[i - 1] * \
                                 exp(-self.g * (self.data_H[i] - self.data_H[i - 1]) / (self.R * self.data_T[i - 1]))

    def T(self, h):
        H = self.r * h / (self.r + h)
        if H < min(self.data_H):
            return self.data_T[0]
        elif H >= max(self.data_H):
            return self.data_T[-1]
        else:
            return self.T_H(H)

    def p(self, h):
        H = self.r * h / (self.r + h)
        if H < min(self.data_H):
            return self.data_p[0]
        elif H >= max(self.data_H):
            return self.data_p[-1]
        else:
            i = 0
            for j in range(len(self.data_B)):
                if H >= self.data_H[j]:
                    i = j
            if self.data_B[i] != 0:
                return self.data_p[i] * (self.T_H(H) / self.data_T[i]) ** (-self.g / (self.data_B[i] * self.R))
            else:
                return self.data_p[i] * exp(-self.g * (H - self.data_H[i]) / (self.R * self.data_T[i]))

    def rho(self, h):
        return self.p(h) / (self.R * self.T(h))

    def a(self, h):
        return (self.k * self.R * self.T(h)) ** 0.5

    def mu(self, T):
        return self.beta_S * T ** 1.5 / (T + self.S)

    def lam(self, T):
        return 2.648151e-3 * T ** 1.5 / (T + 245.4 * 10 ** (-12 / T))