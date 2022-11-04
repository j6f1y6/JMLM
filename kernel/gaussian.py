import numpy as np
from math import exp
from .base import Kernel


class Gaussian(Kernel):

    def compute(self, X, center, sigma):
        if sigma == 0:
            if np.linalg.norm(X - center) == 0: return 1
            return 0
        return exp(-(np.linalg.norm(X - center) ** 2 / (2 * sigma ** 2)))
