import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod
from math import sqrt
from copy import deepcopy

class GroupwiseMatrix:
    def __init__(self,
                 G: np.ndarray,
                 x: np.ndarray):
        assert x.shape[1] == 1, "x must be an n x 1 vector"
        n = x.shape[0]

        Y = G * (x @ x.T) + np.random.normal(loc = 0.0, scale = 1.0, size = (n, n))
        Ys = G * Y

        self._x = x
        self._Lgd = G**2 + (G.T)**2
        self._Ygd = Ys + Ys.T
        self._Ysym = (1.0 / (np.sqrt(self._Lgd) + 1e-12)) * self._Ygd

        self._Yamp = deepcopy(self._Ygd)
        np.fill_diagonal(self._Yamp, np.zeros((n,)))
        self._Lamp = deepcopy(self._Lgd)
        np.fill_diagonal(self._Lamp, np.zeros((n,)))

    def x(self) -> np.ndarray:
        return self._x

    def Yamp(self) -> np.ndarray:
        return self._Yamp

    def Ygd(self) -> np.ndarray:
        return self._Ygd

    def Ysym(self) -> np.ndarray:
        return self._Ysym

    def Lamp(self) -> np.ndarray:
        return self._Lamp

    def Lgd(self) -> np.ndarray:
        return self._Lgd

class Prior(ABC):
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def sample(self) -> np.ndarray:
        raise NotImplementedError

class GaussianPrior(Prior):
    def __init__(self,
                 nDims: int):
        self._nDims = nDims

    def dimension(self) -> int:
        return self._nDims

    def sample(self) -> np.ndarray:
        return np.random.normal(loc = 0.0, scale = 1.0, size = (self._nDims, 1))

class DiscretePrior(Prior):
    def __init__(self,
                 nDims: int,
                 probs: np.ndarray,
                 vals: np.ndarray):
        assert probs.shape[0] == vals.shape[0], "Probs and vals must have same length"
        assert len(probs.shape) == 1 and len(vals.shape) == 1, "Probs and vals must be one dimensional"

        self._nDims = nDims
        self._p = probs
        self._x = vals
        self._secondMoment = (probs * vals**2).sum()

    def dimension(self) -> int:
        return self._nDims

    def probs(self) -> np.ndarray:
        return self._p

    def vals(self) -> np.ndarray:
        return self._x

    def secondMoment(self) -> float:
        return self._secondMoment

    def sample(self) -> np.ndarray:
        return np.random.choice(self._x, size = (self._nDims, 1), p = self._p)

def generateTwoGroupMatrix(alpha: float,
                           lmbda: float,
                           uPrior: Prior,
                           vPrior: Prior) -> GroupwiseMatrix:
    nu = uPrior.dimension()
    nv = vPrior.dimension()

    nDims = nu + nv
    scale = 1.0 / sqrt(float(nDims))

    lmbda11 = (1.0 - alpha) * lmbda
    lmbda12 = alpha * lmbda
    G = np.ones((nDims, nDims))
    G[:nu, :nu], G[nu:, nu:] = sqrt(lmbda11) * G[:nu, :nu], sqrt(lmbda11) * G[nu:, nu:]
    G[:nu, nu:], G[nu:, :nu] = sqrt(lmbda12) * G[:nu, nu:], sqrt(lmbda12) * G[nu:, :nu]
    G = scale * G

    u = uPrior.sample()
    v = vPrior.sample()
    x = np.empty((nDims, 1))
    x[:nu] = u
    x[nu:] = v
    return GroupwiseMatrix(G, x)
