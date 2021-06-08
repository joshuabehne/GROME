import numpy as np
from matplotlib import pyplot as plt
from math import sqrt, log
from typing import Tuple, Callable, Optional, Any, Dict
from abc import ABC, abstractmethod
from functools import partial
from copy import deepcopy

from scipy.optimize import minimize, Bounds, shgo, root, minimize_scalar
from scipy.integrate import quad
from scipy.linalg import eigh

from numba import float64, jit

from sklearn.utils.extmath import randomized_svd

def symmetrize(X: np.ndarray) -> np.ndarray:
    return (1.0 / sqrt(2.0)) * (X.T + X)

class MatrixPrior(ABC):
    @abstractmethod
    def dimension(self) -> Tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def sample(self) -> np.ndarray:
        raise NotImplementedError

class GaussianMatrixPrior(MatrixPrior):
    def __init__(self,
                 nDims: int,
                 dDims: int,
                 sigma: float):
        self._nDims = nDims
        self._dDims = dDims
        self._sigma = sigma

    def dimension(self) -> Tuple[int, int]:
        return self._nDims, self._dDims

    def sample(self) -> np.ndarray:
        return np.random.normal(loc = 0.0, scale = self._sigma, size = (self._nDims, self._dDims))

class DiscreteMatrixPrior(MatrixPrior):
    def __init__(self,
                 nDims: int,
                 dDims: int,
                 probs: np.ndarray,
                 vals: np.ndarray):
        assert probs.shape[0] == vals.shape[0], "Probs and vals must have same length"
        assert len(probs.shape) == 1 and len(vals.shape) == 1, "Probs and vals must be one dimensional"

        self._nDims = nDims
        self._dDims = dDims
        self._p = probs
        self._x = vals
        self._secondMoment = (probs * vals**2).sum()

    def dimension(self) -> Tuple[int, int]:
        return self._nDims, self._dDims

    def probs(self) -> np.ndarray:
        return self._p

    def vals(self) -> np.ndarray:
        return self._x

    def secondMoment(self) -> float:
        return self._secondMoment

    def sample(self):
        return np.random.choice(self._x, size = (self._nDims, self._dDims), p = self._p)

@jit(nopython = True, cache = True)
def partialLogIntegrand(x: float,
                        sgamma: float,
                        vals: np.ndarray,
                        probs: np.ndarray) -> float:
    gscale = 1.0 / sqrt(2.0 * np.pi)
    mix = probs * np.exp(-0.5 * ((x - (sgamma * vals))**2))
    mixDiff = mix * vals * (x - (sgamma * vals))
    mixSum = mix.sum()
    if mixSum <  1e-9:
        return 0.0

    return gscale * mixDiff.sum() * log(gscale * mixSum)

def computePartialLog(gamma: float,
                      partialRootGamma: float,
                      vals: np.ndarray,
                      probs: np.ndarray) -> float:
    return partialRootGamma * (quad(partialLogIntegrand, -np.inf, np.inf, args = (sqrt(gamma), vals, probs))[0])

def computeDiscreteKLDiv(gamma: float,
                         partialRootGamma: float,
                         partialGamma: float,
                         secondMoment: float,
                         vals: np.ndarray,
                         probs: np.ndarray) -> float:
    return 0.5 * partialGamma * secondMoment + computePartialLog(gamma, partialRootGamma, vals, probs)

def discreteLimitsGrad(q: np.ndarray,
                       uProbs: np.ndarray,
                       uVals: np.ndarray,
                       vProbs: np.ndarray,
                       vVals: np.ndarray,
                       uMom2: float,
                       vMom2: float,
                       l11: float,
                       l12: float,
                       l22: float) -> np.ndarray:
    assert q.shape[0] == 4, "Q should only have 4 entries"

    q1t, q2t, q1, q2 = q[0], q[1], q[2], q[3]
    if (q1t < 0.0 or q2t < 0.0):
        return np.ones((4,))

    if (q1 < 0.0 or q2 < 0.0):
        return np.ones((4,))

    if (q1 > 0.5 or q2 > 0.5):
        return np.ones((4,))

    gammau = l11 * (q1 + q1t) + l12 * (q2 + q2t)
    gammav = l22 * (q2 + q2t) + l12 * (q1 + q1t)
    sgammau = sqrt(gammau)
    sgammav = sqrt(gammav)

    grad = np.empty((4,))

    klDivUq1t = computeDiscreteKLDiv(gammau, (0.5 * l11) / sgammau, l11, uMom2, uVals, uProbs)
    klDivVq1t = computeDiscreteKLDiv(gammav, (0.5 * l12) / sgammav, l12, vMom2, vVals, vProbs)
    grad[0] = 0.5 * (klDivUq1t + klDivVq1t) - 0.5 * (l11 * q1 + l12 * q2)

    klDivUq2t = computeDiscreteKLDiv(gammau, (0.5 * l12) / sgammau, l12, uMom2, uVals, uProbs)
    klDivVq2t = computeDiscreteKLDiv(gammav, (0.5 * l22) / sgammav, l22, vMom2, vVals, vProbs)
    grad[1] = 0.5 * (klDivUq2t + klDivVq2t) - 0.5 * (l12 * q1 + l22 * q2)

    klDivUq1 = computeDiscreteKLDiv(gammau, (0.5 * l11) / sgammau, l11, uMom2, uVals, uProbs)
    klDivVq1 = computeDiscreteKLDiv(gammav, (0.5 * l12) / sgammav, l12, vMom2, vVals, vProbs)
    grad[2] = 0.5 * (klDivUq1 + klDivVq1) - 0.5 * (l11 * q1t + l12 * q2t)

    klDivUq2 = computeDiscreteKLDiv(gammau, (0.5 * l12) / sgammau, l12, uMom2, uVals, uProbs)
    klDivVq2 = computeDiscreteKLDiv(gammav, (0.5 * l22) / sgammav, l22, vMom2, vVals, vProbs)
    grad[3] = 0.5 * (klDivUq2 + klDivVq2) - 0.5 * (l12 * q1t + l22 * q2t)
    return grad

def getDiscreteFundLimits(alpha: float,
                          lmbda: float,
                          uPrior: DiscreteMatrixPrior,
                          vPrior: DiscreteMatrixPrior) -> Tuple[float]:
    if (alpha == 0.5):
        l11 = (1.0 - alpha - 1e-5) * lmbda
        l12 = (alpha + 1e-5) * lmbda
        l22 = (1.0 - alpha - 1e-5) * lmbda
    else:
        l11 = (1.0 - alpha) * lmbda
        l12 = alpha * lmbda
        l22 = (1.0 - alpha) * lmbda

    uProbs = uPrior.probs()
    uVals = uPrior.vals()
    vProbs = vPrior.probs()
    vVals = vPrior.vals()
    uMom2 = uPrior.secondMoment()
    vMom2 = vPrior.secondMoment()
    q0 = np.array([0.25, 0.25, 0.25, 0.25])
    res0 = root(discreteLimitsGrad, q0, args = (uProbs, uVals, vProbs, vVals, uMom2, vMom2, l11, l12, l22))
    q0s = res0.x
    # assert (q[0] >= 0.0 or abs(q[0]) < 1e-7) and (q[1] >= 0.0 or abs(q[1]) < 1e-7), f"Qt was infeasable -> q1t = {q[0]} and q2t = {q[1]}"
    q0s = np.clip(q0s, 0.0, 0.5)
    q10, q20 = q0s[2], q0s[3]
    mse0 = 1.0 - q10**2 - q20**2 - 2.0 * q10 * q20
    if (np.linalg.norm(res0.fun) > 1e-9):
        # Don't trust MMSE value here
        print('Root did not meet tolerance criteria')
        mse0 = 1.0

    q1 = np.array([0.45, 0.45, 0.45, 0.45])
    res1 = root(discreteLimitsGrad, q1, args = (uProbs, uVals, vProbs, vVals, uMom2, vMom2, l11, l12, l22))
    q1s = res1.x
    q1s = np.clip(q1s, 0.0, 0.5)
    q11, q21 = q1s[2], q1s[3]
    mse1 = 1.0 - q11**2 - q21**2 - 2.0 * q11 * q21
    if (np.linalg.norm(res1.fun) > 1e-9):
        # Don't trust MMSE value here
        print('Root did not meet tolerance criteria')
        mse1 = 1.0

    mse = min(mse0, mse1)
    print(f'MSE value was: {mse}')
    return [mse]

def getMixedMatrixMSE(u: np.ndarray,
                      v: np.ndarray,
                      uHat: np.ndarray,
                      vHat: np.ndarray) -> float:
    assert u.shape == uHat.shape, "U and Uhat must be the same shape"
    assert v.shape == vHat.shape, "V and Vhat must be the same shape"

    assert not np.any(np.isnan(uHat)), "uHat had a Nan"
    assert not np.any(np.isnan(vHat)), "vHat had a Nan"
    assert not np.any(np.isnan(u)), "u had a Nan"
    assert not np.any(np.isnan(v)), "v had a Nan"

    if (np.any(np.isnan(uHat))):
        print('Found a Nan in u estimate')
        return 2.0

    if (np.any(np.isnan(vHat))):
        print('Found a Nan in v estimate')
        return 2.0

    nu = u.shape[0]
    nv = v.shape[0]
    nDims = float(nu + nv)
    scale = 2.0 / (nDims**2)

    # Make estimators unit norm
    uHatn = uHat / np.linalg.norm(uHat)
    vHatn = vHat / np.linalg.norm(vHat)

    uNorm2 = (u.T @ u)[0, 0]
    vNorm2 = (v.T @ v)[0, 0]
    uCorr2 = (abs((u.T @ uHatn)[0, 0]))**2
    vCorr2 = (abs((v.T @ vHatn)[0, 0]))**2
    return scale * (uNorm2**2 + vNorm2**2 + 2.0 * uNorm2 * vNorm2 - uCorr2**2 - vCorr2**2 - 2.0 * uCorr2 * vCorr2)

def getSVDMSE(alpha: float,
              lmbda: float,
              uPrior: MatrixPrior,
              vPrior: MatrixPrior) -> float:
    nu, du = uPrior.dimension()
    nv, dv = vPrior.dimension()
    A, u, v = getMixedMatrix(lmbda, alpha, uPrior, vPrior)
    # xhat = getSVDRecon(A)
    xhat = getLeadingEigenVector(A)
    uHat = xhat[:nu]
    vHat = xhat[nu:]
    return 0.5 * getMixedMatrixMSE(u, v, uHat, vHat)

def getEigenMSE(alpha: float,
                lmbda: float,
                uPrior: MatrixPrior,
                vPrior: MatrixPrior) -> float:
    nu, du = uPrior.dimension()
    nv, dv = vPrior.dimension()
    A, u, v = getMixedMatrix(lmbda, alpha, uPrior, vPrior)
    xHat = getLeadingEigenVector(A)
    uHat = xHat[:nu]
    vHat = xHat[nu:]
    return 0.5 * getMixedMatrixMSE(u, v, uHat, vHat)

def getMixedMatrix(lmbda: float,
                   alpha: float,
                   uPrior: MatrixPrior,
                   vPrior: MatrixPrior,
                   sym: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nu, du = uPrior.dimension()
    nv, dv = vPrior.dimension()
    assert du == dv, "Rank of U and V matrices must match"

    nDims = nu + nv
    scale = 1.0 / sqrt(float(nDims))
    # scale = 1.0 /sqrt(nu)

    lmbda11 = (1.0 - alpha) * lmbda
    lmbda12 = alpha * lmbda
    U = uPrior.sample()
    V = vPrior.sample()
    W = np.random.normal(loc = 0.0, scale = 1.0, size = (nDims, nDims))
    A = scale * np.block([[sqrt(lmbda11) * (U @ U.T), sqrt(lmbda12) * (U @ V.T)], [sqrt(lmbda12) * (V @ U.T), sqrt(lmbda11) * (V @ V.T)]])
    # A = np.block([[sqrt(lmbda11 / float(nu)) * (U @ U.T), sqrt(lmbda12 / float(nu)) * (U @ V.T)], [sqrt(lmbda12 / float(nu)) * (V @ U.T), sqrt(lmbda11 / float(nv)) * (V @ V.T)]])
    if sym:
        return symmetrize(A + W), U, V
    else:
        return A + W, U, V

def getCorr(x: np.ndarray,
            xHat: np.ndarray) -> float:
    corr = abs((x.T @ xHat)[0, 0]) / (np.linalg.norm(x) * np.linalg.norm(xHat))
    if (np.isnan(corr)):
        return 0.0

    return corr

def runFunc(func: Callable[[Any], None],
            arglist: np.ndarray,
            nOutputs: int,
            nMonteCarlo: Optional[int] = None,
            outFile: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    nArgs = arglist.shape[0]
    outputs = np.empty((nArgs, nOutputs + 1))
    for idx in range(nArgs):
        arg = arglist[idx]
        print(f'Argument was: {arg}')
        outputs[idx, 0] = arg
        if (nMonteCarlo is not None):
            outSum = np.zeros((nOutputs, ))
            for _ in range(nMonteCarlo):
                outSum = outSum + np.asarray(func(arg))

            out = outSum / float(nMonteCarlo)

        else:
            out = np.asarray(func(arg))

        outputs[idx, 1:] = out

    if (outFile is not None):
        outputs.tofile(outFile)

    return outputs

def getEigCorr(alpha: float,
               lmbda: float,
               uPrior: MatrixPrior,
               vPrior: MatrixPrior) -> Tuple[float, float]:
    nu, _ = uPrior.dimension()
    nv, _ = vPrior.dimension()
    A, u, v = getMixedMatrix(lmbda, alpha, uPrior, vPrior)
    # xHat = getLeadingEigenVector(A)
    xHat, _ = getLeadingSingularVectors(A)
    # w, X = np.linalg.eig(A)
    # return getCorr(u, X[:nu, 0].reshape(nu, 1)), getCorr(v, X[nu:, 0].reshape(nv, 1))
    return getCorr(u, xHat[:nu]), getCorr(v, xHat[nu:])

def equalizeMatrix(alpha: float,
                   lmbda: float,
                   nDimsU: int,
                   A: np.ndarray,
                   tol: float = 1e-16) -> np.ndarray:
    m, n = A.shape
    out = np.empty((m, n))
    lmbda11 = (1.0 - alpha) * lmbda
    lmbda12 = alpha * lmbda
    out[:nDimsU, :nDimsU] = (1.0 / (sqrt(lmbda11) + tol)) * A[:nDimsU, :nDimsU]
    out[nDimsU:, nDimsU:] = (1.0 / (sqrt(lmbda11) + tol)) * A[nDimsU:, nDimsU:]
    out[:nDimsU, nDimsU:] = (1.0 / (sqrt(lmbda12) + tol)) * A[:nDimsU, nDimsU:]
    out[nDimsU:, :nDimsU] = (1.0 / (sqrt(lmbda12) + tol)) * A[nDimsU:, :nDimsU]
    return out

def getSVDCorrEqual(alpha: float,
                    lmbda: float,
                    uPrior: MatrixPrior,
                    vPrior: MatrixPrior) -> Tuple[float, float]:
    nu, _ = uPrior.dimension()
    A, u, v = getMixedMatrix(lmbda, alpha, uPrior, vPrior)
    Aeq = equalizeMatrix(alpha, lmbda, nu, A)
    xHat, _ = getLeadingSingularVectors(Aeq)
    # xHat = getLeadingEigenVector(Aeq)
    return getCorr(u, xHat[:nu]), getCorr(v, xHat[nu:])

def getLeadingEigenVector(A: np.ndarray) -> np.ndarray:
    m, n = A.shape
    assert m == n, "Cannot get Eigenvectors on a non-square matrix"
    # w, V = np.linalg.eig(A)
    _, V = eigh(A, subset_by_index = (n - 1, n - 1))
    return V
    # arg = np.argmax(np.abs(w))
    # return V[:, arg].reshape(m, 1)

def getEigenRecon(A: np.ndarray) -> np.ndarray:
    m, n = A.shape
    assert m == n, "Cannot get Eigenvectors on a non-square matrix"
    w, V = np.linalg.eig(A)
    arg = np.argmax(np.abs(w))
    lev = V[:, arg].reshape(m, 1)
    return w[arg] * (lev @ lev.T)

def getLeadingSingularVectors(A: np.ndarray) -> Tuple[np.ndarray]:
    U, s, Vh = randomized_svd(A, 1)
    # U, s, Vh = np.linalg.svd(A)
    return U[:, 0].reshape(U.shape[0], 1), Vh[0, :].reshape(Vh.shape[1], 1)

def getSVDRecon(A: np.ndarray) -> np.ndarray:
    U, s, Vh = np.linalg.svd(A)
    return sqrt(s[0]) * U[:, 0].reshape(U.shape[0], 1)

def getIndividualCorr(alpha: float,
                      lmbda: float,
                      uPrior: MatrixPrior,
                      vPrior: MatrixPrior) -> Tuple[float, float, float, float, float, float]:
    nu, _ = uPrior.dimension()
    A, u, v = getMixedMatrix(lmbda, alpha, uPrior, vPrior)
    A11, A12, A21, A22 = A[:nu, :nu], A[:nu, nu:], A[nu:, :nu], A[nu:, nu:]
    uHat1 = getLeadingEigenVector(A11)
    uHat2, vHat1 = getLeadingSingularVectors(A12)
    vHat2, uHat3 = getLeadingSingularVectors(A21)
    vHat3 = getLeadingEigenVector(A22)
    return (getCorr(u, uHat1),
            getCorr(u, uHat2),
            getCorr(u, uHat3),
            getCorr(v, vHat1),
            getCorr(v, vHat2),
            getCorr(v, vHat3))

def prepForSum(x: np.ndarray,
               xref: np.ndarray) -> np.ndarray:
    # First check to see if correlation is negative
    if ((x.T @ xref)[0, 0] < 0.0):
        return -x / np.linalg.norm(x)

    return x / np.linalg.norm(x)

def getSNRWeightedMSE(alpha: float,
                      lmbda: float,
                      uPrior: MatrixPrior,
                      vPrior: MatrixPrior) -> float:
    nu, _ = uPrior.dimension()
    A, u, v = getMixedMatrix(lmbda, alpha, uPrior, vPrior)
    A11, A12, A21, A22 = A[:nu, :nu], A[:nu, nu:], A[nu:, :nu], A[nu:, nu:]
    uHat1, _ = getLeadingSingularVectors(A11)
    uHat2, vHat1 = getLeadingSingularVectors(A12)
    vHat2, uHat3 = getLeadingSingularVectors(A21)
    vHat3 = getLeadingEigenVector(A22)
    uHat1 = prepForSum(uHat1, u)
    uHat2 = prepForSum(uHat2, u)
    uHat3 = prepForSum(uHat3, u)
    vHat1 = prepForSum(vHat1, v)
    vHat2 = prepForSum(vHat2, v)
    vHat3 = prepForSum(vHat3, v)
    w11 = (1.0 - alpha) / (1.0 + alpha)
    w12 = alpha / (1.0 + alpha)
    uHat = w11 * uHat1 + w12 * (uHat2 + uHat3)
    vHat = w11 * vHat3 + w12 * (vHat1 + vHat2)
    return 0.5 * getMixedMatrixMSE(u, v, uHat, vHat)

def getChoiceMSE(alpha: float,
                 lmbda: float,
                 uPrior: MatrixPrior,
                 vPrior: MatrixPrior) -> float:
    nu, _ = uPrior.dimension()
    A, u, v = getMixedMatrix(lmbda, alpha, uPrior, vPrior)
    A11, A12, A21, A22 = A[:nu, :nu], A[:nu, nu:], A[nu:, :nu], A[nu:, nu:]
    if alpha <= 0.25:
        # Use eigenvectors of diagonals
        uHat = getLeadingEigenVector(A11)
        vHat = getLeadingEigenVector(A22)

    elif alpha > 0.25 and alpha < 0.75:
        xhat = getLeadingEigenVector(A)
        uHat = xhat[:nu]
        vHat = xhat[nu:]
    else:
        uHat, vHat = getLeadingSingularVectors(A12)
        # vHat2, uHat2 = getLeadingSingularVectors(A21)
        # uHat1 = prepForSum(uHat1, u)
        # uHat2 = prepForSum(uHat2, u)
        # vHat1 = prepForSum(vHat1, v)
        # vHat2 = prepForSum(vHat2, v)
        # uHat = uHat1 + uHat2
        # vHat = vHat1 + vHat2

    return 0.5 * getMixedMatrixMSE(u, v, uHat, vHat)

def getSNRWeightedCorr(alpha: float,
                       lmbda: float,
                       uPrior: MatrixPrior,
                       vPrior: MatrixPrior) -> Tuple[float, float]:
    nu, _ = uPrior.dimension()
    A, u, v = getMixedMatrix(lmbda, alpha, uPrior, vPrior)
    A11, A12, A21, A22 = A[:nu, :nu], A[:nu, nu:], A[nu:, :nu], A[nu:, nu:]
    uHat1 = getLeadingEigenVector(A11)
    uHat2, vHat1 = getLeadingSingularVectors(A12)
    vHat2, uHat3 = getLeadingSingularVectors(A21)
    vHat3 = getLeadingEigenVector(A22)
    uHat1 = prepForSum(uHat1, uHat1)
    uHat2 = prepForSum(uHat2, uHat1)
    uHat3 = prepForSum(uHat3, uHat1)
    vHat1 = prepForSum(vHat1, vHat1)
    vHat2 = prepForSum(vHat2, vHat1)
    vHat3 = prepForSum(vHat3, vHat1)
    w11 = (1.0 - alpha) / (1.0 + alpha)
    w12 = alpha / (1.0 + alpha)
    # if (w11 >= w12):
    #     uHat = uHat1
    #     vHat = vHat3
    # else:
    #     uHat = uHat2
    #     vHat = vHat1
    uHat = w11 * uHat1 + w12 * (uHat2 + uHat3)
    vHat = w11 * vHat3 + w12 * (vHat1 + vHat2)
    return getCorr(u, uHat), getCorr(v, vHat)

def mixedMatrixObjective(x: np.ndarray,
                         A: np.ndarray,
                         nDimsU: int,
                         nDimsV: int,
                         lmbda11: float,
                         lmbda12: float,
                         lmbda22: float) -> float:
    xc = x.reshape(-1, 1)
    u, v = xc[:nDimsU], xc[nDimsU:]
    assert u.shape[0] == nDimsU, "Dimensions of u were incorrect"
    assert u.shape[1] == 1, "U must be a vector"
    assert v.shape[0] == nDimsV, "Dimensions of v were incorrect"
    assert v.shape[1] == 1, "V must be a vector"

    assert A.shape[0] == nDimsU + nDimsV, "Outer dimension of A was inconsistent"
    assert A.shape[1] == nDimsU + nDimsV, "Inner dimension of A was inconsistent"

    lmbda21 = lmbda12

    A11 = lmbda11 * (u @ u.T)
    A12 = lmbda12 * (u @ v.T)
    A21 = lmbda21 * (v @ u.T)
    A22 = lmbda22 * (v @ v.T)
    Ahat = np.block([[A11, A12], [A21, A22]])
    return ((A - Ahat)**2).sum()

def mixedMatrixObjectiveEqual(x: np.ndarray,
                              A: np.ndarray,
                              nDimsU: int,
                              nDimsV: int) -> float:
    xc = x.reshape(-1, 1)
    u, v = xc[:nDimsU], xc[nDimsU:]
    assert u.shape[0] == nDimsU, "Dimensions of u were incorrect"
    assert u.shape[1] == 1, "U must be a vector"
    assert v.shape[0] == nDimsV, "Dimensions of v were incorrect"
    assert v.shape[1] == 1, "V must be a vector"

    assert A.shape[0] == nDimsU + nDimsV, "Outer dimension of A was inconsistent"
    assert A.shape[1] == nDimsU + nDimsV, "Inner dimension of A was inconsistent"

    Ahat = xc @ xc.T
    return ((A - Ahat)**2).sum()

def mixedMatrixGrad(x: np.ndarray,
                    A: np.ndarray,
                    nDimsU: int,
                    nDimsV: int,
                    lmbda11: float,
                    lmbda12: float,
                    lmbda22: float) -> np.ndarray:
    xc = x.reshape(-1, 1)
    u, v = xc[:nDimsU], xc[nDimsU:]
    assert u.shape[0] == nDimsU, "Dimensions of u were incorrect"
    assert u.shape[1] == 1, "U must be a vector"
    assert v.shape[0] == nDimsV, "Dimensions of v were incorrect"
    assert v.shape[1] == 1, "V must be a vector"

    assert A.shape[0] == nDimsU + nDimsV, "Outer dimension of A was inconsistent"
    assert A.shape[1] == nDimsU + nDimsV, "Inner dimension of A was inconsistent"

    lmbda21 = lmbda12

    A11, A12, A21, A22 = A[:nDimsU, :nDimsU], A[:nDimsU, nDimsU:], A[nDimsU:, :nDimsU], A[nDimsU:, nDimsU:]
    uNorm = (u.T @ u)[0, 0]
    vNorm = (v.T @ v)[0, 0]
    uu = u @ u.T
    uv = u @ v.T
    vu = v @ u.T
    vv = v @ v.T

    output = np.empty((nDimsU + nDimsV, 1))
    uTerm1 = -2.0 * lmbda11 * ((A11 + A11.T) @ u)
    uTerm2 = -2.0 * ((lmbda12 * A12 + lmbda21 * A21.T) @ v)
    uTerm3 = 2.0 * (2.0 * lmbda11**2 * uNorm + (lmbda12**2 + lmbda21**2) * vNorm) * u
    output[0:nDimsU] = uTerm1 + uTerm2 + uTerm3
    vTerm1 = -2.0 * lmbda22 * ((A22 + A22.T) @ v)
    vTerm2 = -2.0 * ((lmbda12 * A12.T + lmbda21 * A21) @ u)
    vTerm3 = 2.0 * (2.0 * lmbda22**2 * vNorm + (lmbda12**2 + lmbda21**2) * uNorm) * v
    output[nDimsU:] = vTerm1 + vTerm2 + vTerm3
    # output[-4] = 2.0 * lmbda11 * (uu**2).sum() - 2.0 * (A11 * uu).sum()
    # output[-3] = 2.0 * lmbda12 * (uv**2).sum() - 2.0 * (A12 * uv).sum()
    # output[-2] = 2.0 * lmbda21 * (vu**2).sum() - 2.0 * (A21 * vu).sum()
    # output[-1] = 2.0 * lmbda22 * (vv**2).sum() - 2.0 * (A22 * vv).sum()
    return output.reshape(-1)

def mixedMatrixGradEqual(x: np.ndarray,
                         A: np.ndarray,
                         nDimsU: int,
                         nDimsV: int) -> np.ndarray:
    xc = x.reshape(-1, 1)
    u, v = xc[:nDimsU], xc[nDimsU:]
    assert u.shape[0] == nDimsU, "Dimensions of u were incorrect"
    assert u.shape[1] == 1, "U must be a vector"
    assert v.shape[0] == nDimsV, "Dimensions of v were incorrect"
    assert v.shape[1] == 1, "V must be a vector"

    assert A.shape[0] == nDimsU + nDimsV, "Outer dimension of A was inconsistent"
    assert A.shape[1] == nDimsU + nDimsV, "Inner dimension of A was inconsistent"

    A11, A12, A21, A22 = A[:nDimsU, :nDimsU], A[:nDimsU, nDimsU:], A[nDimsU:, :nDimsU], A[nDimsU:, nDimsU:]
    uNorm = (u.T @ u)[0, 0]
    vNorm = (v.T @ v)[0, 0]
    uu = u @ u.T
    uv = u @ v.T
    vu = v @ u.T
    vv = v @ v.T

    output = np.empty((nDimsU + nDimsV, 1))
    uTerm1 = -2.0 * ((A11 + A11.T) @ u)
    uTerm2 = -2.0 * (( A12 + A21.T) @ v)
    uTerm3 = 2.0 * (2.0 * uNorm + 2.0 * vNorm) * u
    output[0:nDimsU] = uTerm1 + uTerm2 + uTerm3
    vTerm1 = -2.0 * ((A22 + A22.T) @ v)
    vTerm2 = -2.0 * ((A12.T + A21) @ u)
    vTerm3 = 2.0 * (2.0 * vNorm + 2.0 * uNorm) * v
    output[nDimsU:] = vTerm1 + vTerm2 + vTerm3
    return output.reshape(-1)

def getSpectralMSE(alpha: float,
                   lmbda: float,
                   uPrior: MatrixPrior,
                   vPrior: MatrixPrior) -> Tuple[float, float]:
    nu, _ = uPrior.dimension()
    nv, _ = vPrior.dimension()
    A, u, v = getMixedMatrix(lmbda, alpha, uPrior, vPrior)
    lmbda11 = (1.0 - alpha) * lmbda
    lmbda12 = alpha * lmbda
    # if (alpha < 0.5):
    #     uInit, _ = getLeadingSingularVectors(A[:nu, :nu])
    #     vInit, _ = getLeadingSingularVectors(A[nu:, nu:])
    # else:
    #     uInit, vInit = getLeadingSingularVectors(A[:nu, nu:])
    # xInit = np.empty((nu + nv, 1))
    # xInit[0:nu] = uInit
    # xInit[nu:] = vInit
    xInit = np.random.normal(size = (nu + nv, 1))
    objFunc = partial(mixedMatrixObjective, A = A, nDimsU = nu, nDimsV = nv, lmbda11 = sqrt(lmbda11), lmbda12 = sqrt(lmbda12), lmbda22 = sqrt(lmbda11))
    gradFunc = partial(mixedMatrixGrad, A = A, nDimsU = nu, nDimsV = nv, lmbda11 = sqrt(lmbda11), lmbda12 = sqrt(lmbda12), lmbda22 = sqrt(lmbda11))
    res = minimize(objFunc, xInit, method = 'BFGS', jac = gradFunc)
    xhat = (res.x).reshape(-1, 1)
    uHat = xhat[:nu]
    vHat = xhat[nu:]
    return 0.5 * getMixedMatrixMSE(u, v, uHat, vHat)

def getSpectralCorrEqual(alpha: float,
                         lmbda: float,
                         uPrior: MatrixPrior,
                         vPrior: MatrixPrior) -> Tuple[float, float]:
    nu, _ = uPrior.dimension()
    nv, _ = vPrior.dimension()
    A, u, v = getMixedMatrix(lmbda, alpha, uPrior, vPrior)
    Aeq = equalizeMatrix(alpha, lmbda, nu, A)
    lmbda11 = (1.0 - alpha) * lmbda
    lmbda12 = alpha * lmbda
    if (alpha < 0.5):
        uInit, _ = getLeadingSingularVectors(A[:nu, :nu])
        vInit, _ = getLeadingSingularVectors(A[nu:, nu:])
    else:
        uInit, vInit = getLeadingSingularVectors(A[:nu, nu:])
    xInit = np.empty((nu + nv, 1))
    xInit[0:nu] = uInit
    xInit[nu:] = vInit
    objFunc = partial(mixedMatrixObjectiveEqual, A = Aeq, nDimsU = nu, nDimsV = nv)
    gradFunc = partial(mixedMatrixGradEqual, A = Aeq, nDimsU = nu, nDimsV = nv)
    res = minimize(objFunc, xInit, method = 'BFGS', jac = gradFunc)
    x = (res.x).reshape(-1, 1)
    uHat = x[:nu]
    vHat = x[nu:]
    return getCorr(u, uHat), getCorr(v, vHat)

@jit(nopython = True, cache = True)
def normalInnerObj(q: np.ndarray,
                   l11: float,
                   l12: float,
                   l22: float) -> float:
    assert q.shape[0] == 4, "Q should only have 4 entries"

    q1t, q2t, q1, q2 = q[0], q[1], q[2], q[3]
    su = l11 * (q1 + q1t) + l12 * (q2 + q2t)
    sv = l22 * (q2 + q2t) + l12 * (q1 + q1t)
    c1 = (l11 * q1 + l12 * q2) * q1t
    c2 = (l12 * q1 + l22 * q2) * q2t

    return 0.25 * (su - log(1.0 + su)) + 0.25 * (sv - log(1.0 + sv)) - 0.5 * (c1 + c2)

@jit(nopython = True, cache = True)
def normalFundLimitsGrad(q: np.ndarray,
                         l11: float,
                         l12: float,
                         l22: float) -> np.ndarray:
    assert q.shape[0] == 4, "Q should only have 4 entries"

    q1t, q2t, q1, q2 = q[0], q[1], q[2], q[3]
    su = l11 * (q1 + q1t) + l12 * (q2 + q2t)
    sv = l22 * (q2 + q2t) + l12 * (q1 + q1t)

    suf = 1.0 - (1.0 / (1.0 + su))
    svf = 1.0 - (1.0 / (1.0 + sv))

    grad = np.empty((4,))
    grad[0] = 0.25 * (l11 * suf + l12 * svf) - 0.5 * (l11 * q1 + l12 * q2)
    grad[1] = 0.25 * (l12 * suf + l22 * svf) - 0.5 * (l12 * q1 + l22 * q2)
    grad[2] = 0.25 * (l11 * suf + l12 * svf) - 0.5 * (l11 * q1t + l12 * q2t)
    grad[3] = 0.25 * (l12 * suf + l22 * svf) - 0.5 * (l12 * q1t + l22 * q2t)
    return grad

@jit(nopython = True, cache = True)
def normalFundLimitsJac(q: np.ndarray,
                        l11: float,
                        l12: float,
                        l22: float) -> np.ndarray:
    assert q.shape[0] == 4, "Q should only have 4 entries"

    q1t, q2t, q1, q2 = q[0], q[1], q[2], q[3]
    su = l11 * (q1 + q1t) + l12 * (q2 + q2t)
    sv = l22 * (q2 + q2t) + l12 * (q1 + q1t)

    su2 = (1.0 / (1.0 + su))**2
    sv2 = (1.0 / (1.0 + sv))**2

    jac = np.empty((4, 4))
    jac[0, 0] = 0.25 * ((l11**2) * su2 + (l12**2) * sv2)
    jac[1, 1] = 0.25 * ((l12**2) * su2 + (l22**2) * sv2)
    jac[2, 2] = 0.25 * ((l11**2) * su2 + (l12**2) * sv2)
    jac[3, 3] = 0.25 * ((l12**2) * su2 + (l22**2) * sv2)
    jac[0, 1] = 0.25 * (l11 * l12 * su2 + l22 * l12 * sv2)
    jac[1, 0] = 0.25 * (l11 * l12 * su2 + l22 * l12 * sv2)
    jac[0, 2] = 0.25 * ((l11**2) * su2 + (l12**2) * sv2) - 0.5 * l11
    jac[2, 0] = 0.25 * ((l11**2) * su2 + (l12**2) * sv2) - 0.5 * l11
    jac[0, 3] = 0.25 * (l11 * l12 * su2 + l22 * l12 * sv2) - 0.5 * l12
    jac[3, 0] = 0.25 * (l11 * l12 * su2 + l22 * l12 * sv2) - 0.5 * l12
    jac[1, 2] = 0.25 * (l11 * l12 * su2 + l22 * l12 * sv2) - 0.5 * l12
    jac[2, 1] = 0.25 * (l11 * l12 * su2 + l22 * l12 * sv2) - 0.5 * l12
    jac[1, 3] = 0.25 * ((l12**2) * su2 + (l22**2) * sv2) - 0.5 * l22
    jac[3, 1] = 0.25 * ((l12**2) * su2 + (l22**2) * sv2) - 0.5 * l22
    jac[2, 3] = 0.25 * (l11 * l12 * su2 + l22 * l12 * sv2)
    jac[3, 2] = 0.25 * (l11 * l12 * su2 + l22 * l12 * sv2)
    return jac

def getNormalFundLimits(alpha: float,
                        lmbda: float) -> Tuple[float]:
    l11 = (1.0 - alpha) * lmbda
    l12 = alpha * lmbda
    l22 = (1.0 - alpha) * lmbda
    bounds = Bounds([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 0.5, 0.5])
    q0 = np.array([0.25, 0.25, 0.25, 0.25])
    res = root(normalFundLimitsGrad, q0, jac = normalFundLimitsJac, args = (l11, l12, l22))
    q = res.x
    assert (q[0] >= 0.0 or abs(q[0]) < 1e-7) and (q[1] >= 0.0 or abs(q[1]) < 1e-7), f"Qt was infeasable -> q1t = {q[0]} and q2t = {q[1]}"
    q = np.clip(q, 0.0, 0.5)
    q1, q2 = q[2], q[3]
    mse = 1.0 - q1**2 - q2**2 - 2.0 * q1 * q2
    print(f'MSE value was: {mse} \nq1: {q1} q2: {q2}')
    return [mse]

def normalOuterObj(q: np.ndarray,
                   l11: float,
                   l12: float,
                   l22: float,
                   qtStore: Optional[Dict[float, np.ndarray]] = None) -> float:
    normalBounds = Bounds([0.0, 0.0], [np.inf, np.inf])
    qt0 = np.array([0.25, 0.25])
    # normalBounds = Bounds([0.0, 0.0], [1.0, 1.0])
    # qt0 = np.array([0.5, 0.5])
    res = minimize(normalInnerObj, qt0, method = 'trust-constr', jac = normalFundLimitsGrad, hess = normalFundLimitsJac,
                   constraints = list(), bounds = normalBounds, args = (q, l11, l12, l22))
    if (qtStore is not None):
        qtStore[-1.0 * res.fun] = res.x

    return -1.0 * res.fun

def getNormalLandscape(nqs: int,
                       l11: float,
                       l12: float,
                       l22: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q1s = np.linspace(0.0, 0.5, nqs)
    q2s = np.linspace(0.0, 0.5, nqs)
    Q1, Q2 = np.meshgrid(q1s, q2s)
    q1 = Q1.flatten()
    q2 = Q2.flatten()
    objFunc = partial(normalOuterObj, l11 = l11, l12 = l12, l22 = l22)
    objVals = list(map(lambda x: objFunc(np.array([x[0], x[1]])), zip(q1, q2)))
    z = np.asarray(objVals)
    Z = z.reshape(Q1.shape)
    return Q1, Q2, Z

def alphaZeroObj(q: np.ndarray,
                 lmbda: float) -> float:
    q1, q2 = q[0], q[1]
    q1s = max(0, (1.0 / (lmbda * (1.0 - 2.0 * q1))) - q1)
    q2s = max(0, (1.0 / (lmbda * (1.0 - 2.0 * q2))) - q2)
    linTerm = lmbda * (q1 + q1s + q2 + q2s) - 2.0 - 2.0 * lmbda * (q1 * q1s + q2 * q2s)
    logTerm = log(q1 + q1s) + log(q2 + q2s) + log(lmbda**2)
    return -0.5 * (linTerm - logTerm)

def getAlphaZeroLimits(lmbda: float) -> Tuple[float]:
    bounds = [(0.0, 0.5), (0.0, 0.5)]
    q0 = np.array([0.1, 0.1])
    res = shgo(alphaZeroObj, bounds = bounds, args = (lmbda,), iters = 50)
    q = res.x
    q1, q2 = q[0], q[1]
    mse = 1.0 - q1 - q2
    print(f'MSE was {mse} q1: {q1} q2: {q2}')
    return [mse]

class GaussDenoiser:
    def __init__(self,
                 epsilon: float,
                 nIters: int,
                 l11: float,
                 l12: float):
        '''
            @brief Instantiate the object
            @param epsilon The amount of initial correlation to use on (0, 1)
            @param nIters The number of iterations to run
            @param l11 The lambda_11 value
            @param l12 The lambda_12 value
        '''
        self._epsilon = epsilon
        self._nIters = nIters
        self._l11 = l11
        self._l12 = l12

        # Create the parameters over all iterations to be run
        params = list()
        for idx in range(nIters):
            if (idx == 0):
                mc = sqrt(l12) * epsilon
                sc = epsilon
                # sc = 1.0 + epsilon**2
                ms = sqrt(l11) * epsilon
                ss = epsilon
                # ss = 1.0 + epsilon**2
                bc = sqrt(l12) * epsilon
                tc = epsilon
                # tc = 1.0 + epsilon**2
                bs = sqrt(l11) * epsilon
                ts = epsilon
                # ts = 1.0 + epsilon**2

            else:
                mcp, scp, msp, ssp, bcp, tcp, bsp, tsp = params[idx - 1]
                gc = bcp**2 / (bcp**2 + tcp)
                mc = sqrt(l12) * gc
                sc = gc
                gs = (mcp**2 / scp) + (msp**2 / ssp)
                ms = sqrt(l11) * (gs / (1.0 + gs))
                ss = (gs / (1.0 + gs))
                oc = mc**2 / (mc**2 + sc**2)
                bc = sqrt(l12) * oc
                tc = oc
                os = (bcp**2 / tcp) + (bsp**2 / tsp)
                bs = sqrt(l11) * (os / (1.0 + os))
                ts = os / (1.0 + os)

            params.append((mc, sc, ms, ss, bc, tc, bs, ts))

        self._params = params

    def etau(self,
             uc: np.ndarray,
             us: np.ndarray,
             idx: int) -> np.ndarray:
        mc, sc, ms, ss, _, _, _, _ = self._params[idx]
        scale = 1.0 / ((mc**2 * ss) + (ms**2 * sc) + (sc * ss) + 1e-12)
        # scale = 1.0 / (mc**2 + sc + ms**2 + ss + 1e-12)
        return scale * ((ss * mc * uc) + (sc * ms * us))

    def etav(self,
             vc: np.ndarray,
             vs: np.ndarray,
             idx: int) -> np.ndarray:
        _, _, _, _, bc, tc, bs, ts = self._params[idx]
        # scale = 1.0 / (bc**2 + tc + bs**2 + ts + 1e-12)
        scale = 1.0 / ((bc**2 * ts) + (bs**2 * tc) + (tc * ts) + 1e-12)
        return scale * ((ts * bc * vc) + (tc * bs * vs))

    def ucDiv(self,
              uc: np.ndarray,
              us: np.ndarray,
              idx: int) -> float:
        mc, sc, ms, ss, _, _, _, _ = self._params[idx]
        return (ss * mc) / ((mc**2 * ss) + (ms**2 * sc) + (sc * ss) + 1e-12)

    def usDiv(self,
              uc: np.ndarray,
              us: np.ndarray,
              idx: int) -> float:
        mc, sc, ms, ss, _, _, _, _ = self._params[idx]
        return (sc * ms) / ((mc**2 * ss) + (ms**2 * sc) + (sc * ss) + 1e-12)

    def vcDiv(self,
              vc: np.ndarray,
              vs: np.ndarray,
              idx: int) -> float:
        _, _, _, _, bc, tc, bs, ts = self._params[idx]
        return (ts * bc) / ((bc**2 * ts) + (bs**2 * tc) + (tc * ts) + 1e-12)

    def vsDiv(self,
              vc: np.ndarray,
              vs: np.ndarray,
              idx: int) -> float:
        _, _, _, _, bc, tc, bs, ts = self._params[idx]
        return (tc * bs) / ((bc**2 * ts) + (bs**2 * tc) + (tc * ts) + 1e-12)

    def plotSNRs(self) -> None:
        ucSNRs = list()
        usSNRs = list()
        vcSNRs = list()
        vsSNRs = list()
        for mc, sc, ms, ss, bc, tc, bs, ts in self._params:
            ucSNRs.append(mc**2 / sc)
            usSNRs.append(ms**2 / ss)
            vcSNRs.append(bc**2 / tc)
            vsSNRs.append(bs**2 / ts)

        plt.plot(ucSNRs, label = r'$u_c SNR$')
        plt.plot(usSNRs, label = r'$u_s SNR$')
        plt.plot(vcSNRs, label = r'$v_c SNR$')
        plt.plot(vsSNRs, label = r'$v_s SNR$')
        plt.legend()
        plt.grid()
        plt.show()
        return

def GaussAMP(u: np.ndarray,
             v: np.ndarray,
             Yu: np.ndarray,
             Yuv: np.ndarray,
             Yvu: np.ndarray,
             Yv: np.ndarray,
             alpha: float,
             lmbda: float,
             stopTol: float,
             maxIters: int) -> Tuple[np.ndarray, np.ndarray]:
    # Pre-compute the quantities we can
    n = u.shape[0]
    scales = sqrt((1.0 - alpha) * lmbda / (2.0 * float(n)))
    scales2 = scales**2
    scalea = sqrt(alpha * lmbda / (2.0 * float(n)))
    scalea2 = scalea**2
    np.fill_diagonal(Yu, 0.5 * np.diagonal(Yu))
    np.fill_diagonal(Yv, 0.5 * np.diagonal(Yv))
    Yu2 = Yu**2
    Yv2 = Yv**2
    Yuv2 = Yuv**2
    Yvu2 = Yvu**2
    Ru = Yu2 - 1.0
    Rv = Yv2 - 1.0
    Ruv = Yuv2 - 1.0
    Rvu = Yvu2 - 1.0
    uOld = np.zeros((n, 1))
    vOld = np.zeros((n, 1))
    uHat = 1e-3 * np.random.normal(size = (n, 1))
    vHat = 1e-3 * np.random.normal(size = (n, 1))
    sigma = np.zeros((n, 1))
    tau = np.zeros((n, 1))

    # Run the algorithm
    for idx in range(maxIters):
        uHat2 = uHat**2
        vHat2 = vHat**2
        gul = scales * (Yu.T @ uHat) + scalea * (Yvu.T @ vHat) - ((scales2 * (Yu2.T @ sigma) + scalea2 * (Yvu2.T @ tau)) * uOld)
        guq = scales2 * (Yu2.T @ uHat2) - scales2 * (Ru.T @ (uHat2 + sigma)) + scalea2 * (Yvu2.T @ vHat2) - scalea2 * (Rvu.T @ (vHat2 + tau))
        gvl = scalea * (Yuv.T @ uHat) + scales * (Yv.T @ vHat) - ((scalea2 * (Yuv2.T @ sigma) + scales2 * (Yv2.T @ tau)) * vOld)
        gvq = scalea2 * (Yuv2.T @ uHat2) - scalea2 * (Ruv.T @ (uHat2 + sigma)) + scales2 * (Yv2.T @ vHat2) - scales2 * (Rv.T @ (vHat2 + tau))
        uOld = deepcopy(uHat)
        vOld = deepcopy(vHat)
        uHat = gul / (1.0 + guq)
        vHat = gvl / (1.0 + gvq)
        sigma = 1.0 / (1.0 + guq)
        tau = 1.0 / (1.0 + gvq)
        conv = (1.0 / (2.0 * float(n))) * (np.linalg.norm(uHat - uOld) + np.linalg.norm(vHat - vOld))
        if conv <= stopTol:
            # We have converged
            print(f'AMP converged after {idx + 1} iterations')
            break

    return uHat, vHat

def getGaussMSE(alpha: float,
                lmbda: float,
                uPrior: MatrixPrior,
                vPrior: MatrixPrior,
                tol: float = 1e-9,
                maxIters = 256) -> float:
    nu, _ = uPrior.dimension()
    A, u, v = getMixedMatrix(lmbda, alpha, uPrior, vPrior)
    Yu, Yuv, Yvu, Yv = A[:nu, :nu], A[:nu, nu:], A[nu:, :nu], A[nu:, nu:]
    uHat, vHat = GaussAMP(u, v, Yu, Yuv, Yvu, Yv, alpha, lmbda, tol, maxIters)
    return 0.5 * getMixedMatrixMSE(u, v, uHat, vHat)

def discreteAMP(n: int,
                au: np.ndarray,
                pu: np.ndarray,
                av: np.ndarray,
                pv: np.ndarray,
                Yu: np.ndarray,
                Yuv: np.ndarray,
                Yvu: np.ndarray,
                Yv: np.ndarray,
                alpha: float,
                lmbda: float,
                stopTol: float,
                maxIters: int) -> Tuple[np.ndarray, np.ndarray]:
    # Pre-compute the quantities we can
    scales = sqrt((1.0 - alpha) * lmbda / (2.0 * float(n)))
    scales2 = scales**2
    scalea = sqrt(alpha * lmbda / (2.0 * float(n)))
    scalea2 = scalea**2
    np.fill_diagonal(Yu, 0.5 * np.diagonal(Yu))
    np.fill_diagonal(Yv, 0.5 * np.diagonal(Yv))
    Yu2 = Yu**2
    Yv2 = Yv**2
    Yuv2 = Yuv**2
    Yvu2 = Yvu**2
    Ru = Yu2 - 1.0
    Rv = Yv2 - 1.0
    Ruv = Yuv2 - 1.0
    Rvu = Yvu2 - 1.0
    au2 = au**2
    Au = np.tile(au.T, (n, 1))
    Pu = np.tile(pu.T, (n, 1))
    APu = Au * Pu
    APu2 = (Au**2) * Pu
    av2 = av**2
    Av = np.tile(av.T, (n, 1))
    Pv = np.tile(pv.T, (n, 1))
    APv = Av * Pv
    APv2 = (Av**2) * Pv
    uOld = np.zeros((n, 1))
    vOld = np.zeros((n, 1))
    uHat = 1e-3 * np.random.normal(size = (n, 1))
    vHat = 1e-3 * np.random.normal(size = (n, 1))
    sigma = np.zeros((n, 1))
    tau = np.zeros((n, 1))

    # Run the algorithm
    for idx in range(maxIters):
        uHat2 = uHat**2
        vHat2 = vHat**2
        gul = scales * (Yu.T @ uHat) + scalea * (Yvu.T @ vHat) - ((scales2 * (Yu2.T @ sigma) + scalea2 * (Yvu2.T @ tau)) * uOld)
        guq = scales2 * (Yu2.T @ uHat2) - scales2 * (Ru.T @ (uHat2 + sigma)) + scalea2 * (Yvu2.T @ vHat2) - scalea2 * (Rvu.T @ (vHat2 + tau))
        gvl = scalea * (Yuv.T @ uHat) + scales * (Yv.T @ vHat) - ((scalea2 * (Yuv2.T @ sigma) + scales2 * (Yv2.T @ tau)) * vOld)
        gvq = scalea2 * (Yuv2.T @ uHat2) - scalea2 * (Ruv.T @ (uHat2 + sigma)) + scales2 * (Yv2.T @ vHat2) - scales2 * (Rv.T @ (vHat2 + tau))
        uOld = deepcopy(uHat)
        vOld = deepcopy(vHat)
        uHat, sigma = discreteEta(guq, gul, au, au2, Au, Pu, APu, APu2)
        vHat, tau = discreteEta(gvq, gvl, av, av2, Av, Pv, APv, APv2)
        conv = (1.0 / (2.0 * float(n))) * (np.linalg.norm(uHat - uOld) + np.linalg.norm(vHat - vOld))
        if conv <= stopTol:
            # We have converged
            print(f'AMP converged after {idx + 1} iterations')
            break

    return uHat, vHat

def discreteEta(gammaq: np.ndarray,
                gammal: np.ndarray,
                a: np.ndarray,
                a2: np.ndarray,
                A: np.ndarray,
                P: np.ndarray,
                AP: np.ndarray,
                AP2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    expon = np.exp((gammal @ a.T) - 0.5 * (gammaq @ a2.T))
    num = ((AP * expon).sum(axis = 1)).reshape(-1, 1)
    den = ((P * expon).sum(axis = 1)).reshape(-1, 1)
    xHat = num / den
    num = ((AP2 * expon).sum(axis = 1)).reshape(-1, 1)
    xDiv = (num / den) - (xHat**2)
    return xHat, xDiv

def getDiscreteMSE(alpha: float,
                   lmbda: float,
                   uPrior: MatrixPrior,
                   vPrior: MatrixPrior,
                   tol: float = 1e-9,
                   maxIters = 256) -> float:
    nu, _ = uPrior.dimension()
    A, u, v = getMixedMatrix(lmbda, alpha, uPrior, vPrior)
    Yu, Yuv, Yvu, Yv = A[:nu, :nu], A[:nu, nu:], A[nu:, :nu], A[nu:, nu:]
    uHat, vHat = discreteAMP(nu,
                             (uPrior.vals()).reshape(-1, 1),
                             (uPrior.probs()).reshape(-1, 1),
                             (vPrior.vals()).reshape(-1, 1),
                             (vPrior.probs()).reshape(-1, 1),
                             Yu, Yuv, Yvu, Yv, alpha, lmbda, tol, maxIters)
    return 0.5 * getMixedMatrixMSE(u, v, uHat, vHat)

@jit(nopython = True, cache = True)
def projGradDescObj(beta: float,
                    scales: float,
                    scalea: float,
                    nDims: int,
                    A: np.ndarray,
                    x: np.ndarray,
                    grad: np.ndarray) -> float:
    # import ipdb; ipdb.set_trace()
    xNew = x - beta * grad
    matHat = xNew @ xNew.T
    matHat[:nDims, :nDims], matHat[nDims:, nDims:] = scales * matHat[:nDims, :nDims], scales * matHat[nDims:, nDims:]
    matHat[:nDims, nDims:], matHat[nDims:, :nDims] = scalea * matHat[:nDims, nDims:], scalea * matHat[nDims:, :nDims]
    return (1.0 / (4.0 * float(nDims))) * np.linalg.norm(A - matHat)**2

@jit(nopython = True, cache = True)
def projGradDesc(alpha: float,
                 lmbda: float,
                 nDims: int,
                 xInit: np.ndarray,
                 Ys11: np.ndarray,
                 Ys12: np.ndarray,
                 Ys21: np.ndarray,
                 Ys22: np.ndarray,
                 maxIters: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    sn = sqrt(float(nDims))
    scales = sqrt((1 - alpha) * lmbda / (2.0 * float(nDims)))
    scalea = sqrt(alpha * lmbda / (2.0 * float(nDims)))
    scales2 = scales**2
    scalea2 = scalea**2
    x = xInit # np.random.normal(size = (2 * nDims, 1))
    grad = np.empty((2 * nDims, 1))
    for k in range(maxIters):
        u, v = x[:nDims], x[nDims:]
        uNorm2 = (u.T @ u)[0, 0] # np.linalg.norm(u)**2
        vNorm2 = (v.T @ v)[0, 0] # np.linalg.norm(v)**2
        # grad[:nDims] = -2.0 * scales * (Ys11 @ u) - 2.0 * scalea * (Ys12 @ v) + 2.0 * lmbda * u
        # grad[nDims:] = -2.0 * scales * (Ys22 @ v) - 2.0 * scalea * (Ys21 @ u) + 2.0 * lmbda * v
        grad[:nDims] = -2.0 * scales * (Ys11 @ u) - 2.0 * scalea * (Ys12 @ v) + 4.0 * (scales2 * uNorm2 + scalea2 * vNorm2) * u
        grad[nDims:] = -2.0 * scales * (Ys22 @ v) - 2.0 * scalea * (Ys21 @ u) + 4.0 * (scales2 * vNorm2 + scalea2 * uNorm2) * v
        # if (np.linalg.norm(grad) < 1e-9):
            # print(f'Converged after {k} iterations')
        # beta = 1.0 / ((1.0 + float(k)) * (np.linalg.norm(grad) + 1e-9))
        # beta = 1.0 / (1.0 + float(k))
        beta = 1e-1
        # res = minimize_scalar(projGradDescObj,
        #                       bounds = [0.0, 32.0],
        #                       method = 'bounded',
        #                       args = (scales, scalea, nDims, A, x, grad),
        #                       options = {'maxiter': 16, 'disp': False})
        # beta = res.x
        x = x - beta * grad
        # uNew = (sn / np.linalg.norm(x[:nDims])) * x[:nDims]
        # vNew = (sn / np.linalg.norm(x[nDims:])) * x[nDims:]
        # x[:nDims] = uNew
        # x[nDims:] = vNew
        # print(f'New MSE: {getMixedMatrixMSE(ur, vr, uNew, vNew)}')
        # print(f'Beta: {beta} grad-norm: {np.linalg.norm(grad)}')

    return x[:nDims], x[nDims:]

def getProjGradMSE(alpha: float,
                   lmbda: float,
                   uPrior: MatrixPrior,
                   vPrior: MatrixPrior) -> float:
    nu, _ = uPrior.dimension()
    A, u, v = getMixedMatrix(lmbda, alpha, uPrior, vPrior, sym = False)
    Y11, Y12, Y21, Y22 = A[:nu, :nu], A[:nu, nu:], A[nu:, :nu], A[nu:, nu:]
    Yu = Y11 + Y11.T
    Yuv = Y12 + Y21.T
    Yvu = Y21 + Y12.T
    Yv = Y22 + Y22.T
    # xInit = np.random.normal(size = (2 * nu, 1))
    if (alpha < 0.5):
        uInit, _ = getLeadingSingularVectors(A[:nu, :nu])
        vInit, _ = getLeadingSingularVectors(A[nu:, nu:])
    else:
        uInit, vInit = getLeadingSingularVectors(A[:nu, nu:])
    xInit = np.empty((2 * nu, 1))
    xInit[:nu] = uInit
    xInit[nu:] = vInit
    uHat, vHat = projGradDesc(alpha, lmbda, nu, xInit, Yu, Yuv, Yvu, Yv)
    return 0.5 * getMixedMatrixMSE(u, v, uHat, vHat)

def mixedSpectral(Yuu: np.ndarray,
                  Yvv: np.ndarray,
                  Yuv: np.ndarray,
                  Yvu: np.ndarray,
                  u: np.ndarray,
                  v: np.ndarray,
                  nGrid: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    estimators = list()
    errors = list()
    omegas = np.linspace(0.0, 1.0, nGrid)
    for omega in omegas:
        Yu = (1.0 - omega) * Yuu + (omega * Yuv)
        nu, _ = Yu.shape
        w, uHat = eigh(Yu, subset_by_index = (nu - 1, nu - 1))
        Yv = (1.0 - omega) * Yvv + (omega * Yvu)
        nv, _ = Yv.shape
        w, vHat = eigh(Yv, subset_by_index = (nv - 1, nv - 1))
        errors.append(0.5 * getMixedMatrixMSE(u, v, uHat, vHat))
        estimators.append((uHat, vHat))

    return estimators[np.argmin(np.asarray(errors))]

def getMixedSpectralMSE(alpha: float,
                        lmbda: float,
                        uPrior: MatrixPrior,
                        vPrior: MatrixPrior) -> float:
    nu, _ = uPrior.dimension()
    A, u, v = getMixedMatrix(lmbda, alpha, uPrior, vPrior)
    Y11, Y12, Y21, Y22 = A[:nu, :nu], A[:nu, nu:], A[nu:, :nu], A[nu:, nu:]
    Yuv = Y12 @ Y12.T
    Yvu = Y12.T @ Y12
    uHat, vHat = mixedSpectral(Y11, Y22, Yuv, Yvu, u, v)
    return 0.5 * getMixedMatrixMSE(u, v, uHat, vHat)

if __name__ == '__main__':
    # nDimsU = 128
    # nDimsV = 128
    nDimsU = 1024
    nDimsV = 1024
    # nAlpha = 1024
    nAlpha = 32
    nRuns = 64
    uPrior = GaussianMatrixPrior(nDimsU, 1, 1.0)
    vPrior = GaussianMatrixPrior(nDimsV, 1, 1.0)

    # alpha = 0.01
    # lmbda = 10.0

    # lmbdas = np.linspace(0.1, 5.0, nAlpha)
    # corr2s = list()
    # mses = list()
    # for lmbda in lmbdas:
    #     print(f'Running experiment for lambda = {lmbda}')
    #     scale = sqrt(lmbda / float(nDimsU))
    #     corr2Sum = 0.0
    #     mseSum = 0.0
    #     for _ in range(nRuns):
    #         u = uPrior.sample()
    #         v = vPrior.sample()
    #         uNorm2 = (np.linalg.norm(u))**2
    #         vNorm2 = (np.linalg.norm(v))**2
    #         A = scale * (u @ v.T) + np.random.normal(size = (nDimsU, nDimsV))
    #         uHat, vHat = getLeadingSingularVectors(A)
    #         uHat = uHat / np.linalg.norm(uHat)
    #         vHat = vHat / np.linalg.norm(vHat)
    #         ucorr2 = abs((u.T @ uHat)[0, 0])**2
    #         vcorr2 = abs((v.T @ vHat)[0, 0])**2
    #         mse = (1.0 / float(nDimsU * nDimsV)) * (uNorm2 * vNorm2 - ucorr2 * vcorr2)
            # corr2Sum = corr2Sum + (corr2 / uNorm2)
            # mseSum = mseSum + mse

        # corr2s.append(corr2Sum / float(nRuns))
        # mses.append(mseSum / float(nRuns))

    # plt.scatter(lmbdas, corr2s, label = r'Emperical $Corr^2$')
    # corrs = list(map(lambda l: 0.0 if l <= 1.0 else 1.0 - (1.0 / l), lmbdas))
    # plt.plot(lmbdas, corrs, label = r'Theoretical $Corr^2$')
    # plt.legend()
    # plt.xlabel(r'$\lambda$')
    # plt.ylabel(r'$Corr^2$')
    # plt.show()

    # plt.scatter(lmbdas, mses, label = 'Emperical MSEs')
    # tmses = list(map(lambda l: 1.0 if l <= 1.0 else (1.0 / l) * (2.0 - (1.0 / l)), lmbdas))
    # plt.plot(lmbdas, tmses, label = 'MMSE')
    # plt.legend()
    # plt.xlabel(r'$\lambda$')
    # plt.ylabel('MSE')
    # plt.show()

    probs = np.array([0.5, 0.5])
    vals = np.array([-1.0, 1.0])
    # probs = np.array([0.9, 0.1])
    # vals = np.array([-(1.0 / 3.0), 3.0])
    gaussVals = np.linspace(-5.0, 5.0, 50)
    gaussProbs = (1.0 / sqrt(2.0 * np.pi)) * np.exp(-0.5 * (gaussVals)**2)
    gaussProbs = gaussProbs / gaussProbs.sum()
    uDPrior = DiscreteMatrixPrior(nDimsU, 1, probs, vals)
    vDPrior = DiscreteMatrixPrior(nDimsV, 1, probs, vals)
    # uDPrior = DiscreteMatrixPrior(nDimsU, 1, gaussProbs, gaussVals)
    # vDPrior = DiscreteMatrixPrior(nDimsV, 1, gaussProbs, gaussVals)

    # eigList = list()
    # equalList = list()
    # powList = list()
    # indList = list()
    # snrList = list()
    # spectList = list()
    # lmbdas = np.asarray([1.0, 2.0, 5.0, 10.0])
    limList = list()
    lmbdas = np.asarray([2.0])
    # alphas = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.499, 0.6, 0.7, 0.8, 0.9, 1.0])
    # alphas = np.array([0.0, 1.0])
    # alphas = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    # lmbdas = np.linspace(0.1, 5.0, nAlpha)
    algLmbdas = np.linspace(0.1, 5.0, 32)
    AMPList = list()
    PCAList = list()
    spectList = list()
    gradList = list()
    # optList = list()
    alphas = np.linspace(0.0, 1.0, nAlpha)

    for lmbda in lmbdas:
        pcaMat = np.fromfile(f'data/PCAMSE_l{lmbda}.bin')
        pcaMat = pcaMat.reshape(32, 2)
        ampMat = np.fromfile(f'data/AMPMSE_l{lmbda}.bin')
        ampMat = ampMat.reshape(32, 2)
        spectMat = np.fromfile(f'data/SpectMSE_l{lmbda}.bin')
        spectMat = spectMat.reshape(32, 2)
        # limMat = np.fromfile(f'data/FundLimitsRad_l{lmbda}.bin')
        # limMat = limMat.reshape(32, 2)
        PCAList.append(pcaMat)
        AMPList.append(ampMat)
        spectList.append(spectMat)
        # limList.append(limMat)

    for lmbda in lmbdas:
        print(f'Running experimnent for lambda = {lmbda}')
        # spectFunc = partial(getSpectralCorrEqual, lmbda = lmbda, uPrior = uPrior, vPrior = vPrior)
        # spectList.append(runFunc(spectFunc, alphas, 2, nMonteCarlo = nRuns, outFile = f'outputs/SpectOutputs_l{lmbda}.bin'))
        # snrFunc = partial(getSNRWeightedCorr, lmbda = lmbda, uPrior = uPrior, vPrior = vPrior)
        # snrList.append(runFunc(snrFunc, alphas, 2, nMonteCarlo = nRuns, outFile = f'outputs/WeightSnrOutputs_l{lmbda}.bin'))
        # indFunc = partial(getIndividualCorr, lmbda = lmbda, uPrior = uPrior, vPrior = vPrior)
        # indList.append(runFunc(indFunc, alphas, 6, nMonteCarlo = nRuns, outFile = f'outputs/IndOutputs_l{lmbda}.bin'))
        # eqFunc = partial(getSVDCorrEqual, lmbda = lmbda, uPrior = uPrior, vPrior = vPrior)
        # equalList.append(runFunc(eqFunc, alphas, 2, nMonteCarlo = nRuns, outFile = f'outputs/SVDEq_l{lmbda}.bin'))
        # PCAFunc = partial(getSVDMSE, lmbda = lmbda, uPrior = uDPrior, vPrior = vDPrior)
        # PCAList.append(runFunc(PCAFunc, alphas, 1, nMonteCarlo = nRuns, outFile = f'data/PCAMSERad_l{lmbda}.bin'))
        # AMPFunc = partial(getDiscreteMSE, lmbda = lmbda, uPrior = uDPrior, vPrior = vDPrior)
        # AMPList.append(runFunc(AMPFunc, alphas, 1, nMonteCarlo = nRuns, outFile = f'data/AMPMSERad_l{lmbda}.bin'))
        # spectFunc = partial(getMixedSpectralMSE, lmbda = lmbda, uPrior = uDPrior, vPrior = vDPrior)
        # spectList.append(runFunc(spectFunc, alphas, 1, nMonteCarlo = nRuns, outFile = f'data/SpectMSERad_l{lmbda}.bin'))
        # limFunc = partial(getDiscreteFundLimits, lmbda = lmbda, uPrior = uDPrior, vPrior = vDPrior)
        # limList.append(runFunc(limFunc, alphas, 1, outFile = f'data/FundLimitsRad_l{lmbda}.bin'))
        gradFunc = partial(getProjGradMSE, lmbda = lmbda, uPrior = uPrior, vPrior = vPrior)
        gradList.append(runFunc(gradFunc, alphas, 1, nMonteCarlo = nRuns, outFile = f'data/GradDec_l{lmbda}.bin'))
        # limFunc = partial(getNormalFundLimits, lmbda = lmbda)
        # limList.append(runFunc(limFunc, alphas, 1, outFile = f'outputs/FundLimits_l{lmbda}.bin'))
        # powFunc = partial(getPowerWeightedCorr, lmbda = lmbda, uPrior = uPrior, vPrior = vPrior)
        # powList.append(runFunc(powFunc, alphas, 2, nMonteCarlo = nRuns, outFile = f'PowOutputs_l{lmbda}.bin'))

    # ampMat = np.fromfile(f'data/AMPMSE_l{2.0}.bin')
    # ampMat = ampMat.reshape(32, 2)
    # AMPList = [ampMat]

    # spectMat = np.fromfile(f'data/SpectMSE_l{2.0}.bin')
    # spectMat = spectMat.reshape(32, 2)
    # abList = [spectMat]

    for lmbda, pcaMat, ampMat, spectMat, gradMat in zip(lmbdas, PCAList, AMPList, spectList, gradList):
        # plt.plot(limMat[:, 0], limMat[:, 1], label = 'MMSE')
        mmse = (1.0 / lmbda) * (2.0 - (1.0 / lmbda))
        plt.hlines(mmse, 0.0, 1.0, label = 'MMSE')
        plt.scatter(pcaMat[:, 0], pcaMat[:, 1], label = r'Joint PCA', s = 8, marker = "v")
        plt.scatter(ampMat[:, 0], ampMat[:, 1], label = r'AMP', s = 8, marker = "x")
        plt.scatter(spectMat[:, 0], spectMat[:, 1], label = r'Weighted PCA', s = 8, marker = '^')
        plt.scatter(gradMat[:, 0], gradMat[:, 1], label = r'Gradient Descent', s = 8, marker = 's')

    plt.xlabel(r'$\alpha$')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

    # for lmbda, ampMat, abMat in zip(lmbdas, AMPList, abList):
    #     mmse = (1.0 / lmbda) * (2.0 - (1.0 / lmbda))
    #     plt.hlines(mmse, 0.0, 1.0, label = 'MMSE')
    #     plt.scatter(ampMat[:, 0], ampMat[:, 1], c = 'b', label = r'AMP', s = 8, marker = "x")
    #     plt.scatter(abMat[:, 0], abMat[:, 1], c = 'r', label = r'Spectral Method', s = 8, marker = "x")

    # plt.xlabel(r'$\alpha$')
    # plt.ylabel('MSE')
    # plt.legend()
    # plt.show()

    # for alpha in alphas:
    #     optMat = np.fromfile(f'outputs/DAMPMSE_a{alpha}.bin', dtype = np.float64)
    #     optMat = optMat.reshape(16, 2)
    #     plt.scatter(optMat[:, 0], optMat[:, 1], label = r'AMP MSE ($\alpha$ = ' + f'{alpha})', s = 8)

    # for alpha in alphas:
    #     limMat = np.fromfile(f'outputs/FundLimitsGAD_a{alpha}.bin')
    #     limMat = limMat.reshape(1024, 2)
    #     plt.plot(limMat[:, 0], limMat[:, 1], label = r'MMSE ($\alpha$ = ' + f'{alpha})')

    # svdList = list()
    # PCAList = list()
    # AMPList = list()
    # for alpha in alphas:
        # print(f'Running experiment for alpha = {alpha}')
        # optFunc = lambda x: getSpectralMSE(alpha = alpha, lmbda = x, uPrior = uPrior, vPrior = vPrior)
        # svdList.append(runFunc(optFunc, lmbdas, 1, nMonteCarlo = nRuns, outFile = f'outputs/OptMSER_a{alpha}.bin'))
        # projFunc = lambda x: getProjGradMSE(alpha = alpha, lmbda = x, uPrior = uPrior, vPrior = vPrior)
        # svdList.append(runFunc(projFunc, lmbdas, 1, nMonteCarlo = nRuns, outFile = f'data/GradDec_a{alpha}.bin'))
        # PCAFunc = lambda x: getSVDMSE(alpha = alpha, lmbda = x, uPrior = uPrior, vPrior = vPrior)
        # PCAList.append(runFunc(PCAFunc, lmbdas, 1, nMonteCarlo = nRuns, outFile = f'data/PCAMSE_a{alpha}.bin'))
        # spectFunc = lambda x: getMixedSpectralMSE(alpha = alpha, lmbda = x, uPrior = uPrior, vPrior = vPrior)
        # svdList.append(runFunc(spectFunc, lmbdas, 1, nMonteCarlo = nRuns, outFile = f'data/Spectral_a{alpha}.bin'))
        # svdFunc = lambda x: getSpectralMSE(alpha = alpha, lmbda = x, uPrior = uPrior, vPrior = vPrior)
        # svdList.append(runFunc(svdFunc, lmbdas, 1, nMonteCarlo = nRuns, outFile = f'outputs/OptMSE_a{alpha}.bin'))
        # AMPFunc = lambda x: getGaussMSE(alpha = alpha, lmbda = x, uPrior = uPrior, vPrior = vPrior)
        # AMPList.append(runFunc(AMPFunc, lmbdas, 1, nMonteCarlo = nRuns, outFile = f'data/RAMPMSE_a{alpha}.bin'))
        # AMPFunc = lambda x: getDiscreteMSE(alpha = alpha, lmbda = x, uPrior = uDPrior, vPrior = vDPrior)
        # AMPList.append(runFunc(AMPFunc, algLmbdas, 1, nMonteCarlo = nRuns, outFile = f'data/AMPGAD_a{alpha}.bin'))
        # limFunc = lambda x: getNormalFundLimits(alpha = alpha, lmbda = x)
        # limList.append(runFunc(limFunc, lmbdas, 1, outFile = f'outputs/FundLimits_a{alpha}.bin'))
        # limList.append(runFunc(getAlphaZeroLimits, lmbdas, 1, outFile = f'outputs/FundLimits0_a{alpha}.bin'))
        # limFunc = lambda x: getDiscreteFundLimits(alpha = alpha, lmbda = x, uPrior = uDPrior, vPrior = vDPrior)
        # if (alpha == 0.0):
            # limList.append(runFunc(limFunc, lmbdas, 1, outFile = f'data/FundLimitsRad_a{alpha}.bin'))

    # for alpha, limMat in zip(alphas, limList):
        # plt.plot(limMat[:, 0], limMat[:, 1], label = r'MMSE ($\alpha$ = ' + f'{alpha})')

    # for alpha in alphas:
    #     gradMat = np.fromfile(f'data/GradDec_a{alpha}.bin')
    #     gradMat = gradMat.reshape(nAlpha, 2)
    #     plt.scatter(gradMat[:, 0], gradMat[:, 1], label = r'$\alpha$ = ' + f'{alpha}', s = 8, marker = "x")

    # for alpha, svdMat in zip(alphas, svdList):
    #     plt.scatter(svdMat[:, 0], svdMat[:, 1], label = r'$\alpha$ = ' + f'{alpha}', s = 8, marker = "x")

    # mlmbdas = np.linspace(0.0, 5.0, 4096)
    # mses = list(map(lambda l: 1.0 if l <= 1.0 else (1.0 / l) * (2.0 - (1.0 / l)), mlmbdas))
    # plt.plot(mlmbdas, mses, label = 'MMSE')

    # plt.legend()
    # plt.xlabel(r'$\lambda$')
    # plt.ylabel('MSE')
    # plt.title('Spectral MSE')
    # plt.show()

    # for alpha, AMPMat in zip(alphas, AMPList):
        # plt.scatter(AMPMat[:, 0], AMPMat[:, 1], label = r'$\alpha$ = ' + f'{alpha}', s = 8, marker = "x")

    # limMat = limList[0]
    # plt.plot(limMat[:, 0], limMat[:, 1], label = 'MMSE')

    # plt.legend()
    # plt.xlabel(r'$\lambda$')
    # plt.ylabel('MSE')
    # plt.title('AMP MSE')
    # plt.show()

    # for alpha, PCAMat in zip(alphas, PCAList):
    #     plt.scatter(PCAMat[:, 0], PCAMat[:, 1], label = r'$\alpha$ = ' + f'{alpha}', s = 8, marker = "x")

    # plt.plot(mlmbdas, mses, label = 'MMSE')

    # plt.legend()
    # plt.xlabel(r'$\lambda$')
    # plt.ylabel('MSE')
    # plt.title('PCA MSE')
    # plt.show()

    # Create and plot figures
    # fig, axes = plt.subplots(2, 2, sharex = True, sharey = True)
    # for lmbda, eqMat, optMat in zip(lmbdas, equalList, spectList):
    #     axes[0, 0].plot(eqMat[:, 0], eqMat[:, 1], label = r'$\lambda$ = ' + f'{lmbda}')
    #     axes[0, 1].plot(eqMat[:, 0], eqMat[:, 2], label = r'$\lambda$ = ' + f'{lmbda}')
    #     axes[1, 0].plot(optMat[:, 0], optMat[:, 1], label = r'$\lambda$ = ' + f'{lmbda}')
    #     axes[1, 1].plot(optMat[:, 0], optMat[:, 2], label = r'$\lambda$ = ' + f'{lmbda}')

    # axes[0, 0].set_title('Equalized SVD Vector U Correlation')
    # axes[0, 1].set_title('Equalized SVD Vector V Correlation')
    # axes[1, 0].set_title('Optimization Based U Correlation')
    # axes[1, 1].set_title('Optimization Based V Correlation')

    # fig, axes = plt.subplots(2, 3)
    # for lmbda, mat in zip(lmbdas, indList):
    #     axes[0, 0].plot(mat[:, 0], mat[:, 1], label = r'$\lambda$ = ' + f'{lmbda}')
    #     axes[0, 1].plot(mat[:, 0], mat[:, 2], label = r'$\lambda$ = ' + f'{lmbda}')
    #     axes[0, 2].plot(mat[:, 0], mat[:, 3], label = r'$\lambda$ = ' + f'{lmbda}')
    #     axes[1, 0].plot(mat[:, 0], mat[:, 4], label = r'$\lambda$ = ' + f'{lmbda}')
    #     axes[1, 1].plot(mat[:, 0], mat[:, 5], label = r'$\lambda$ = ' + f'{lmbda}')
    #     axes[1, 2].plot(mat[:, 0], mat[:, 6], label = r'$\lambda$ = ' + f'{lmbda}')

    # axes[0, 0].set_title(r'$\hat{U}_{11}$ Correlation')
    # axes[0, 1].set_title(r'$\hat{U}_{12}$ Correlation')
    # axes[0, 2].set_title(r'$\hat{U}_{21}$ Correlation')
    # axes[1, 0].set_title(r'$\hat{V}_{12}$ Correlation')
    # axes[1, 1].set_title(r'$\hat{V}_{21}$ Correlation')
    # axes[1, 2].set_title(r'$\hat{V}_{22}$ Correlation')

    # for ax in axes.flat:
    #     ax.set(xlabel = r'$\alpha$', ylabel = 'Correlation')
    #     ax.grid()
        # ax.legend()

    # for ax in axes.flat:
    #     ax.label_outer()

    # handles, labels = axes[0, 0].get_legend_handles_labels()
    # fig.legend(handles, labels)

    # plt.show()
