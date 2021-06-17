import numpy as np
from math import sqrt, log
from typing import Tuple

from numba import jit
from scipy.integrate import quad
from scipy.optimize import root

from datagen import DiscretePrior

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
    q0 = np.array([0.25, 0.25, 0.25, 0.25])
    res = root(normalFundLimitsGrad, q0, jac = normalFundLimitsJac, args = (l11, l12, l22))
    q = res.x
    assert (q[0] >= 0.0 or abs(q[0]) < 1e-7) and (q[1] >= 0.0 or abs(q[1]) < 1e-7), f"Qt was infeasable -> q1t = {q[0]} and q2t = {q[1]}"
    q = np.clip(q, 0.0, 0.5)
    q1, q2 = q[2], q[3]
    mse = 1.0 - q1**2 - q2**2 - 2.0 * q1 * q2
    print(f'MSE value was: {mse} \nq1: {q1} q2: {q2}')
    return [mse]

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
                          uPrior: DiscretePrior,
                          vPrior: DiscretePrior) -> Tuple[float]:
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
