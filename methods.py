import numpy as np
from copy import deepcopy
from typing import Tuple, Callable

from scipy.linalg import eigh
from numba import jit
from sklearn.utils.extmath import randomized_svd

from datagen import GroupwiseMatrix

# Define some metrics to analyze the performance of various methods

def computeSingleBlockMSE(x: np.ndarray,
                          xHat: np.ndarray) -> float:
    assert x.shape == xHat.shape, "x and xHat must have the same shape"
    assert x.shape[1], "x must be an n x 1 vector"
    assert xHat.shape[1], "xHat must be an n x 1 vector"
    assert not np.any(np.isnan(x)), "x had a Nan entry"
    assert not np.any(np.isnan(xHat)), "xHat had a Nan entry"

    nDims = float(x.shape[0])
    scale = 1.0 / float(nDims**2)
    overlap2 = ((x.T @ xHat)[0, 0])**2
    xHatNorm2 = (xHat.T @ xHat)[0, 0] + 1e-12
    a = overlap2 / xHatNorm2
    xNorm4 = ((x.T @ x)[0, 0])**2
    return scale * (xNorm4 - a**2)

def computeTwoGroupMSE(x: np.ndarray,
                       xHat: np.ndarray,
                       nu: int,
                       nv: int) -> float:
    assert x.shape[0] == nu + nv, "x must have the shape of nu + nv"
    u, v = x[:nu], x[nu:]
    uHat, vHat = xHat[:nu], xHat[nu:]
    mseuu = computeSingleBlockMSE(u, uHat)
    msevv = computeSingleBlockMSE(v, vHat)
    return 0.5 * (mseuu + msevv)

# Define the various methods

def computeLeadingEigenVector(A: np.ndarray) -> np.ndarray:
    m, n = A.shape
    assert m == n, "Cannot get Eigenvectors on a non-square matrix"
    _, V = eigh(A, subset_by_index = (n - 1, n - 1))
    return V

gamma = 1e-1
maxIters = 512

def weightedPCA(groupMat: GroupwiseMatrix,
                nu: int,
                nv: int,
                nGrid: int = 32) -> np.ndarray:
    Y = groupMat.Ysym()
    x = groupMat.x()
    xHat = np.empty((nu + nv, 1))
    Yuu, Y12, Y21, Yvv = Y[:nu, :nu], Y[:nu, nu:], Y[nu:, :nu], Y[nu:, nu:]
    Yuv = Y12 @ Y12.T
    Yvu = Y12.T @ Y12
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
        xHat[:nu], xHat[nu:] = uHat, vHat
        errors.append(computeTwoGroupMSE(x, xHat, nu, nv))
        estimators.append(deepcopy(xHat))

    return estimators[np.argmin(np.asarray(errors))]

def gradDesc(groupMat: GroupwiseMatrix,
             gamma: float,
             maxIters: int,
             randomInit: bool = False) -> np.ndarray:
    Y = groupMat.Ygd()
    L = groupMat.Lgd()
    n, _ = Y.shape
    if randomInit:
        x = np.random.normal(size = (n, 1))
    else:
        Ysym = groupMat.Ysym()
        x = computeLeadingEigenVector(Ysym) + 1e-3 * np.random.normal(size = (n, 1))

    # Run the gradient descent updates
    for k in range(maxIters):
        x = x + gamma * ((Y @ x) - (L @ (x**2)) * x)

    return x

stopTol = 1e-9
maxIters = 256

@jit(nopython = True, cache = True)
def gaussEta(a: np.ndarray,
             b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = 1.0 / (1.0 + b)
    return a * v, v

@jit(nopython = True, cache = True)
def discreteEta(a: np.ndarray,
                b: np.ndarray,
                p: np.ndarray,
                x: np.ndarray,
                x2: np.ndarray,
                X: np.ndarray,
                P: np.ndarray,
                XP: np.ndarray,
                XP2: np.ndarray) -> np.ndarray:
    expon = np.exp((a @ x.T) - 0.5 * (b @ x2.T))
    num = ((XP * expon).sum(axis = 1)).reshape(-1, 1)
    den = ((P * expon).sum(axis = 1)).reshape(-1, 1)
    m = num / den
    num = ((XP2 * expon).sum(axis = 1)).reshape(-1, 1)
    v = (num / den) - (m**2)
    return m, v

def getDiscreteEtaFunc(p: np.ndarray,
                       x: np.ndarray,
                       n: int) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    pt = p.reshape(-1, 1)
    xt = x.reshape(-1, 1)
    x2 = xt**2
    X = np.tile(xt.T, (n, 1))
    P = np.tile(pt.T, (n, 1))
    XP = X * P
    XP2 = X * XP
    return lambda a,b: discreteEta(a, b, pt, xt, x2, X, P, XP, XP2)

@jit(nopython = True, cache = True)
def mixedEta(a: np.ndarray,
             b: np.ndarray,
             etaU: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
             etaV: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
             nu: int,
             nv: int) -> np.ndarray:
    m = np.empty((nu + nv, 1))
    v = np.empty((nu + nv, 1))
    m[:nu], v[:nu] = etaU(a[:nu], b[:nu])
    m[nu:], v[nu:] = etaV(a[nu:], b[nu:])
    return m

def amp(groupMat: GroupwiseMatrix,
        eta: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
        stopTol: float,
        maxIters: int,
        verbose: bool = False) -> np.ndarray:
    Y = groupMat.Yamp()
    Y2 = Y**2
    L = groupMat.Lamp()
    n, _ = Y.shape
    scale = 1.0 / (float(n))
    m = 1e-3 * np.random.normal(loc = 0.0, scale = 1.0, size = (n , 1))
    mprev = np.zeros((n, 1))
    v = np.zeros((n, 1))
    for idx in range(maxIters):
        a = Y @ m - (Y2 @ v) * mprev
        b = L @ (m**2) + (L - Y2) @ v
        mprev = deepcopy(m)
        m, v = eta(a, b)
        conv = scale * np.linalg.norm(m - mprev)
        if conv <= stopTol:
            if verbose:
                print(f'AMP converged after {idx + 1} iterations')
            break

    return m

# Define the methods that produce the MSE for the two-group model

def getJointPCATwoGroupMSE(gm: GroupwiseMatrix,
                           nu: int,
                           nv: int) -> np.ndarray:
    xHat = computeLeadingEigenVector(gm.Ysym())
    return np.array([computeTwoGroupMSE(gm.x(), xHat, nu, nv)])

def getWeightedPCATwoGroupMSE(gm: GroupwiseMatrix,
                              nu: int,
                              nv: int,
                              nGrid: int) -> np.ndarray:
    xHat = weightedPCA(gm, nu, nv, nGrid)
    return np.array([computeTwoGroupMSE(gm.x(), xHat, nu, nv)])

def getGradDescTwoGroupMSE(gm: GroupwiseMatrix,
                           nu: int,
                           nv: int,
                           gamma: float,
                           maxIters: int) -> np.ndarray:
    xHat = gradDesc(gm, gamma, maxIters)
    return np.array([computeTwoGroupMSE(gm.x(), xHat, nu, nv)])

def getGaussAMPTwoGroupMSE(gm: GroupwiseMatrix,
                           nu: int,
                           nv: int,
                           stopTol: float,
                           maxIters: int) -> np.ndarray:
    xHat = amp(gm, gaussEta, stopTol, maxIters)
    return np.array([computeTwoGroupMSE(gm.x(), xHat, nu, nv)])

def getDiscreteAMPTwoGroupMSE(gm: GroupwiseMatrix,
                              nu: int,
                              nv: int,
                              eta: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
                              stopTol: float,
                              maxIters: int) -> np.ndarray:
    xHat = amp(gm, eta, stopTol, maxIters)
    return np.array([computeTwoGroupMSE(gm.x(), xHat, nu, nv)])
