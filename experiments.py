import numpy as np
from math import sqrt, log
from typing import List, Tuple, Callable, Optional, Any, Dict
from functools import partial

from datagen import Prior, GaussianPrior, DiscretePrior, GroupwiseMatrix, generateTwoGroupMatrix
from limits import getDiscreteFundLimits

def getMeanSDStandard(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.mean(X[:, 1:], axis = 1), np.std(X[:, 1:], axis = 1)

def runTwoGroupCompFineAlpha(funcs: List[Callable[[GroupwiseMatrix], np.ndarray]],
                             alphas: np.ndarray,
                             uPrior: Prior,
                             vPrior: Prior,
                             lmbda: float,
                             nMonteCarlo: int = 1,
                             outFiles: Optional[str] = None,
                             verbose: bool = True) -> List[np.ndarray]:
    assert outFiles is None or len(funcs) == len(outFiles), "A separate file must be specified for each function"
    nComp = len(funcs)
    nAlpha = alphas.shape[0]
    nu = uPrior.dimension()
    nv = vPrior.dimension()
    outputs = list()
    for _ in range(nComp):
        outputs.append(None)

    for aidx, alpha in zip(range(nAlpha), alphas):
        if verbose:
            print(f'Running comparisons for alpha = {alpha}')

        for midx in range(nMonteCarlo):
            # Generate the data
            groupMat = generateTwoGroupMatrix(alpha, lmbda, uPrior, vPrior)
            for idx, output, func in zip(range(nComp), outputs, funcs):
                result = func(groupMat)

                nOut = result.shape[0]
                if (output is None):
                    outputs[idx] = np.empty((nAlpha, nOut * nMonteCarlo))
                    output = outputs[idx]

                if midx == 0:
                    output[aidx, 0] = alpha

                output[aidx, 1 + midx * nOut : 1 + (midx + 1) * nOut] = result

    if outFiles is not None:
        for output, outFile in zip(outputs, outFiles):
            output.tofile(outFile)

    return outputs

def runTwoGroupCompFineLmbda(funcs: List[Callable[[GroupwiseMatrix], np.ndarray]],
                             lmbdas: np.ndarray,
                             uPrior: Prior,
                             vPrior: Prior,
                             alpha: float,
                             nMonteCarlo: int = 1,
                             outFiles: Optional[str] = None,
                             verbose: bool = True) -> List[np.ndarray]:
    assert outFiles is None or len(funcs) == len(outFiles), "A separate file must be specified for each function"
    nComp = len(funcs)
    nLmbda = lmbdas.shape[0]
    nu = uPrior.dimension()
    nv = vPrior.dimension()
    outputs = list()
    for _ in range(nComp):
        outputs.append(None)

    for lidx, lmbda in zip(range(nLmbda), lmbdas):
        if verbose:
            print(f'Running comparisons for lambda = {lmbda}')

        for midx in range(nMonteCarlo):
            # Generate the data
            groupMat = generateTwoGroupMatrix(alpha, lmbda, uPrior, vPrior)
            for idx, output, func in zip(range(nComp), outputs, funcs):
                result = func(groupMat)

                nOut = result.shape[0]
                if (output is None):
                    outputs[idx] = np.empty((nLmbda, nOut * nMonteCarlo))
                    output = outputs[idx]

                if midx == 0:
                    output[lidx, 0] = lmbda

                output[lidx, 1 + midx * nOut : 1 + (midx + 1) * nOut] = result

    if outFiles is not None:
        for output, outFile in zip(outputs, outFiles):
            output.tofile(outFile)

    return outputs

if __name__ == '__main__':
    import sys
    import os
    from matplotlib import pyplot as plt
    import tikzplotlib as tikzplt

    import methods as mt

    # Define the setup to reproduce figure 1
    nDimsU = 1024
    nDimsV = 1024
    nAlpha = 32
    nMonteCarlo = 64
    nWeightedPCAGrid = 32
    maxItersGD = 512
    maxItersAMP = 256
    gammaGD = 1e-1
    tolAMP = 1e-9

    lmbda = 2.0
    alphas = np.linspace(0.0, 1.0, nAlpha)

    gaussPriorU = GaussianPrior(nDimsU)
    gaussPriorV = GaussianPrior(nDimsV)

    radProbs = np.array([0.5, 0.5])
    radVals = np.array([-1.0, 1.0])
    radPriorU = DiscretePrior(nDimsU, radProbs, radVals)
    radPriorV = DiscretePrior(nDimsV, radProbs, radVals)
    radEta = mt.getDiscreteEtaFunc(radProbs, radVals, nDimsU + nDimsV)

    Define the functions to be compared
    jointPCAFunc = lambda gm: mt.getJointPCATwoGroupMSE(gm, nDimsU, nDimsV)
    weightedPCAFunc = lambda gm: mt.getWeightedPCATwoGroupMSE(gm, nDimsU, nDimsV, nWeightedPCAGrid)
    gradDescFunc = lambda gm: mt.getGradDescTwoGroupMSE(gm, nDimsU, nDimsV, gammaGD, maxItersGD)
    gaussAMPFunc = lambda gm: mt.getGaussAMPTwoGroupMSE(gm, nDimsU, nDimsV, tolAMP, maxItersAMP)
    radAMPFunc = lambda gm: mt.getDiscreteAMPTwoGroupMSE(gm, nDimsU, nDimsV, radEta, tolAMP, maxItersAMP)

    gaussFuncList = [jointPCAFunc, weightedPCAFunc, gradDescFunc, gaussAMPFunc]
    radFuncList = [jointPCAFunc, weightedPCAFunc, gradDescFunc, radAMPFunc]

    Now generate the data to create the plots
    outFileRoot = "data"
    labels = ["Joint PCA", "Weighted PCA", "Gradient Descent", "AMP"]
    gaussFileNames = list()
    radFileNames = list()
    for label in labels:
        gaussFileNames.append(os.path.join(outFileRoot, "Gauss " + label + " l2.0.bin"))
        radFileNames.append(os.path.join(outFileRoot, "Rad " + label + " l2.0.bin"))

    gaussResults = runTwoGroupCompFineAlpha(gaussFuncList,
                                            alphas,
                                            gaussPriorU,
                                            gaussPriorV,
                                            lmbda,
                                            nMonteCarlo,
                                            gaussFileNames)

    radResults = runTwoGroupCompFineAlpha(radFuncList,
                                          alphas,
                                          radPriorU,
                                          radPriorV,
                                          lmbda,
                                          nMonteCarlo,
                                          radFileNames)

    Create the plots themselves
    colors = ["b", "g", "r", "m"]
    markers = ["v", "^", "s", "x"]

    mmse = (1.0 / lmbda) * (2.0 - (1.0 / lmbda))
    plt.hlines(mmse, 0.0, 1.0, label = 'MMSE')
    for gdata, label, color, marker in zip(gaussResults, labels, colors, markers):
        means, sds = getMeanSDStandard(gdata)
        plt.scatter(gdata[:, 0], means, s = 8, c = color, marker = marker, label = label)
        plt.errorbar(gdata[:, 0], means, yerr = sds, ecolor = color)

    plt.xlabel(r'$\alpha$')
    plt.ylabel('Two Group MSE')
    plt.legend()
    tikzplt.save("figures/fig1a.tex")
    plt.close()

    testAlpha = 0.0 # Any alpha will produce the same result
    mmse = getDiscreteFundLimits(testAlpha, lmbda, radPriorU, radPriorV)
    plt.hlines(mmse, 0.0, 1.0, label = 'MMSE')
    for rdata, label, color, marker in zip(radResults, labels, colors, markers):
        means, sds = getMeanSDStandard(rdata)
        plt.scatter(rdata[:, 0], means, s = 8, c = color, marker = marker, label = label)
        plt.errorbar(rdata[:, 0], means, yerr = sds, ecolor = color)

    plt.xlabel(r'$\alpha$')
    plt.ylabel('Two Group MSE')
    plt.legend()
    tikzplt.save("figures/fig1b.tex")
    plt.close()

    # Define the setup to reproduce figure 2
    nDimsU = 128
    nDimsV = 128
    nLmbda = 32
    nMonteCarlo = 32
    nWeightedPCAGrid = 32
    maxItersGD = 512
    maxItersAMP = 256
    gammaGD = 1e-1
    tolAMP = 1e-9

    alphas = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    lmbdas = np.linspace(0.1, 5.0, nLmbda)

    gaussPriorU = GaussianPrior(nDimsU)
    gaussPriorV = GaussianPrior(nDimsV)

    # Define the functions to be compared
    jointPCAFunc = lambda gm: mt.getJointPCATwoGroupMSE(gm, nDimsU, nDimsV)
    weightedPCAFunc = lambda gm: mt.getWeightedPCATwoGroupMSE(gm, nDimsU, nDimsV, nWeightedPCAGrid)
    gradDescFunc = lambda gm: mt.getGradDescTwoGroupMSE(gm, nDimsU, nDimsV, gammaGD, maxItersGD)
    gaussAMPFunc = lambda gm: mt.getGaussAMPTwoGroupMSE(gm, nDimsU, nDimsV, tolAMP, maxItersAMP)

    gaussFuncList = [jointPCAFunc, weightedPCAFunc, gradDescFunc, gaussAMPFunc]

    # Generate the data to create the plots
    outFileRoot = "data"
    labels = ["Joint PCA", "Weighted PCA", "Gradient Descent", "AMP"]
    jointPCAList = list()
    weightPCAList = list()
    gdList = list()
    ampList = list()
    for alpha in alphas:
        fileNames = list()
        for label in labels:
            fileNames.append(os.path.join(outFileRoot, "Gauss " + label + f' a{alpha}.bin'))

        jointRes, weightRes, gdRes, ampRes = runTwoGroupCompFineLmbda(gaussFuncList,
                                                                      lmbdas,
                                                                      gaussPriorU,
                                                                      gaussPriorV,
                                                                      alpha,
                                                                      nMonteCarlo,
                                                                      fileNames)
        jointPCAList.append(jointRes)
        weightPCAList.append(weightRes)
        gdList.append(gdRes)
        ampList.append(ampRes)

    # Generate the plots
    mlmbdas = np.linspace(0.0, 5.0, 4096)
    mses = list(map(lambda l: 1.0 if l <= 1.0 else (1.0 / l) * (2.0 - (1.0 / l)), mlmbdas))

    plt.plot(mlmbdas, mses, label = 'MMSE')
    for alpha, jointMat in zip(alphas, jointPCAList):
        means, _ = getMeanSDStandard(jointMat)
        plt.scatter(jointMat[:, 0], means, marker = "x", s = 8, label = r'$\alpha = ' + f'{alpha}')

    plt.xlabel(r'$\lambda$')
    plt.ylabel('Two Group MSE')
    plt.legend()
    tikzplt.save("figures/fig2a.tex")
    # plt.show()
    plt.close()

    plt.plot(mlmbdas, mses, label = 'MMSE')
    for alpha, weightMat in zip(alphas, weightPCAList):
        means, _ = getMeanSDStandard(weightMat)
        plt.scatter(weightMat[:, 0], means, marker = "x", s = 8, label = r'$\alpha = ' + f'{alpha}')

    plt.xlabel(r'$\lambda$')
    plt.ylabel('Two Group MSE')
    plt.legend()
    tikzplt.save("figures/fig2b.tex")
    # plt.show()
    plt.close()

    plt.plot(mlmbdas, mses, label = 'MMSE')
    for alpha, gdMat in zip(alphas, gdList):
        means, _ = getMeanSDStandard(gdMat)
        plt.scatter(gdMat[:, 0], means, marker = "x", s = 8, label = r'$\alpha = ' + f'{alpha}')

    plt.xlabel(r'$\lambda$')
    plt.ylabel('Two Group MSE')
    plt.legend()
    tikzplt.save("figures/fig2c.tex")
    # plt.show()
    plt.close()

    plt.plot(mlmbdas, mses, label = 'MMSE')
    for alpha, ampMat in zip(alphas, ampList):
        means, _ = getMeanSDStandard(ampMat)
        plt.scatter(ampMat[:, 0], means, marker = "x", s = 8, label = r'$\alpha = ' + f'{alpha}')

    plt.xlabel(r'$\lambda$')
    plt.ylabel('Two Group MSE')
    plt.legend()
    tikzplt.save("figures/fig2d.tex")
    # plt.show()
    plt.close()
