# GROME
This repository contains the code for generating figures in the paper *Fundamental limits for rank-one matrix estimation with groupwise heteroskedasticity* by Joshua K. Behne and Galen Reeves. The arxiv version of this paper can be found at [https://arxiv.org/abs/2106.11950](https://arxiv.org/abs/2106.11950).

Running this code requires the following potentially non-standard libraries: numpy, scipy, sklearn, numba, and most importantly tikplotlib.

There is a total of 4 files containing code in this project and they are summarized as:
1. datagen.py: Contains the necessary code for generating observations from the groupwise spiked matrix model introduced in the paper.
2. limits.py: Contains the algorithms for solving the variational formulas giving the asymptotic relative entropy and MMSE in the groupwise spiked matrix setting. Note this file is required to generate the fundamental limits which our estimation algorithms will be compared to.
3. methods.py: Contains the various estimation methods considered in the paper including PCA, weighted PCA, gradient descent, and AMP.
4. experiments.py: Running this as main will generate the figures from the paper and place them into folder "figures" as tikz figures (i.e., .tex files). Also gives a nice template for setting up new experiments with the groupwise spiked matrix model.
