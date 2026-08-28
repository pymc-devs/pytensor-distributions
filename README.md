# PyTensor-distributions

PyTensor-powered distributions

## Overview

PyTensor-distributions provides a collection of core probability expressions implemented in PyTensor. 

The goal of this package is to work as a unified, well-tested, and performant source for core probability expressions. This reduces redundancy across packages and allows other libraries to focus on providing specialized features and APIs on top of these core expressions.

The core expressions implemented in PyTensor-distributions are:

- `pdf`: the probability density function of a distribution. (We use `pdf` for both discrete and continuous distributions)
- `cdf`: the cumulative distribution function of a distribution.
- `ppf`: the percent point function (inverse of cdf) of a distribution.
- `sf`: the survival function (1 - cdf) of a distribution.
- `isf`: the inverse survival function (inverse of sf) of a distribution.
- `logpdf`: the log-probability of a distribution. (We use `pdf` for both discrete and continuous distributions)
- `logcdf`: the log-cumulative distribution function of a distribution.
- `logsf`: the log-survival function (1 - cdf) of a distribution
- `mean`: the mean of a distribution.
- `mode`: the mode of a distribution.
- `median`: the median of a distribution.
- `var`: the variance of a distribution.
- `std`: the standard deviation of a distribution.
- `skewness`: the skewness of a distribution.
- `kurtosis`: the (excess) kurtosis of a distribution.
- `lmoments`: The first L-moment is the mean, so we omit it. For the third and fourth L-moments, we use the ratios.
- `entropy`: the entropy of a distribution.
- `rvs`: the random variates of a distribution.

Some distributions may not implement all of these expressions.

The package follows a very minimal design, with one file per distribution and some utility files for common functions and tests. The core expressions are implemented as a collection of functions; no classes are used. This is on purpose, to keep the package as simple as possible and let other packages implement more structured APIs on top of these core expressions.

PyTensor-distributions is still in early development, and we are still adding more distributions and tests and working on improving robustness.


## Contributions

PyTensor-distributions is a community project and welcomes contributions.


## Code of Conduct

We follow the PyMC [Code of Conduct](hhttps://github.com/pymc-devs/pymc/blob/main/CODE_OF_CONDUCT.md)

## Donations

PyTensor-distributions, like other PyMC projects, is a non-profit project under the NumFOCUS umbrella. If you want to financially support PyTensor-distributions or other PyMC projects, you can donate [here](https://numfocus.org/donate-to-pymc).

## Sponsors
[![NumFOCUS](https://www.numfocus.org/wp-content/uploads/2017/07/NumFocus_LRG.png)](https://numfocus.org)
