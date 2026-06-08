import pytensor.tensor as pt

from pytensor_distributions.lmoments import _lmoments


def mean(lower, c, upper):
    return (lower + c + upper) / 3


def mode(lower, c, upper):
    shape = pt.broadcast_arrays(lower, c, upper)[0]
    return pt.full_like(shape, c)


def median(lower, c, upper):
    cond = pt.ge(c, (lower + upper) / 2)
    med1 = lower + pt.sqrt((upper - lower) * (c - lower) / 2)
    med2 = upper - pt.sqrt((upper - lower) * (upper - c) / 2)
    return pt.switch(cond, med1, med2)


def var(lower, c, upper):
    return (lower**2 + upper**2 + c**2 - lower * c - c * upper - lower * upper) / 18


def std(lower, c, upper):
    return pt.sqrt(var(lower, c, upper))


def skewness(lower, c, upper):
    num = pt.sqrt(2) * (lower + upper - 2 * c) * (2 * lower - upper - c) * (lower - 2 * upper + c)
    denom = 5 * pt.pow(lower**2 + upper**2 + c**2 - lower * c - c * upper - lower * upper, 1.5)
    return num / denom


def kurtosis(lower, c, upper):
    shape = pt.broadcast_arrays(lower, c, upper)[0]
    return pt.full_like(shape, -3 / 5)


def lmoment1(lower, c, upper):
    return mean(lower, c, upper)


def lmoment2(lower, c, upper):
    return _lmoments(ppf, lower, c, upper, r=2)


def lmoment3(lower, c, upper):
    return _lmoments(ppf, lower, c, upper, r=3)


def lmoment4(lower, c, upper):
    return _lmoments(ppf, lower, c, upper, r=4)


def entropy(lower, c, upper):
    lower, c, upper = pt.broadcast_arrays(lower, c, upper)
    return 0.5 + pt.log((upper - lower) / 2)


def pdf(x, lower, c, upper):
    res = pt.switch(
        pt.lt(x, c),
        2 * (x - lower) / ((upper - lower) * (c - lower)),
        2 * (upper - x) / ((upper - lower) * (upper - c)),
    )
    return pt.switch(pt.bitwise_and(pt.le(lower, x), pt.le(x, upper)), res, 0.0)


def cdf(x, lower, c, upper):
    x = pt.as_tensor_variable(x)
    return pt.switch(
        pt.le(x, lower),
        0.0,
        pt.switch(
            pt.le(x, c),
            ((x - lower) ** 2) / ((upper - lower) * (c - lower)),
            pt.switch(
                pt.lt(x, upper),
                1 - ((upper - x) ** 2) / ((upper - lower) * (upper - c)),
                1.0,
            ),
        ),
    )


def ppf(q, lower, c, upper):
    return pt.switch(
        pt.lt(q, ((c - lower) / (upper - lower))),
        lower + pt.sqrt((upper - lower) * (c - lower) * q),
        upper - pt.sqrt((upper - lower) * (upper - c) * (1 - q)),
    )


def sf(x, lower, c, upper):
    return pt.exp(logsf(x, lower, c, upper))


def isf(q, lower, c, upper):
    return ppf(1 - q, lower, c, upper)


def rvs(lower, c, upper, size=None, random_state=None):
    u = pt.random.uniform(0.0, 1.0, size=size, rng=random_state, return_next_rng=True)[1]
    return ppf(u, lower, c, upper)


def logpdf(x, lower, c, upper):
    res = pt.switch(
        pt.lt(x, c),
        pt.log(2 * (x - lower) / ((upper - lower) * (c - lower))),
        pt.log(2 * (upper - x) / ((upper - lower) * (upper - c))),
    )
    return pt.switch(pt.bitwise_and(pt.le(lower, x), pt.le(x, upper)), res, -pt.inf)


def logcdf(x, lower, c, upper):
    return pt.switch(
        pt.le(x, lower),
        -pt.inf,
        pt.switch(
            pt.le(x, c),
            pt.log(((x - lower) ** 2) / ((upper - lower) * (c - lower))),
            pt.switch(
                pt.lt(x, upper),
                pt.log1p(-((upper - x) ** 2) / ((upper - lower) * (upper - c))),
                0,
            ),
        ),
    )


def logsf(x, lower, c, upper):
    return pt.switch(
        pt.ge(x, upper),
        -pt.inf,
        pt.switch(
            pt.ge(x, c),
            pt.log(((upper - x) ** 2) / ((upper - lower) * (upper - c))),
            pt.switch(
                pt.gt(x, lower),
                pt.log1p(-((x - lower) ** 2) / ((upper - lower) * (c - lower))),
                0,
            ),
        ),
    )
