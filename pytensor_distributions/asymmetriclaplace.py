import pytensor.tensor as pt

from pytensor_distributions.helper import ppf_bounds_cont


def mean(mu, b, kappa):
    return (1 / kappa - kappa) * b + mu


def mode(mu, b, kappa):
    shape = pt.broadcast_arrays(mu, b, kappa)[0]
    return pt.full_like(shape, mu)


def median(mu, b, kappa):
    return pt.switch(
        pt.gt(kappa, 1),
        mu + kappa * b * pt.log((1 + kappa**2) / (2 * kappa**2)),
        mu - pt.log((1 + kappa**2) / 2) / (kappa / b),
    )


def var(mu, b, kappa):
    return ((1 / kappa) ** 2 + kappa**2) * b**2


def std(mu, b, kappa):
    return pt.sqrt(var(mu, b, kappa))


def skewness(mu, b, kappa):
    return 2.0 * (1 - pt.power(kappa, 6)) / pt.power(1 + pt.power(kappa, 4), 1.5)


def kurtosis(mu, b, kappa):
    return 6.0 * (1 + pt.power(kappa, 8)) / pt.power(1 + pt.power(kappa, 4), 2)


def lmoment1(mu, b, kappa):
    return mean(mu, b, kappa)


def lmoment2(mu, b, kappa):
    shape = pt.broadcast_arrays(mu, b, kappa)[0]
    kappa_sq = kappa**2
    num = b * (1.0 + kappa_sq + kappa_sq**2)
    denom = 2.0 * kappa * (1.0 + kappa_sq)
    return pt.full_like(shape, num / denom)


def lmoment3(mu, b, kappa):
    shape = pt.broadcast_arrays(mu, b, kappa)[0]
    kappa_sq = kappa**2
    kappa_four = kappa_sq**2
    kappa_six = kappa_sq * kappa_four
    num = 1.0 + 2.0 * kappa_sq - 2.0 * kappa_four - kappa_six
    denom = 3.0 * (1.0 + 2.0 * kappa_sq + 2.0 * kappa_four + kappa_six)
    return pt.full_like(shape, num / denom)


def lmoment4(mu, b, kappa):
    shape = pt.broadcast_arrays(mu, b, kappa)[0]
    kappa_sq = kappa**2
    kappa_four = kappa_sq**2
    kappa_six = kappa_sq * kappa_four
    kappa_eight = kappa_four**2
    num = 1.0 + 3.0 * kappa_sq + 9.0 * kappa_four + 3.0 * kappa_six + kappa_eight
    denom = 6.0 * (1.0 + 3.0 * kappa_sq + 4.0 * kappa_four + 3.0 * kappa_six + kappa_eight)
    return pt.full_like(shape, num / denom)


def entropy(mu, b, kappa):
    return 1 + pt.log(kappa + 1 / kappa) + pt.log(b)


def cdf(x, mu, b, kappa):
    x_norm = (x - mu) / b
    kap_inv = 1 / kappa
    kap_kapinv = kappa + kap_inv

    return pt.switch(
        pt.ge(x_norm, 0),
        1 - pt.exp(-x_norm * kappa) * (kap_inv / kap_kapinv),
        pt.exp(x_norm * kap_inv) * (kappa / kap_kapinv),
    )


def ppf(q, mu, b, kappa):
    kap_inv = 1 / kappa
    kap_kapinv = kappa + kap_inv

    q_ppf = pt.switch(
        pt.ge(q, kappa / kap_kapinv),
        -pt.log((1 - q) * kap_kapinv * kappa) * kap_inv,
        pt.log(q * kap_kapinv / kappa) * kappa,
    )

    return ppf_bounds_cont(q_ppf * b + mu, q, -pt.inf, pt.inf)


def pdf(x, mu, b, kappa):
    return pt.exp(logpdf(x, mu, b, kappa))


def sf(x, mu, b, kappa):
    return 1 - cdf(x, mu, b, kappa)


def isf(x, mu, b, kappa):
    return ppf(1 - x, mu, b, kappa)


def rvs(mu, b, kappa, size=None, random_state=None):
    random_samples = pt.random.uniform(
        -kappa, 1 / kappa, size=size, rng=random_state, return_next_rng=True
    )[1]
    sgn = pt.sign(random_samples)
    return mu - (1 / (1 / b * sgn * pt.power(kappa, sgn))) * pt.log(
        1 - random_samples * sgn * pt.power(kappa, sgn)
    )


def logpdf(x, mu, b, kappa):
    x_norm = (x - mu) / b
    kap_inv = 1 / kappa

    ald_x = pt.switch(pt.ge(x_norm, 0), x_norm * -kappa, x_norm * kap_inv)

    ald_x = ald_x - pt.log(kappa + kap_inv)
    return ald_x - pt.log(b)


def logcdf(x, mu, b, kappa):
    return pt.log(cdf(x, mu, b, kappa))


def logsf(x, mu, b, kappa):
    return pt.log(sf(x, mu, b, kappa))


def from_q(q):
    kappa = (q / (1 - q)) ** 0.5
    return kappa


def to_q(kappa):
    q = kappa**2 / (1 + kappa**2)
    return q
