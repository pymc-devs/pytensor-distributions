"""Common utilities for testing distributions against scipy implementations."""

import numpy as np
import pytensor.tensor as pt
from numpy.testing import assert_allclose
from scipy.stats import lmoment


def broadcast_x(x, bcast_shape):
    """Reshape x so it broadcasts against parameters with bcast_shape."""
    if bcast_shape:
        return x.reshape((-1, *np.ones(len(bcast_shape), dtype=int)))
    return x


def run_distribution_tests(
    p_dist,
    sp_dist,
    p_params,
    sp_params,
    support,
    is_discrete=False,
    name=None,
    entropy_rtol=1e-4,
    pdf_rtol=1e-6,
    logpdf_rtol=1e-6,
    random_rtol=1e-6,
    cdf_rtol=1e-6,
    sf_rtol=1e-6,
    logcdf_rtol=1e-6,
    logsf_rtol=1e-6,
    isf_rtol=1e-6,
    median_rtol=1e-6,
    mean_rtol=1e-6,
    std_rtol=1e-6,
    var_rtol=1e-6,
    skewness_rtol=1e-6,
    kurtosis_rtol=1e-6,
    skip_mode=False,
    skip_isf=False,
    skip_standard_deviation=False,
    skip_variance=False,
    skip_skewness=False,
    skip_kurtosis=False,
    use_quantiles_for_rvs=False,
    quantiles_sample_size=25_000,
):
    scipy_dist = sp_dist(**sp_params)

    p_param_vals = [param.eval() if hasattr(param, "eval") else param for param in p_params]
    param_info = f"\nPyTensor params: {p_param_vals}\nSciPy params: {sp_params}"

    # rvs size must include the broadcast shape of the parameters for batched params
    bcast_shape = np.broadcast_shapes(*(np.shape(v) for v in p_param_vals))

    # Entropy
    actual = p_dist.entropy(*p_params).eval()
    expected = scipy_dist.entropy()
    assert_allclose(
        actual, expected, rtol=entropy_rtol, err_msg=f"Entropy test failed with {param_info}"
    )

    # Random variates
    size = (20, *bcast_shape)
    rng_p = pt.random.default_rng(1)
    actual_rvs = p_dist.rvs(*p_params, size=size, random_state=rng_p).eval()
    rng_n = np.random.default_rng(1)
    expected_rvs = scipy_dist.rvs(size, random_state=rng_n)

    if use_quantiles_for_rvs:
        p_rvs = p_dist.rvs(
            *p_params, size=(quantiles_sample_size, *bcast_shape), random_state=rng_p
        ).eval()
        s_rvs = scipy_dist.rvs((quantiles_sample_size, *bcast_shape), random_state=rng_n)
        assert_allclose(
            np.quantile(p_rvs, [0.25, 0.5, 0.75], axis=0),
            np.quantile(s_rvs, [0.25, 0.5, 0.75], axis=0),
            rtol=1e-1,
            atol=0.05,  # Allow small absolute tolerance for values near zero
            err_msg=f"Random variates (quantiles) test failed with {param_info}",
        )
    else:
        assert_allclose(
            actual_rvs,
            expected_rvs,
            rtol=random_rtol,
            err_msg=f"Random variates test failed with {param_info}",
        )

    extended_vals = np.concatenate(
        [
            actual_rvs.ravel(),
            support,
            [support[0] - 1],
            [support[0] - 2],
            [support[1] + 1],
            [support[1] + 2],
        ]
    )

    # Evaluate removing this once scipy > 1.19 get released
    if name == "exgaussian":
        extended_vals = extended_vals[np.isfinite(extended_vals)]

    extended_vals = broadcast_x(extended_vals, bcast_shape)

    # PDF
    actual_pdf = p_dist.pdf(extended_vals, *p_params).eval()
    try:
        expected_pdf = scipy_dist.pdf(extended_vals)
    except AttributeError:
        expected_pdf = scipy_dist.pmf(extended_vals)

    assert_allclose(
        actual_pdf,
        expected_pdf,
        rtol=pdf_rtol,
        atol=1e-308,
        err_msg=f"PDF test failed with {param_info}",
    )

    # logPDF
    actual_logpdf = p_dist.logpdf(extended_vals, *p_params).eval()
    try:
        expected_logpdf = scipy_dist.logpdf(extended_vals)
    except AttributeError:
        expected_logpdf = scipy_dist.logpmf(extended_vals)

    assert_allclose(
        actual_logpdf,
        expected_logpdf,
        rtol=logpdf_rtol,
        err_msg=f"logPDF test failed with {param_info}",
    )

    # CDF
    actual_cdf = p_dist.cdf(extended_vals, *p_params).eval()
    expected_cdf = scipy_dist.cdf(extended_vals)
    assert_allclose(
        actual_cdf,
        expected_cdf,
        rtol=cdf_rtol,
        atol=1e-12,
        err_msg=f"CDF test failed with {param_info}",
    )

    # logCDF
    actual_logcdf = p_dist.logcdf(extended_vals, *p_params).eval()
    expected_logcdf = scipy_dist.logcdf(extended_vals)
    assert_allclose(
        actual_logcdf,
        expected_logcdf,
        rtol=logcdf_rtol,
        atol=1e-12,
        err_msg=f"logCDF test failed with {param_info}",
    )

    # SF
    actual_sf = p_dist.sf(extended_vals, *p_params).eval()
    expected_sf = scipy_dist.sf(extended_vals)
    assert_allclose(
        actual_sf,
        expected_sf,
        rtol=sf_rtol,
        atol=1e-12,
        err_msg=f"SF test failed with {param_info}",
    )

    # logSF
    actual_logsf = p_dist.logsf(extended_vals, *p_params).eval()
    expected_logsf = scipy_dist.logsf(extended_vals)
    resolvable = (expected_sf > 1e-10) & np.isfinite(expected_logsf) & np.isfinite(actual_logsf)
    assert_allclose(
        actual_logsf[resolvable],
        expected_logsf[resolvable],
        rtol=logsf_rtol,
        atol=1e-12,
        err_msg=f"logSF test failed with {param_info}",
    )
    ok_deep = (
        (np.isnan(actual_logsf) & np.isnan(expected_logsf))
        | (actual_logsf == -np.inf)
        | (np.isfinite(actual_logsf) & (actual_logsf <= -22.0))
    )
    assert np.all(ok_deep[~resolvable]), f"logSF deep-tail test failed with {param_info}"

    # ISF
    if not skip_isf:
        x_vals = broadcast_x(np.array([-1, 0, 0.25, 0.5, 0.75, 1, 2]), bcast_shape)
        actual_isf = p_dist.isf(x_vals, *p_params).eval()
        expected_isf = scipy_dist.isf(x_vals)
        assert_allclose(
            actual_isf,
            expected_isf,
            rtol=isf_rtol,
            atol=1e-15,
            err_msg=f"ISF test failed with {param_info}",
        )

    # Mean
    mean = p_dist.mean(*p_params).eval()
    expected_mean = scipy_dist.mean()
    assert_allclose(
        mean, expected_mean, rtol=mean_rtol, err_msg=f"Mean test failed with {param_info}"
    )

    # Median
    median = p_dist.median(*p_params).eval()
    expected_median = scipy_dist.median()
    assert_allclose(
        median,
        expected_median,
        rtol=median_rtol,
        atol=1e-15,
        err_msg=f"Median test failed with {param_info}",
    )

    # Mode
    if not skip_mode:
        mode_val = p_dist.mode(*p_params).eval()
        if is_discrete:
            eps = 1
        else:
            q_vals = broadcast_x(np.array([0.6, 0.4]), bcast_shape)
            eps = np.diff(p_dist.ppf(q_vals, *p_params).eval(), axis=0) * 0.01

        pdf_mode = p_dist.pdf(mode_val, *p_params).eval()
        pdf_left = p_dist.pdf(mode_val - eps, *p_params).eval()
        pdf_right = p_dist.pdf(mode_val + eps, *p_params).eval()

        assert np.all(pdf_mode >= pdf_left - 1e-4), (
            f"Mode test (left) failed with {param_info}: pdf_mode={pdf_mode}, pdf_left={pdf_left}"
        )
        assert np.all(pdf_mode >= pdf_right - 1e-4), (
            f"Mode test (right) failed with {param_info}: "
            f"pdf_mode={pdf_mode}, pdf_right={pdf_right}"
        )

    # Standard deviation
    if not skip_standard_deviation:
        std = p_dist.std(*p_params).eval()
        expected_std = scipy_dist.std()
        assert_allclose(
            std,
            expected_std,
            rtol=std_rtol,
            err_msg=f"Standard deviation test failed with {param_info}",
        )

    # Variance
    if not skip_variance:
        var = p_dist.var(*p_params).eval()
        expected_var = scipy_dist.var()
        assert_allclose(
            var, expected_var, rtol=var_rtol, err_msg=f"Variance test failed with {param_info}"
        )

    # Skewness
    if not skip_skewness:
        skewness = p_dist.skewness(*p_params).eval()
        expected_skewness = scipy_dist.stats(moments="s")
        assert_allclose(
            skewness,
            expected_skewness,
            rtol=skewness_rtol,
            err_msg=f"Skewness test failed with {param_info}",
        )

    # Kurtosis
    if not skip_kurtosis:
        kurtosis = p_dist.kurtosis(*p_params).eval()
        expected_kurtosis = scipy_dist.stats(moments="k")
        assert_allclose(
            kurtosis,
            expected_kurtosis,
            rtol=kurtosis_rtol,
            err_msg=f"Kurtosis test failed with {param_info}",
        )

    # L-moments
    run_lmoments_test(p_dist=p_dist, p_params=p_params, name=name)


def make_params(*values, dtype=None):
    """Create PyTensor constant parameters."""
    if dtype:
        return tuple(pt.constant(v, dtype=dtype) for v in values)
    return tuple(pt.constant(v) for v in values)


def run_lmoments_test(p_dist, p_params, name=None, rtol=5e-2, atol=5e-2, sample_size=50_000):
    """Compare the distribution's lmoment2-4 against sample L-moments."""
    if name in ["cauchy"]:
        assert np.all(np.isnan(p_dist.lmoment2(*p_params).eval()))
        assert np.all(np.isnan(p_dist.lmoment3(*p_params).eval()))
        assert np.all(np.isnan(p_dist.lmoment4(*p_params).eval()))
        return

    if name in ["halfcauchy"]:
        assert np.all(p_dist.lmoment2(*p_params).eval() == float("inf"))
        assert np.all(p_dist.lmoment3(*p_params).eval() == float("inf"))
        assert np.all(p_dist.lmoment4(*p_params).eval() == float("inf"))
        return

    else:
        bcast_shape = np.broadcast_shapes(
            *(np.shape(p.eval() if hasattr(p, "eval") else p) for p in p_params)
        )
        rng = pt.random.default_rng(42)
        data = p_dist.rvs(*p_params, size=(sample_size, *bcast_shape), random_state=rng).eval()

        s_l2, s_tau3, s_tau4 = lmoment(data, order=[2, 3, 4])
        s_l2, s_tau3, s_tau4 = np.atleast_1d(s_l2), np.atleast_1d(s_tau3), np.atleast_1d(s_tau4)

        if name in ["halfstudentt", "frechet"]:
            param = np.atleast_1d(p_params[0].eval())
            s_l2[param <= 1] = np.inf
            s_tau3[param <= 2] = np.inf
            s_tau4[param <= 3] = np.inf

        if name in ["inversegamma", "pareto"]:
            param = np.atleast_1d(p_params[0].eval())
            s_l2[param <= 1] = np.inf
            s_tau3[param <= 1] = np.inf
            s_tau4[param <= 1] = np.inf

        if name in ["loglogistic", "betaprime"]:
            param = np.atleast_1d(p_params[1].eval())
            s_l2[param <= 1] = np.inf
            s_tau3[param <= 1] = np.inf
            s_tau4[param <= 1] = np.inf

        if name == "vonmises":
            s_tau3 = np.zeros_like(s_tau3)

        param_vals = [p.eval() if hasattr(p, "eval") else p for p in p_params]
        param_info = f"\n{name} params: {param_vals}"

        assert_allclose(
            p_dist.lmoment2(*p_params).eval(),
            s_l2,
            rtol=rtol,
            atol=atol,
            err_msg=f"L-moment 2 failed{param_info}",
        )
        assert_allclose(
            p_dist.lmoment3(*p_params).eval(),
            s_tau3,
            rtol=rtol,
            atol=atol,
            err_msg=f"L-skewness (tau3) failed{param_info}",
        )
        assert_allclose(
            p_dist.lmoment4(*p_params).eval(),
            s_tau4,
            rtol=rtol,
            atol=atol,
            err_msg=f"L-kurtosis (tau4) failed{param_info}",
        )
