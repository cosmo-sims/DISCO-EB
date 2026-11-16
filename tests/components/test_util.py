"""The neutrino momentum bins and their Gauss-Laguerre rule (``_util``)."""

import numpy as np
import pytest

from discoeb.components._util import (
    generalized_gauss_laguerre_weights,
    neutrino_momentum_bins,
)


@pytest.mark.parametrize("n", [6, 10])
def test_the_rule_is_numpys_for_alpha_zero(n):
    nodes, weights = generalized_gauss_laguerre_weights(n, 0.0)
    expected_nodes, expected_weights = np.polynomial.laguerre.laggauss(n)
    np.testing.assert_allclose(nodes, expected_nodes, rtol=1e-10)
    np.testing.assert_allclose(weights, expected_weights, rtol=1e-8)


def test_the_rule_integrates_polynomials_against_the_weight():
    """An ``n``-point rule is exact to degree ``2n - 1``: the integrals are gamma functions."""
    from math import gamma

    nodes, weights = generalized_gauss_laguerre_weights(6, 1.0)
    for degree in (0, 3, 11):
        integral = float(np.sum(np.asarray(weights) * np.asarray(nodes) ** degree))
        assert integral == pytest.approx(gamma(degree + 2.0), rel=1e-9)


@pytest.mark.parametrize("nqmax", [3, 4, 5, 8])
def test_one_massless_flavour_integrates_to_unit_density(nqmax):
    q, w, dlnf0 = neutrino_momentum_bins(nqmax)
    assert q.shape == w.shape == dlnf0.shape == (nqmax,)
    assert np.sum(w) == pytest.approx(1.0, rel=5e-3)
    np.testing.assert_allclose(dlnf0, -q / (1.0 + np.exp(-q)))
    assert not q.flags.writeable
    assert neutrino_momentum_bins(nqmax)[0] is q  # cached
