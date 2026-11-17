"""Momentum quadrature for massive neutrinos: CAMB's bins and a Gauss-Laguerre rule."""

import functools

import jax
import jax.numpy as jnp
import jax.scipy
import numpy as np


def generalized_gauss_laguerre_weights(n, alpha):
    """
    Return the nodes and weights of the n-point generalized Gauss-Laguerre
    rule for integrals of the form
        ∫₀∞ x^α f(x) e^(-x) dx.

    Parameters:
        n : int
            Number of quadrature points.
        alpha : float
            The exponent in the weight function x^α e^(-x).

    Returns:
        nodes : ndarray
            The quadrature nodes (abscissae).
        weights : ndarray
            The quadrature weights.
    """
    # i = 1, ..., n
    i = jnp.arange(1, n + 1)

    # Diagonal: a_i = 2i - 1 + alpha
    a = 2 * i - 1 + alpha

    # Off-diagonal for i = 1, ..., n-1: b_i = sqrt(i (i + alpha))
    i_off = jnp.arange(1, n)
    b = jnp.sqrt(i_off * (i_off + alpha))

    # Symmetric tridiagonal Jacobi matrix (Golub-Welsch)
    J = jnp.diag(a) + jnp.diag(b, 1) + jnp.diag(b, -1)

    # The eigenvalues are the nodes
    nodes, V = jnp.linalg.eigh(J)

    # Each weight is the squared first component of its eigenvector times the
    # zeroth moment Γ(α+1).
    if isinstance(alpha, int):
        weights = (V[0, :] ** 2) * jax.scipy.special.factorial(alpha)
    else:
        weights = (V[0, :] ** 2) * jax.scipy.special.gamma(alpha + 1)

    return nodes, weights


def _neutrino_momentum_bins(nqmax: int) -> tuple[jax.Array, jax.Array]:
    fermi_dirac_const = 7 * jnp.pi**4 / 120
    if nqmax == 3:
        q = jnp.array([0.913201, 3.37517, 7.79184])
        dlfdlq = -q / (1 + jnp.exp(-q))
        w = jnp.array([0.0687359, 3.31435, 2.29911]) / (-0.25 * dlfdlq)
    elif nqmax == 4:
        q = jnp.array([0.7, 2.62814, 5.90428, 12.0])
        dlfdlq = -q / (1 + jnp.exp(-q))
        w = jnp.array([0.0200251, 1.84539, 3.52736, 0.289427]) / (-0.25 * dlfdlq)
    elif nqmax == 5:
        q = jnp.array([0.583165, 2.0, 4.0, 7.26582, 13.0])
        dlfdlq = -q / (1 + jnp.exp(-q))
        w = jnp.array([0.0081201, 0.689407, 2.8063, 2.05156, 0.12681]) / (
            -0.25 * dlfdlq
        )
    else:
        alpha = 1
        q, w = generalized_gauss_laguerre_weights(nqmax, alpha)
        w *= q**3 / (1 + jnp.exp(-q)) * q**-alpha
    return q, w / fermi_dirac_const


def neutrino_momentum_bins(nqmax: int):
    """Return ``(q, w, dlnf0_dlnq)`` momentum bins as read-only NumPy arrays.

    ``q`` are the nodes (in units of ``k_B T_nu0 / c``) and ``w`` the weights,
    normalised so one massless flavour integrates to unit density. For
    ``nqmax`` of 3, 4 or 5 they are CAMB's high-accuracy formulas, for higher
    counts a generalised Gauss-Laguerre rule. ``dlnf0/dlnq = -q/(1 +
    exp(-q))`` is the log-derivative of the Fermi-Dirac background
    distribution that sources the ``psi`` hierarchy. Cached per ``nqmax``,
    because the rule's eigendecomposition, op by op, cost a second
    of compiles per call.
    """

    return _momentum_bins(int(nqmax))


@functools.lru_cache(maxsize=None)
def _momentum_bins(nqmax: int):
    with jax.ensure_compile_time_eval():
        q, w = _neutrino_momentum_bins(nqmax)
    q = np.asarray(q, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    dlfdlq = -q / (1.0 + np.exp(-q))
    for array in (q, w, dlfdlq):
        array.flags.writeable = False
    return q, w, dlfdlq
