"""The generalised Gauss-Laguerre rule behind the massive-neutrino momentum bins."""

import jax
import jax.numpy as jnp
import jax.scipy


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
