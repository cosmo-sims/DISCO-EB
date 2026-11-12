import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class spline_interpolation(object):
    def __init__(
        self,
        xin: jnp.ndarray,
        yin: jnp.ndarray,
    ):
        """
        Build a natural cubic spline through the points (x, y).

        Solves the tridiagonal system for the second derivatives with the
        Thomas algorithm, with natural boundary conditions (S[0] = S[-1] = 0).

        Args:
            x: 2D array of x-coordinates, sorted in increasing order.
            y: 2D array of y-coordinates (first axis as x, second the batch).
        """
        # Promote 1d xin or yin to 2d with a batch axis
        xin = xin.reshape(xin.shape[0], -1)
        yin = yin.reshape(yin.shape[0], -1)

        # Forward-fill trailing NaNs. Filled x grows by 1 percent per point to
        # stay increasing.
        def _fill_forward(last_observed_yi, yi, fac):
            yi = jnp.where(jnp.isnan(yi), last_observed_yi * fac, yi)
            return yi, yi

        n = yin.shape[0]
        _, y = jax.lax.scan(lambda c, v: _fill_forward(c, v, 1.0), yin[0], yin)
        _, x = jax.lax.scan(lambda c, v: _fill_forward(c, v, 1.01), xin[0], xin)

        if n < 2:
            raise ValueError("There must be at least two data points.")

        # Knot spacings
        h = x[1:] - x[:-1]  # shape: (n-1,)

        # Second derivatives S at the knots, by the Thomas algorithm
        m = n - 2  # number of interior points
        y = y + jnp.zeros((1, x.shape[1]))  # broadcast y to x's batch axis
        B = y.shape[1]
        Bx = x.shape[1]
        if m > 0:
            # Tridiagonal system for S[1] ... S[n-2]
            a = h[:-1]  # lower diagonal (length m)
            b_diag = 2 * (h[:-1] + h[1:])  # main diagonal (length m)
            c = h[1:]  # upper diagonal (length m)
            d = 6 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])

            # Forward elimination
            cp = jnp.zeros((m, Bx))
            cp = cp.at[0].set(c[0] / b_diag[0])

            def forward_step_cp(i, cp):
                denom = b_diag[i] - a[i] * cp[i - 1]
                return cp.at[i].set(c[i] / denom)

            cp = jax.lax.fori_loop(1, m - 1, forward_step_cp, cp)

            dp = jnp.zeros((m, B))
            dp = dp.at[0].set(d[0] / b_diag[0])

            def forward_step_dp(i, dp):
                denom = b_diag[i] - a[i] * cp[i - 1]
                return dp.at[i].set((d[i] - a[i] * dp[i - 1]) / denom)

            dp = jax.lax.fori_loop(1, m, forward_step_dp, dp)

            # Backward substitution
            S_interior = jnp.zeros((m, B))
            S_interior = S_interior.at[m - 1].set(dp[m - 1])

            def backward_step(i, S_int):
                idx = m - 2 - i
                S_int = S_int.at[idx].set(dp[idx] - cp[idx] * S_int[idx + 1])
                return S_int

            S_interior = jax.lax.fori_loop(0, m - 1, backward_step, S_interior)

            zeros = jnp.zeros((1, B))
            # Natural boundary conditions: S[0] = S[-1] = 0.
            S_full = jnp.concatenate([zeros, S_interior, zeros], axis=0)
        else:
            # Two points: linear interpolation
            S_full = jnp.zeros((2, B))

        self._n_ = n
        self._x_, self._y_, self._S_full_ = x, y, S_full

    # pytree flattening
    def tree_flatten(self):
        return (self._x_, self._y_, self._S_full_), {"n": self._n_}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)  # skip __init__
        obj._x_, obj._y_, obj._S_full_ = children
        obj._n_ = aux_data["n"]
        return obj

    def _find_index(self, x_new):
        """Return the interval index ``i`` with ``x[i] <= x_new < x[i+1]``, clipped to the grid."""
        x_new_ = jnp.atleast_2d(x_new)
        return jnp.clip(
            jnp.sum(self._x_[:, None, :] < x_new_[None, :, :], axis=0) - 1,
            0,
            self._n_ - 2,
        )

    def evaluate(self, x_new: jnp.ndarray):
        """
        Evaluate the spline at new x positions.

        Args:
            x_new: scalar or 1D array of x values.

        Returns:
            The interpolated y values.
        """
        x_new = jnp.atleast_1d(x_new)
        x_new = x_new.reshape(x_new.shape[0], -1)  # add a batch axis if missing
        idx = self._find_index(x_new)
        xidx = jnp.take_along_axis(self._x_, idx, axis=0)
        xidxp1 = jnp.take_along_axis(self._x_, idx + 1, axis=0)
        yidx = jnp.take_along_axis(self._y_, idx, axis=0)
        yidxp1 = jnp.take_along_axis(self._y_, idx + 1, axis=0)
        Sidx = jnp.take_along_axis(self._S_full_, idx, axis=0)
        Sidxp1 = jnp.take_along_axis(self._S_full_, idx + 1, axis=0)

        h_local = xidxp1 - xidx
        d_val = x_new - xidx  # local offset
        t = d_val / h_local  # normalized coordinate
        A = 1 - t
        B = t
        # Cubic from the end values and second derivatives
        y_new = (
            A * yidx
            + B * yidxp1
            + ((A**3 - A) * Sidx + (B**3 - B) * Sidxp1) * (h_local**2) / 6.0
        )
        return y_new


@register_pytree_node_class
class FastUniformCubicSpline1D(object):
    """Natural cubic spline on a uniform grid, stored as polynomial coefficients in the local coordinate."""

    def __init__(self, xin: jnp.ndarray, yin: jnp.ndarray):
        h = xin[1] - xin[0]
        dx_inv = 1.0 / h

        n = yin.shape[0]

        # Forward-fill trailing NaNs in y
        def _fill_forward(last, v):
            v = jnp.where(jnp.isnan(v), last, v)
            return v, v

        _, y = jax.lax.scan(_fill_forward, yin[0], yin)

        if n < 3:
            raise ValueError("There must be at least three data points.")

        dx_inv_sq = dx_inv * dx_inv
        d = (6.0 * dx_inv_sq) * (y[:-2] - 2.0 * y[1:-1] + y[2:])

        # Thomas algorithm: forward elimination
        def _forward_step(carry, d_i):
            cp_prev, dp_prev = carry
            cp_curr = 1.0 / (4.0 - cp_prev)
            dp_curr = cp_curr * (d_i - dp_prev)
            return (cp_curr, dp_curr), (cp_curr, dp_curr)

        cp0 = 0.25
        dp0 = 0.25 * d[0]

        _, (cp_rest, dp_rest) = jax.lax.scan(_forward_step, (cp0, dp0), d[1:])

        cp = jnp.concatenate([jnp.array([cp0]), cp_rest])
        dp = jnp.concatenate([jnp.array([dp0]), dp_rest])

        # Backward substitution
        def _backward_step(S_next, cp_dp):
            cp_i, dp_i = cp_dp
            S_i = dp_i - cp_i * S_next
            return S_i, S_i

        _, S_rev = jax.lax.scan(
            _backward_step, dp[-1], (cp[:-1], dp[:-1]), reverse=True
        )
        S_interior = jnp.concatenate([S_rev, dp[-1:]])
        S_full = jnp.pad(S_interior, (1, 1)) * h * h / 6.0

        y_i, y_ip1 = y[:-1], y[1:]
        S_i, S_ip1 = S_full[:-1], S_full[1:]
        c0 = y_i
        c1 = y_ip1 - y_i - 2.0 * S_i - S_ip1
        c2 = 3.0 * S_i
        c3 = S_ip1 - S_i

        self._h_ = h
        self._coeffs_ = jnp.stack([c0, c1, c2, c3], axis=-1)
        self._x0_ = xin[0]
        self._dx_inv_ = 1.0 / h
        self._max_idx_ = n - 2

    # pytree flattening
    def tree_flatten(self):
        return (self._coeffs_, self._x0_, self._h_), {"max_idx": self._max_idx_}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)  # skip __init__
        obj._coeffs_, obj._x0_, obj._h_ = children
        obj._max_idx_ = aux_data["max_idx"]
        obj._dx_inv_ = 1.0 / obj._h_
        return obj

    def _find_index_t(self, x_new):
        i = (x_new - self._x0_) * self._dx_inv_
        # The default integer type (int64 under x64, else int32) matches the
        # literal indices in the dynamic slices below.
        idx = jnp.clip(jnp.floor(i).astype(int), 0, self._max_idx_)
        return idx, i - idx.astype(x_new.dtype)

    def evaluate(self, x_new: jnp.ndarray):
        # Interval i with x[i] <= x_new < x[i+1], and t in [0, 1) within it
        idx, t = self._find_index_t(x_new)
        c = jax.lax.dynamic_slice(self._coeffs_, (idx, 0), (1, 4)).reshape(4)
        # Estrin's scheme: more instruction-level parallelism than Horner's
        t2 = t * t
        p01 = c[..., 0] + t * c[..., 1]
        p23 = c[..., 2] + t * c[..., 3]
        y_new = p01 + t2 * p23
        return y_new
