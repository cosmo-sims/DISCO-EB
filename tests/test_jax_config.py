"""Tests of JAX's configuration on import (``_jax_config``)."""

import os

import jax
import jax.numpy as jnp

import discoeb  # noqa: F401


def test_importing_the_package_enables_double_precision():
    assert jax.config.read("jax_enable_x64")
    assert jnp.asarray(1.0).dtype == jnp.float64


def test_jax_does_not_preallocate_the_gpu():
    fraction = "XLA_PYTHON_CLIENT_MEM_FRACTION" in os.environ
    assert fraction or os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
