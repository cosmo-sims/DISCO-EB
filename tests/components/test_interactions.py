"""Interactions (``interactions``): Thomson scattering."""

from helpers import MODELS, build_rhs

# --- The Einstein-Boltzmann component model, on the host.


def test_thomson_scattering_writes_the_photon_and_baryon_derivatives():
    model, _ = build_rhs(MODELS["flat-lcdm"])
    lines = [ell.strip() for ell in model.source.splitlines()]
    assert (
        "d_vb = Baryons__d_vb(Baryons_self, adotoa, vb, k, cs2, clxb) + ThomsonScattering__d_vb_thomson(ThomsonScattering_self, photbar, opacity, slip)"
        in lines
    )
    assert lines.index(
        "Photons__d_theta(Photons_self, k, z, sigma, theta, inv_tau, rsa, c_Photons__d_theta)"
    ) < lines.index(
        "ThomsonScattering__d_theta_thomson(ThomsonScattering_self, opacity, slip, polter, theta, rsa, c_ThomsonScattering__d_theta_thomson)"
    )
    assert (
        "ThomsonScattering__d_e_thomson(ThomsonScattering_self, opacity, polter, e, rsa, c_ThomsonScattering__d_e_thomson)"
        in lines
    )
    # Neutrinos are collisionless: one writer.
    assert sum("__d_n(" in ell for ell in lines) == 1
