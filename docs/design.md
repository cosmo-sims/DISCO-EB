# Design

DISCO-EB turns a list of components into one GPU kernel that integrates the
background, the recombination history and the perturbations of every wave
mode of every cosmology in a batch. This page describes the design choices underlying
the generation of that kernel.

## Assembling equations from components

A {py:class}`~discoeb.components.Component` declares its states, parameters
and settings via class attributes and its physics via methods. Each method
defines a particular quantity, and accepts as arguments the quantities it depends on, which can come from any other part of the model:

```python
class ColdDarkMatter(Species):
    delta_c = State(initial=lambda F: 0.75 * F[0])
    omega_c: ArrayLike = Parameter()

    def rho_c(self, omega_c, a):
        return omega_c / a**3

    @contributes("rho", "rho_matter")
    def rho_cdm(self, rho_c):
        return rho_c

    @derivative("delta_c")
    def continuity_c(self, h_prime):
        return -0.5 * h_prime
```

Here `continuity_c` reads the metric's `h_prime`, which the `Metric`
component provides. `rho_cdm` adds to the sums `rho` and `rho_matter`, which
the Friedmann equation and the initial conditions require. At compile time, DISCO-EB traces through all these methods, sorts them by dependency and compiles them into a single right-hand side function that defines the ODE system. Methods that depend only on cosmological parameters and not conformal time or scale factor will automatically be hoisted out of the ODE solve and evaluated once upfront. Methods that depend on conformal time or scale factor but not on wave mode (e.g. background expansion and non-perturbative recombination equations) will be split out into their own parent system and integrated with larger, independent step sizes than the perturbations, which are more oscillatory and stiff. See [modax hierarchies](https://lawrenceberry.github.io/modax/usage/sparse.html#hierarchies-of-one-way-coupled-blocks) for details.

## Fused integration of background, recombination and perturbations

CAMB and CLASS solve the background first, then recombination, and pass
both to the perturbations as interpolation tables. In contrast, DISCO-EB integrates all
three together. This makes the solve faster and more memory-efficient because each trajectory evaluates the background and recombination physics only at the times needed by the perturbations solver, rather than at a large fixed grid of points. In addition, this approach also makes it easier to propagate errors from the background and recombination into the perturbations.

In practice, the background and recombination systems are less oscillatory and less stiff than the perturbations, so they can be integrated with a larger step size. Since they couple one-way with the perturbations, DISCO-EB integrates them with a [modax hierachy](https://lawrenceberry.github.io/modax/usage/sparse.html#hierarchies-of-one-way-coupled-blocks), essentially a sub-system that can step independently ahead of the perturbations. The code sorts states whose rate does not depend on $k$ either directly or indirectly into such a sub-system.

Components are written with rates in conformal time. The driver divides each
by $a' = a\,aH$ such that the kernel integrates in coordinates of the scale factor (by default from `A_START` $= 2 \times 10^{-7}$ to 1). Every cosmology then ends at $a = 1$ today and we don't need to integrate the background to compute today's conformal age in advance of the ODE solve.

All of this is done at compile time. Once compiled, the compiled integration kernel can be invoked repeatedly on the GPU with different parameters and initial conditions, without recompilation or synchronisation with the CPU.

## Gauge choice and variable conventions

We follow the conventions of [Ma & Bertschinger
(1995)](https://arxiv.org/abs/astro-ph/9506072) in synchronous gauge: the
metric potentials $\eta$ and $h'$, the fluid variables $\delta$ and
$\theta$, and the hierarchies $F_\ell$, $E_\ell$ and $N_\ell$. The
normalisation is unit comoving curvature, $\mathcal{R} = 1$, at the start. A
density is a physical density parameter,

$$
\omega_i(a) = \frac{\rho_i(a)}{\rho_{100}}, \qquad \rho_{100} = \frac{3 H_{100}^2}{8 \pi G}, \quad H_{100} = 100\ \mathrm{km\,s^{-1}\,Mpc^{-1}},
$$

so the Friedmann equation reads $(aH)^2 = H_{100}^2 a^2 \sum_i \omega_i(a) - K$
and the Hubble constant is an output, $aH$ at $a = 1$.

Today $\omega_i(1) = \Omega_i h^2$, but the constant $H_{100}$ defines
$\omega_i$, not $h$. Every density is then independent of $H_0$, and the
kernel never needs the Hubble constant. Taking $H_0$ as an input instead
would add the constraint $h^2 = \sum_i \omega_i(1)$, and some component
would have to absorb it, which couples the components to each other. With
physical densities, curvature enters as one more density,
$K = -\omega_k H_{100}^2$, and a new species adds its own term to the sum
without changing the others. A theory whose expansion rate depends on a
state, such as a scalar field, needs no shooting solve to match a given
$H_0$ and obtains $H_0$ from the solve itself. To work in $\Omega_m$,
$\Omega_b$ and $H_0$, `Model.from_density_parameters` converts them before
the solve and closes the budget with $\omega_\mathrm{de}$.

Wave numbers are in 1/Mpc, conformal times in Mpc and neutrino masses in
eV.

## Optimised integration of a sparse stiff system

The system is stiff, so we integrate it with the Rodas5P Rosenbrock method, which requires factorising the system's Jacobian at every step. modax differentiates the compiled right-hand side with
[Enzyme](https://enzyme.mit.edu) and factorises the result with a sparse LU routine.
To obtain the system's sparsity pattern (which inputs affect which outputs), DISCO-EB traces through all the methods of a particular model, recording which state elements they depend on. Using this pattern, modax can reduce the number of auto-differentiation calls and the complexity and memory requirements of the LU factorisation, which makes the solve faster and more memory-efficient.

Each wave mode of each cosmology is an independent stiff system, so the kernel integrates them in parallel on the GPU, grouped into blocks of 32 threads that can operate independently of each other. By sorting similar wave modes into the same blocks, the groups of threads finish their integrations in similar numbers of steps, wasting as little GPU time as possible compared to a naive JAX `vmap`.

Since we solve hundreds of wave modes for hundreds cosmologies at once across hundreds of scale factors, the state history of a large batch may not fit on the GPU device. To save memory therefore,
`solve_eb` will keep only the state you need at each scale factor. If you need some simple derived quantity, you can even define a custom hook function that computes it on the fly and saves and returns only that, so memory scales only with the number of kept quantities and not with the state size.

## Derivatives by forward sensitivities

DISCO-EB can compute directional derivatives of the E-B state vector (and other hooked quantities derived from the state vector) with respect to the cosmological parameters, and is fully compatible with `jax.jvp` or `jax.jacfwd`, using forward sensitivity analysis: for each tangent direction, the kernel integrates a sensitivity column alongside the state. The result is derivatives at a cost over the plain solve that is roughly directly proportional to the number of tangent directions requested. Reverse mode is not yet implemented because the number of outputs (the E-B state vector) is larger than the inputs (the cosmological parameters) so forward mode is more efficient.
