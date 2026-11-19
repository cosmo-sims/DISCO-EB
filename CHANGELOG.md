# Changelog

Releases are cut by python-semantic-release from the Conventional Commits on
`main`; each entry below is generated, so edit commit messages, not this file.

<!-- version list -->

## v2.1.0 (2026-11-19)

### Bug Fixes

- Cap the sensitivity columns of one launch
  ([`793de16`](https://github.com/cosmo-sims/DISCO-EB/commit/793de16483defb4878ad2dd536550d6dff9a0f87))

- Follow modax 0.5.0's derivative options
  ([`6f5408a`](https://github.com/cosmo-sims/DISCO-EB/commit/6f5408a1b2323ab0d0aa799a861a9508b525ce99))

- Keep --dist out of the default pytest options
  ([`a507e7b`](https://github.com/cosmo-sims/DISCO-EB/commit/a507e7b1f1d076b948a5acdebe5ca143fcecbcc3))

- Reject a CPL history that reaches w = -1
  ([`f42bc2d`](https://github.com/cosmo-sims/DISCO-EB/commit/f42bc2d556bd21bb4e00595ef5623af94a1c3728))

- Stop JAX preallocating GPU memory on import
  ([`a1c14ba`](https://github.com/cosmo-sims/DISCO-EB/commit/a1c14ba1eada0d9ff6786ed446f0b39b156a1bfb))

### Build System

- Require modax 0.4.0
  ([`a0288c0`](https://github.com/cosmo-sims/DISCO-EB/commit/a0288c0b07eac675643a600d2db2b81438fa9b90))

### Chores

- Remove stray files from the repository
  ([`9d74f14`](https://github.com/cosmo-sims/DISCO-EB/commit/9d74f140cff165ca419f816c65134ca4e3fa89b1))

### Documentation

- Add a flat LCDM entry point and update docs
  ([`7c4e774`](https://github.com/cosmo-sims/DISCO-EB/commit/7c4e7748b0520cebad007bc2df46012506caebf8))

- Describe physical densities and the scale-factor solve
  ([`848613c`](https://github.com/cosmo-sims/DISCO-EB/commit/848613ce53e36d19849cfc2fdaac22ce7e133375))

- Describe the sensitivity cap and drop stale notes
  ([`15fcdf8`](https://github.com/cosmo-sims/DISCO-EB/commit/15fcdf8f4d86fc914333ad4af9849e3c2b826b4b))

- Draw the example batch uniformly from a key
  ([`6e7f4b6`](https://github.com/cosmo-sims/DISCO-EB/commit/6e7f4b6924d9970604c15fa9dda664c2bd1dd1c9))

- Mention the parameter sweep in AGENTS.md
  ([`23b45a8`](https://github.com/cosmo-sims/DISCO-EB/commit/23b45a897de9062d9b214e5569a39d734df0739c))

- Merge the examples into one jitted batch
  ([`6f5355a`](https://github.com/cosmo-sims/DISCO-EB/commit/6f5355a153a654ab1e950a49a3099ef12d24269d))

- Port the example notebooks to scale factors
  ([`2eaaff1`](https://github.com/cosmo-sims/DISCO-EB/commit/2eaaff13ec70bb22d284926cf19458d1f0f82455))

- Port the notebooks to example scripts
  ([`628285a`](https://github.com/cosmo-sims/DISCO-EB/commit/628285aaa5e40c1639f7bd45be2db8f6f00da0e2))

- Record the noise-limited test gates
  ([`36f8c76`](https://github.com/cosmo-sims/DISCO-EB/commit/36f8c76702571daac0af9cb25f1462e6d54b8843))

- Record the wider differentiable parameters
  ([`cf16d16`](https://github.com/cosmo-sims/DISCO-EB/commit/cf16d160da5a2053dd49803cc3c27f0e0735a30a))

- Reword the solver module docstrings
  ([`91b4f93`](https://github.com/cosmo-sims/DISCO-EB/commit/91b4f93026f417ffb066df29c64a49c28c62a4d5))

- Rewrite the notebooks
  ([`5b671b0`](https://github.com/cosmo-sims/DISCO-EB/commit/5b671b0d8bcc5444ddeceb82d078437368956c74))

- Rewrite the notebooks
  ([`d7cbc25`](https://github.com/cosmo-sims/DISCO-EB/commit/d7cbc254ccfae8c8c88cf87323767717d6e59b32))

- Shorten the component docstrings
  ([`24ca636`](https://github.com/cosmo-sims/DISCO-EB/commit/24ca6365376b58271766e1c4235004264175d986))

- Shorten the module docstrings
  ([`f52564c`](https://github.com/cosmo-sims/DISCO-EB/commit/f52564c3ec7a8d183e966f7d95a3fe401fe4f158))

- Shorten the module docstrings
  ([`24dfb09`](https://github.com/cosmo-sims/DISCO-EB/commit/24dfb09b2cc9d80794042501b077628ae0bdc2a8))

- Simplify the model docstrings
  ([`e5d4c2f`](https://github.com/cosmo-sims/DISCO-EB/commit/e5d4c2ffe56b69a5fea23706c849c35a4a8640b0))

- Turn the curvature example into N-body ICs
  ([`2c92f83`](https://github.com/cosmo-sims/DISCO-EB/commit/2c92f833e1c7f4c0429ebf42d855a4611c8977b5))

- Turn the examples into notebooks
  ([`694225c`](https://github.com/cosmo-sims/DISCO-EB/commit/694225ca71c9727855459879b8fde4f9a30029ad))

### Features

- Add split_derivatives to solve_eb
  ([`19c4f9e`](https://github.com/cosmo-sims/DISCO-EB/commit/19c4f9e623ef0f6b28911f46ec10ca54df439d88))

- Apply a model to its cosmologies
  ([`89e8405`](https://github.com/cosmo-sims/DISCO-EB/commit/89e8405e9bcacc0a4d60855665410748518ee918))

- Close the budget as a root of H(a=1) = H0
  ([`edefaa1`](https://github.com/cosmo-sims/DISCO-EB/commit/edefaa1e75e5097e7dd87962ece3ff8790add062))

- Differentiate Omegak, mnu, w_DE_0, w_DE_a
  ([`963cd60`](https://github.com/cosmo-sims/DISCO-EB/commit/963cd6022666cab7b605058af3786da9cb6ab973))

- Integrate in the scale factor, with no background pass
  ([`f1d63db`](https://github.com/cosmo-sims/DISCO-EB/commit/f1d63db99e95d09754d6933da3ad6ab136559e80))

- List a component under names of its own
  ([`9c23413`](https://github.com/cosmo-sims/DISCO-EB/commit/9c2341340f2ca53748a594d587324971ea0b8882))

- Name the cosmological model in every solve
  ([`dccb31c`](https://github.com/cosmo-sims/DISCO-EB/commit/dccb31cceb8acfda78a453c2a1913ba7fc1686f5))

- Parametrize models by physical densities
  ([`72161e1`](https://github.com/cosmo-sims/DISCO-EB/commit/72161e1a2ce3f7cc76ccdaa3418bf23ef1d94b0b))

- Take z_re as input and give tau_reio as output
  ([`81cdbef`](https://github.com/cosmo-sims/DISCO-EB/commit/81cdbef796d047fc613c1453a20fac9ce89c0423))

- Trace solve_eb under jit; P(k) from it
  ([`3754a6f`](https://github.com/cosmo-sims/DISCO-EB/commit/3754a6fde025cd07c974546a4a1124c986a7521d))

- Type Solution by the names solve_eb keeps
  ([`4860da1`](https://github.com/cosmo-sims/DISCO-EB/commit/4860da130cef857e8d6ff4a8b01217f698b65114))

### Performance Improvements

- Sum the neutrino density elementwise
  ([`5de2029`](https://github.com/cosmo-sims/DISCO-EB/commit/5de2029f441fbbaba3ad3ba18fac4ce2462ea339))

- **sensitivities**: One replica per variable
  ([`6d4d8e5`](https://github.com/cosmo-sims/DISCO-EB/commit/6d4d8e53de7cccd544e5be1f6def10126e0ea289))

### Refactoring

- Adopt a plain convention
  ([`759bb54`](https://github.com/cosmo-sims/DISCO-EB/commit/759bb54ee854b71755889f838d8fb1d58e53347c))

- Cache recombination_model with functools
  ([`dfe7a15`](https://github.com/cosmo-sims/DISCO-EB/commit/dfe7a15fa17d154a69e91f660ccfc2ce6527c312))

- Declare differentiable keys on components
  ([`f861f60`](https://github.com/cosmo-sims/DISCO-EB/commit/f861f603edb6215fa03fbd0a5b53dd0a0a6ab746))

- Declare keys the solve never reads
  ([`96a395a`](https://github.com/cosmo-sims/DISCO-EB/commit/96a395abc21c67b09809a4f7b712ecb8886f9d54))

- Declare parameters on the components
  ([`200cc7f`](https://github.com/cosmo-sims/DISCO-EB/commit/200cc7fa91ef5a0b78d23ef0c58a8f3fd5d5bd0c))

- Declare sums on the components that read them
  ([`3a34c6e`](https://github.com/cosmo-sims/DISCO-EB/commit/3a34c6e240d304d0302b2b31b5088b9cfa7a33d7))

- Differentiate along directions only
  ([`b8bbad9`](https://github.com/cosmo-sims/DISCO-EB/commit/b8bbad96cf46d41ebb8d7164a038f7a1d19efb37))

- Differentiate through modax's directions
  ([`cdb1558`](https://github.com/cosmo-sims/DISCO-EB/commit/cdb15580b9fa06b22286e353748143380743fc52))

- Drop _sensitivities for modax's rules
  ([`0c73ec2`](https://github.com/cosmo-sims/DISCO-EB/commit/0c73ec2e352e4346c1b76151f3a66ac23b379296))

- Drop ArrayLike hints in power spectrum
  ([`3494fbb`](https://github.com/cosmo-sims/DISCO-EB/commit/3494fbb1f6d3d5af37cf35d52c7f21d1dd64a504))

- Drop PerturbationLayout for the models
  ([`641b27e`](https://github.com/cosmo-sims/DISCO-EB/commit/641b27e9b72e14d3fa390b2533a71c5a6486ded1))

- Drop the gravity interface check
  ([`ca54a10`](https://github.com/cosmo-sims/DISCO-EB/commit/ca54a10586f09b0692689a10f9ffc50306370cbf))

- Drop the spline module
  ([`e037dc3`](https://github.com/cosmo-sims/DISCO-EB/commit/e037dc3800bc60f38189b593cb5e0554024288b4))

- Enable double precision on import
  ([`85ffebc`](https://github.com/cosmo-sims/DISCO-EB/commit/85ffebc94c55ecf470d8ae35d487299ed06f1f83))

- Flatten models/ into models.py
  ([`fcd2b51`](https://github.com/cosmo-sims/DISCO-EB/commit/fcd2b5167825f3f76caefe80b6ebcc03fa255e4b))

- Group modules into four packages
  ([`0295d78`](https://github.com/cosmo-sims/DISCO-EB/commit/0295d78df90216ea6d5057c8942a4d79892a0edd))

- Keep quantities through modax's hook API
  ([`6c8100e`](https://github.com/cosmo-sims/DISCO-EB/commit/6c8100e1fe714b5e501a814e56361ff23066db16))

- Make matter_power_spectrum one cosmology's
  ([`7012712`](https://github.com/cosmo-sims/DISCO-EB/commit/70127128ec0bd2d22237e539e0ac27fe42a91296))

- Make models classes with typed params
  ([`761b20a`](https://github.com/cosmo-sims/DISCO-EB/commit/761b20a284ee2d7558ff79fac8940a582a0bc148))

- Mark private modules with an underscore
  ([`92e75c9`](https://github.com/cosmo-sims/DISCO-EB/commit/92e75c99878410b1d97af6ca63ac979441c483d0))

- Merge ModelSpec into Model
  ([`1290924`](https://github.com/cosmo-sims/DISCO-EB/commit/129092435a8ae130433e41ce2a0571ae6354cab2))

- Pass arrays to matter_power_spectrum
  ([`4eb664d`](https://github.com/cosmo-sims/DISCO-EB/commit/4eb664d22baf10a456dad637cb42cc9e6cf70f12))

- Remove couplings a new model trips on
  ([`ea8db4c`](https://github.com/cosmo-sims/DISCO-EB/commit/ea8db4cdaf225afba06ec85e42a0b4ae564fa022))

- Remove models/_cosmology.py
  ([`a291c25`](https://github.com/cosmo-sims/DISCO-EB/commit/a291c25b44f5bba0008fd7f586fce550e873ce42))

- Remove the default model
  ([`088b145`](https://github.com/cosmo-sims/DISCO-EB/commit/088b145e05245e72a32ed480e496e4a10763f722))

- Remove the Kinematics component
  ([`d9ef524`](https://github.com/cosmo-sims/DISCO-EB/commit/d9ef5245d5db30d5bff66d1e7aec0819914cfd12))

- Rename to solve_matter_power_spectrum
  ([`1f1405d`](https://github.com/cosmo-sims/DISCO-EB/commit/1f1405dd4dec5fd26b591c75aade4a06aa9b8b51))

- Replace CosmologyBatch with a tuple
  ([`259fa09`](https://github.com/cosmo-sims/DISCO-EB/commit/259fa097259bdb929ec6740d6b80cfbd713cc198))

- Return the kept values from solve_eb
  ([`1f0a5db`](https://github.com/cosmo-sims/DISCO-EB/commit/1f0a5dbc5425e43426f129007562d06f91bf2b59))

- Separate the gravity interface from GR
  ([`c89a104`](https://github.com/cosmo-sims/DISCO-EB/commit/c89a104fb7c56b266279e067519b49231d281bd3))

- Split eb_model into species modules
  ([`26a1b81`](https://github.com/cosmo-sims/DISCO-EB/commit/26a1b8147f9946cab0806a1372bfef872dd9cee5))

- Split models into base and model modules
  ([`bba667f`](https://github.com/cosmo-sims/DISCO-EB/commit/bba667f07c6da0f17d4172c21e40126290766853))

- Split perturbations.py by job
  ([`d1cd8cc`](https://github.com/cosmo-sims/DISCO-EB/commit/d1cd8cc916999d30e39b5f1498f346d1cbaa2777))

- Split the background block by tracing
  ([`9aca7b5`](https://github.com/cosmo-sims/DISCO-EB/commit/9aca7b58a7318d54b20ac224934dd9c30665bc41))

- State the published cut's graph once
  ([`0da8171`](https://github.com/cosmo-sims/DISCO-EB/commit/0da8171eaac3170ba46377478d47c008fa1bc47e))

- Sum the initial mode over the species
  ([`8080f07`](https://github.com/cosmo-sims/DISCO-EB/commit/8080f07bf3c9a429a96e947666bdac38fe25539c))

- Trim the solver's test-only helpers
  ([`9fd04ee`](https://github.com/cosmo-sims/DISCO-EB/commit/9fd04eebf12608a7992a9137ebe0627572023520))

- Wrap prepare, pack and launch in one call
  ([`a0aa69b`](https://github.com/cosmo-sims/DISCO-EB/commit/a0aa69bbe583c0539144aa20656f0468aee6e46d))

- **examples**: Simplify the flat LCDM example
  ([`e4b501d`](https://github.com/cosmo-sims/DISCO-EB/commit/e4b501d54a05bac0634fcfc70e831a5375851344))

- **tests**: Shorten module docstrings
  ([`062b090`](https://github.com/cosmo-sims/DISCO-EB/commit/062b0902b828f25a057f9b86506a2cfa920a868d))

### Testing

- Add a species with a parameter of its own
  ([`228dd69`](https://github.com/cosmo-sims/DISCO-EB/commit/228dd69edcab08a4b38ed807958b3dd17f52c128))

- Check the field against its growth factor
  ([`a36af9f`](https://github.com/cosmo-sims/DISCO-EB/commit/a36af9fe75a8cf0ec97617d259534718d00a84c9))

- Compare grad and jacfwd to rounding
  ([`b00dbad`](https://github.com/cosmo-sims/DISCO-EB/commit/b00dbad21eca675d1fc9ca07b52c9694532c704e))

- Compare the split solve within its tolerance
  ([`c026201`](https://github.com/cosmo-sims/DISCO-EB/commit/c026201af21a12dc74f16f230770fdd8f7155a98))

- Keep the default run under five minutes
  ([`c1aeb7d`](https://github.com/cosmo-sims/DISCO-EB/commit/c1aeb7d699fd26c57085143fa52b151adb58c13d))

- Merge the tests into one file per folder
  ([`f268a97`](https://github.com/cosmo-sims/DISCO-EB/commit/f268a97b0427106319e192e240bac56375f67ff8))

- Mirror the source layout one-to-one
  ([`3e9fceb`](https://github.com/cosmo-sims/DISCO-EB/commit/3e9fcebfc8f504bac1b012fa00f4a69ac9c6e4e3))

- Run the host tests in parallel
  ([`ff56e4d`](https://github.com/cosmo-sims/DISCO-EB/commit/ff56e4d12583757541d49cfa0e6775f756e9c015))

- Share kernels between the GPU tests
  ([`32c4606`](https://github.com/cosmo-sims/DISCO-EB/commit/32c4606fa6a68b35fcb8b3902f7f10d7854a2b2e))

- Sweep parameter space for P(k) stability
  ([`71240f4`](https://github.com/cosmo-sims/DISCO-EB/commit/71240f459f47a15e9dba328aae623e4401fef9eb))

- Widen the c_b^2 switch-window gate to 0.25
  ([`830e98f`](https://github.com/cosmo-sims/DISCO-EB/commit/830e98f8864af44eb00d8b5eab001ca2f4039806))


## v2.0.1 (2026-11-12)

### Bug Fixes

- Stop each trajectory at tau0 on modax 0.3.0
  ([`e798ae4`](https://github.com/cosmo-sims/DISCO-EB/commit/e798ae496465cf18d81198a11c37d248050edb3d))

### Documentation

- Add the DISCO-EB logo to the README
  ([`d727a74`](https://github.com/cosmo-sims/DISCO-EB/commit/d727a74bda63ee19ca8c0ccde0ea0e4f791e76f9))

- Capitalise Einstein-Boltzmann in the logo
  ([`eb83c87`](https://github.com/cosmo-sims/DISCO-EB/commit/eb83c872ea4f4074a08ec66fed6acc7f0fd3bbb6))

- Rewrite docstrings and comments plainly
  ([`5a97260`](https://github.com/cosmo-sims/DISCO-EB/commit/5a972606cbda164bc01e652b6b7e965d17404489))

- Show the logo in the docs sidebar
  ([`07239ec`](https://github.com/cosmo-sims/DISCO-EB/commit/07239eca2cee64f154f8bed8203ea36a35780946))


## v2.0.0 (2026-11-11)

### Features

- Release the modax rewrite as disco-eb 2
  ([`2752e09`](https://github.com/cosmo-sims/DISCO-EB/commit/2752e09460d3ae0f009ff3f6be72df7a148a9984))

### Breaking Changes

- The diffrax evolve_perturbations* functions are gone. Solve with solve_eb or
  solve_matter_power_spectrum_batch, and get derivatives from matter_power_spectrum_jax. The CMB
  code moved to DISCO-CMB, and the stand-alone RECFAST solve was removed.


## v1.0.1 (2026-11-11)

### Bug Fixes

- Jax version defined in pyproject.toml
  ([`f2c8035`](https://github.com/cosmo-sims/DISCO-EB/commit/f2c803582a9d1cf6723cc551143e32c0df5bc73c))

### Build System

- Adopt uv, ruff, ty and pre-commit
  ([`ea322e4`](https://github.com/cosmo-sims/DISCO-EB/commit/ea322e4d05c6570967a87983a74280aaaa4e28df))

- Port to modax 0.2.0
  ([`05b854e`](https://github.com/cosmo-sims/DISCO-EB/commit/05b854ed8dcef7bbb3415a5f9b95bfffad04f2ca))

- Publish to PyPI as disco-eb
  ([`40d514d`](https://github.com/cosmo-sims/DISCO-EB/commit/40d514d4b1738751ec72caac8c938b7ad7a2a3c0))

- Require modax 0.2.2 and drop the fp32 hack
  ([`3b37527`](https://github.com/cosmo-sims/DISCO-EB/commit/3b37527f765ebdb89d26ad54640f69c464dc4600))

### Chores

- Relicense under MIT and add CITATION.cff
  ([`1da0756`](https://github.com/cosmo-sims/DISCO-EB/commit/1da0756b3d7f0b71145f2465262278dc4e28aa1c))

- Remove stale scripts and test data
  ([`b65c788`](https://github.com/cosmo-sims/DISCO-EB/commit/b65c788c694b8c1d4185c5880c9add4bf58f92a6))

### Continuous Integration

- Add lint, types, tests, docs and publish workflows
  ([`3524b8c`](https://github.com/cosmo-sims/DISCO-EB/commit/3524b8cab9db48204808851322c28cb814709274))

### Documentation

- Add a changelog for semantic-release
  ([`0d6cad3`](https://github.com/cosmo-sims/DISCO-EB/commit/0d6cad34549bf5160fdb50d1191c888dbb6fa782))

- Add a Sphinx site and a modax-style README
  ([`d59e880`](https://github.com/cosmo-sims/DISCO-EB/commit/d59e8809c13d2d2259c7b4bd8e64598b6efc593e))

- Add AGENTS.md
  ([`59cfc65`](https://github.com/cosmo-sims/DISCO-EB/commit/59cfc655566f146cade68ce3d2d468380d005393))

- Add the Recfast license to recombination.py
  ([`f1dd91f`](https://github.com/cosmo-sims/DISCO-EB/commit/f1dd91faabdbd2dc0adade240855a9b1b33b7d81))

### Refactoring

- De-duplicate component.py
  ([`624d06a`](https://github.com/cosmo-sims/DISCO-EB/commit/624d06abdd3a7422bb0c4a4886064aa53693ca73))

- De-duplicate eb_model.py
  ([`012f836`](https://github.com/cosmo-sims/DISCO-EB/commit/012f836f107a7f4f6fe4b53802d191b4c8b2d81f))

- De-duplicate perturbations.py helpers
  ([`f34bbbe`](https://github.com/cosmo-sims/DISCO-EB/commit/f34bbbeaf2cdc0eb28b561e8c3cfbfa2c708865a))

- Drop the stand-alone RECFAST solve
  ([`93386db`](https://github.com/cosmo-sims/DISCO-EB/commit/93386db400269711df1673e762758dfeb42c0d05))

- Expose the EB interface from discoeb
  ([`3c2a68f`](https://github.com/cosmo-sims/DISCO-EB/commit/3c2a68f2b331cc74dfe2d96248b01d4b81e8dc33))

- One sensitivity path for P(k) and solve_eb
  ([`fb21d3e`](https://github.com/cosmo-sims/DISCO-EB/commit/fb21d3eac5a549fc3a101a03744b3b1a457f64ed))

- Remove dead code
  ([`5bf36db`](https://github.com/cosmo-sims/DISCO-EB/commit/5bf36db5a3868f8999f5f9f97464f5c29a168739))

### Testing

- Share the host build fixtures in conftest
  ([`0f4bdc6`](https://github.com/cosmo-sims/DISCO-EB/commit/0f4bdc60adb87c43c0233bf6fc2c5a993a6316b7))
