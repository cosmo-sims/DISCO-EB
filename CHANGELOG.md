# Changelog

Releases are cut by python-semantic-release from the Conventional Commits on
`main`; each entry below is generated, so edit commit messages, not this file.

<!-- version list -->

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
