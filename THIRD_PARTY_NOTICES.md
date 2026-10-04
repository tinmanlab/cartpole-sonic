# Third-party notices

## MuJoCo 3.14.0

This repository redistributes the official JavaScript/WebAssembly build of MuJoCo 3.14.0 under `vendor/mujoco/`.

- Project: https://github.com/google-deepmind/mujoco
- Package: `@mujoco/mujoco@3.14.0`
- License: Apache License 2.0
- License text: `vendor/mujoco/LICENSE`

No modifications are made to the vendored `mujoco.js` or `mujoco.wasm` files.

## MCP-B browser runtime

The demo loads the MCP-B global browser runtime from unpkg at runtime. It is not vendored in this repository.

## Referenced research/software

GEAR-SONIC, GR00T Whole-Body Control, ProtoMotions, FSQ, and MuJoCo Playground are referenced for educational/research context. Their source code is not copied into this repository except for the MuJoCo WASM artifacts noted above.

## Native module experiment

`native/core_smoke.py` imports NVIDIA's official `UniversalTokenModule` and `BaseModule` from an external checkout pinned to commit `b042411fae38ee4d1af9aac82a37a1f8d14d6dd0` of https://github.com/NVlabs/GR00T-WholeBodyControl. It loads the upstream configuration files and overrides only the documented CartPole dimensions/settings in memory. Those source files are not vendored or patched here; the upstream notices and license remain with the external checkout.

The experiment also directly uses the PyPI `vector-quantize-pytorch` FSQ implementation, pinned in `native/requirements.lock`. The local contribution is the configuration adapter, synthetic test fixtures, provenance checks, numerical acceptance tests and scope documentation—not an independently authored SONIC or FSQ implementation. See `native/README.md` for exact adaptations and claim limits.
