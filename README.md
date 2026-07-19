<table>
  <tr>
    <td>
      <img src="images/qdx_logo_wordart.png" alt="overview" width="200"/>
    </td>
    <td>
      <h1>QEC AI-discovery with PyTorch ⚛️🤖🚀</h1>
    </td>
  </tr>
</table>


[![Paper](https://img.shields.io/badge/npj_qi-10_126_(2024)-b31b1b.svg)](https://www.nature.com/articles/s41534-024-00920-y)  <a href="https://colab.research.google.com/drive/1nU9Xivfms_wXrJmv0F6uFz4_DOWoryhg?usp=sharing" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> 

Code repository for the paper "Simultaneous Discovery of Quantum Error Correction Codes and Encoders with a Noise-Aware Reinforcement Learning Agent" by *Jan Olle, Remmy Zen, Matteo Puviani and Florian Marquardt*.

> **NOTE — PyTorch port.** This is a faithful PyTorch conversion of the
> original JAX implementation (https://github.com/jolle-ag/qdx). All JAX /
> flax / distrax / optax / gymnax dependencies have been replaced by torch
> re-implementations that reproduce the original numerics, while keeping
> every tensor an eagerly-evaluated `torch.Tensor` that can be printed or
> inspected in a debugger at any point (no tracing, no jit):
>
> * `qdx/torch_random.py` — a bit-exact port of JAX's threefry2x32 PRNG
>   (`PRNGKey`, `split`, `fold_in`, `uniform`, `normal`, `categorical`,
>   `permutation`, `randint`, `choice`) plus flax's SHA-1 parameter-RNG
>   folding, so random streams match the JAX version bit for bit. (The PRNG
>   itself runs on NumPy uint32 arithmetic — randomness needs no autodiff —
>   and converts to torch tensors at the boundary.)
> * `qdx/torch_nn.py` — the ActorCritic network (orthogonal initialization
>   via QR), a distrax-compatible `Categorical`, the PPO loss whose
>   gradients are computed by **`torch.autograd`** (replacing
>   `jax.value_and_grad`; `torch.minimum`/`maximum` share JAX's balanced
>   0.5/0.5 tie gradients, and clipping is expressed through them so the
>   gradient semantics match `jnp.clip` exactly), and the optax
>   `chain(clip_by_global_norm, adam)` optimizer with linear LR schedule.
> * `qdx/torch_env_base.py` — the gymnax `Environment` step/reset semantics
>   (including auto-reset key splitting) and the
>   `FlattenObservationWrapper`/`LogWrapper`.
>
> `jax.vmap`/`jax.lax.scan` are replaced by explicit loops, so training is
> sequential and slower than the jitted JAX original — but every quantity is
> debuggable and all floating-point results match the JAX CPU version within
> float32 round-off (see `tests/` for the verification suite).
>
> **Acceleration.** Everything runs on CPU by default (which is the
> configuration verified against JAX). The network and PPO update can be
> moved to an accelerator by setting `config["DEVICE"] = "mps"` (Apple GPU)
> or `"cuda"` — the environments stay on CPU, where their small integer
> tableau updates are fastest. Accelerator floats round slightly differently,
> so expect small numerical drift vs. the CPU/JAX reference.

## Description
This library can be used to train Reinforcement Learning (RL) agents to codiscover quantum error correction (QEC) codes and their encoding circuits *from scratch, without any additional domain knowledge* except how many errors are not detected given the quantum circuit it has built.

The RL agent can be made *noise-aware*, meaning that it learns to produce encoding strategies simultaneously for a range of noise models, making it applicable in very broad situations. 

<img src="images/overview.png" alt="overview" width="800"/>

The whole RL algorithm, including the Clifford simulations of the quantum circuits, is implemented in PyTorch in this port (the original uses Jax and builds upon [PureJaxRL](https://github.com/luchris429/purejaxrl?tab=readme-ov-file)).

## Installation

QDX can be installed by:

1. Cloning the repository

``` bash
git clone https://github.com/HHTseng/qdx-torch.git
cd qdx-torch
```

2. Installing requirements
``` bash
pip install -r requirements.txt
```

## Usage Example

We include a [demo](notebooks/demo.ipynb) jupyter notebook for two different situations: [[7,1,3]] code discovery in a fixed symmetric depolarizing noise channel and noise-aware [[6,1]] code discovery in a biased noise channel. The scripts `demo1.py` and `demo2.py` run the same two examples from the command line.

## Verifying against the JAX original

The `tests/` directory contains a comparison suite that runs the original JAX
implementation side by side with this PyTorch port and checks that PRNG
streams, environment trajectories, network initialization/forward passes,
autograd gradients, optimizer updates and short end-to-end training runs
numerically match. It requires an environment with both the original
dependencies (jax, flax, distrax, optax, gymnax, chex) and this repository's
requirements installed, plus the original repository checked out next to this
one:

``` bash
python tests/run_all.py --jax-repo /path/to/qdx_jolle_ag
```

## Notable changes vs. the upstream JAX repository

This repository is a derived work of [jolle-ag/qdx](https://github.com/jolle-ag/qdx)
(all credit for the method and original implementation belongs to the paper's
authors). It replaces the JAX/Flax/Distrax/Optax/Gymnax stack with PyTorch
while preserving the original numerics and public API:

* **Engine swap.** Every module that touched `jax`/`jax.numpy`/`flax`/
  `distrax`/`optax`/`gymnax`/`chex` was rewritten against `torch`:
  `qdx/torch_random.py`, `qdx/torch_nn.py`, `qdx/torch_env_base.py`,
  `qdx/simulators/*`, `qdx/envs/*`, `qdx/utils.py`, `qdx/make_train.py`,
  `qdx/code_finder.py`, `demo1.py`, `demo2.py`, and the standalone
  `icml2024-AI4science/` mirror (including its CSS-code environment and
  utilities).
* **Bit-exact PRNG.** `qdx/torch_random.py` reimplements JAX's
  threefry2x32 generator and Flax's SHA-1 parameter-key folding, so
  `PRNGKey`/`split`/`fold_in`/`uniform`/`normal`/`categorical`/`permutation`/
  `randint`/`choice` all reproduce the exact same random streams as the JAX
  original.
* **Autograd instead of hand-derived gradients.** The PPO loss's gradient
  is obtained with `torch.autograd` rather than `jax.value_and_grad` (and
  rather than a hand-written backward pass). `torch.minimum`/`maximum` share
  JAX's balanced 0.5/0.5 tie-breaking gradient, and clipping is expressed
  through them so gradient semantics match `jnp.clip` exactly.
* **Debuggability.** Every tensor — observations, trajectories, network
  parameters, gradients — is an eagerly evaluated `torch.Tensor` that can be
  printed or inspected in a debugger at any point; there is no tracing/jit
  step to work around.
* **GPU/accelerator support.** Setting `config["DEVICE"] = "cuda"` or
  `"mps"` moves the network and PPO update to an accelerator (environments
  stay on CPU, since their tableau updates are tiny integer ops that don't
  benefit from a GPU). Verified on Apple Silicon (MPS) and on an NVIDIA
  RTX A6000 (CUDA): correctness holds to ~1e-6/1e-7 vs. CPU, and speedups
  scale with network size — up to **~18x** for a 512-unit hidden layer with
  a 16k batch, and 1.4x end-to-end on a demo-scale training run (see
  `tests/bench_gpu.py`).
* **Circuit-diagram output.** `demo1.py`/`demo2.py` now rasterize the
  discovered-circuit diagram straight to PNG (via `resvg-py`, a pure pip
  dependency with no system libraries required) instead of writing raw SVG,
  which several image viewers render incorrectly.
* **Verification suite.** `tests/` contains a from-scratch JAX-vs-PyTorch
  comparison suite (PRNG, simulators, environments, network/loss/gradients/
  optimizer, CSS environment, and full end-to-end training runs) — 207
  checks, all passing, verified independently on both macOS (CPU/MPS) and
  Linux/CUDA.

## License

The code in this repository is released under the MIT License.

## Citation
``` bib
@article{olle_simultaneous_2024,
  title={Simultaneous Discovery of Quantum Error Correction Codes and Encoders with a Noise-Aware Reinforcement Learning Agent},
  author={Olle, Jan and Zen, Remmy and Puviani, Matteo and Marquardt, Florian},
  url = {https://www.nature.com/articles/s41534-024-00920-y},
  journal={npj Quantum Information 10, Article number: 126 (2024)},
  urldate = {2024-12-03},
  publisher = {npj Quantum Information},
  month = dec,
  year = {2024},
  note = {arXiv:2311.04750 [quant-ph]},
}
```
