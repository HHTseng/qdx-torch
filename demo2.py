#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "demo_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Try common relative locations for the qdx package.
for candidate in (SCRIPT_DIR.parent, SCRIPT_DIR, SCRIPT_DIR.parent.parent):
    if (candidate / "qdx").exists():
        sys.path.insert(0, str(candidate))
        break
else:
    sys.path.append(str((SCRIPT_DIR / "..").resolve()))

from qdx.code_finder import CodeFinder


def save_current_figure(filename: str) -> None:
    fig = plt.gcf()
    out_file = OUTPUT_DIR / filename
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {out_file}")
    plt.show(block=False)


def save_svg_as_png(svg_obj, filename: str) -> None:
    """Rasterize a stim circuit-diagram SVG to PNG (SVGs render poorly in
    plain image viewers/Quick Look; PNG opens everywhere)."""
    import webbrowser

    import resvg_py

    out_file = OUTPUT_DIR / filename
    png_bytes = resvg_py.svg_to_bytes(svg_string=str(svg_obj))
    with open(out_file, "wb") as f:
        f.write(bytes(png_bytes))
    print(f"Saved PNG: {out_file}")
    webbrowser.open(out_file.as_uri())


def build_circuit_from_gates(gates):
    import stim

    actions = []
    for g in gates:
        gate_name = g.split("(")[0].split(".")[1]
        qubit_ids = g.split("(")[1].split(")")[0]
        instruction = '.append("%s", [%s])' % (gate_name, qubit_ids)
        actions.append(instruction)

    circ = stim.Circuit()
    for action in actions:
        eval('circ%s' % action)
    return circ


def plot_metric_curves(x, curves, xlabel: str, ylabel: str, filename: str) -> None:
    plt.figure()
    for curve in curves:
        plt.plot(x, curve)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_current_figure(filename)


# Example of [[6,1]] code discovery with the noise-aware meta agent
# We choose d=4 since we know it will not succeed. We therefore force it to correct the most dangerous errors only
config = {
    "ENV_TYPE": "NOISE-AWARE",  # Possibilities: "STANDARD", "MAX", "DELTA", "NOISE-AWARE"
    "N": 6,
    "K": 1,
    "D": 4,
    "MAX_STEPS": 30,
    "WHICH_GATES": ["cx", "h", "s"],
    "GRAPH": "All-to-All",
    "SOFTNESS": 1,
    "P_I": 0.9,
    "LAMBDA": 1,
    "SEED": 42,
    "LR": 5e-4,
    "NUM_ENVS": 64,
    "NUM_STEPS": 8,
    "TOTAL_TIMESTEPS": 4e6,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 8,
    "GAMMA": .99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.1,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.05,
    "ACTIVATION": "relu",
    "HIDDEN_DIM": 200,
    "ANNEAL_LR": True,
    "NUM_AGENTS": 4,
    "COMPUTE_METRICS": True,
}

finder = CodeFinder(config)
# Training is now more costly. It should train in approx 60 sec.
params, metrics = finder.train()

returns = metrics["returned_episode_returns"]
lengths = metrics["returned_episode_lengths"]

x = np.linspace(0, config["NUM_EPOCHS"], len(returns[0]))
plot_metric_curves(x, returns, "Number of epochs", "Return", "noiseaware_returns.png")
plot_metric_curves(x, lengths, "Number of epochs", "Circuit size", "noiseaware_lengths.png")

# Evaluation is now more costly because each agent is evaluated at 16 different values of cZ
# TODO: Accelerate evaluation with vmap
data = finder.evaluate()

# Technically, the effective distance d_eff should be an integer.
# Here it should be read as the smallest undetected effective weight
print(data[0])

# Finally, let's visualize the discovered circuits
# There might be a bunch of S gates applied at the very end. These can be ignored!
circ = build_circuit_from_gates(data[0]["gates"])
svg = circ.diagram("timeline-svg")
save_svg_as_png(svg, "circuit_noiseaware.png")

plt.show()
