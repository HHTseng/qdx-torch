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


# Example of [[7,1,3]] code discovery
# This corresponds to Fig.10 in the paper
config = {
    "DEVICE": "cuda",
    "ENV_TYPE": "STANDARD",  # Possibilities: "STANDARD", "MAX", "DELTA", "NOISE-AWARE"
    "N": 7,
    "K": 1,
    "D": 3,
    "MAX_STEPS": 20,
    "WHICH_GATES": ["cx", "h"],
    "GRAPH": "All-to-All",
    "SOFTNESS": 1,
    "P_I": 0.9,
    "LAMBDA": 10,
    "SEED": 42,
    "LR": 1e-3,
    "NUM_ENVS": 16,
    "NUM_STEPS": 20,
    "TOTAL_TIMESTEPS": 2e6,
    "UPDATE_EPOCHS": 3,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.02,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.25,
    "ACTIVATION": "relu",
    "HIDDEN_DIM": 32,
    "ANNEAL_LR": True,
    "NUM_AGENTS": 4,
    "COMPUTE_METRICS": True,
}

finder = CodeFinder(config)
# Training should take around 20sec
params, metrics = finder.train()

returns = metrics["returned_episode_returns"]
lengths = metrics["returned_episode_lengths"]

x = np.linspace(0, config["NUM_EPOCHS"], len(returns[0]))
plot_metric_curves(x, returns, "Number of epochs", "Return", "standard_returns.png")
plot_metric_curves(x, lengths, "Number of epochs", "Circuit size", "standard_lengths.png")

data = finder.evaluate()
# We print the results from the first agent, for example
print(data[0])

# Finally, let's visualize the discovered circuits
circ = build_circuit_from_gates(data[0]["gates"])
svg = circ.diagram("timeline-svg")
save_svg_as_png(svg, "circuit_standard.png")

plt.show()