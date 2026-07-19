"""JAX vs PyTorch: tableau simulators and Clifford gate matrices (bit-exact)."""

import numpy as np

from compare_utils import parse_args, repo_on_path, Reporter

args = parse_args(__doc__)

r = Reporter("Simulators (tableaus and gate matrices)")

# ---------------- qdx simulators ----------------
with repo_on_path(args.jax_repo):
    from qdx.simulators import TableauSimulator as TS_jax
    from qdx.simulators import CliffordGates as CG_jax
with repo_on_path(args.torch_repo):
    from qdx.simulators import TableauSimulator as TS_torch
    from qdx.simulators import CliffordGates as CG_torch

n = 6
gj, gn = CG_jax(n), CG_torch(n)
for gate in ["h", "s", "sqrt_x"]:
    for q in range(n):
        r_ok = np.array_equal(np.asarray(getattr(gj, gate)(q)),
                              getattr(gn, gate)(q))
        if not r_ok:
            r.check(f"CliffordGates.{gate}({q})", False)
            break
else:
    r.check("CliffordGates 1-qubit gates (h, s, sqrt_x) all qubits", True)

ok2 = True
for gate in ["cx", "cz", "sqrt_xx"]:
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            if not np.array_equal(np.asarray(getattr(gj, gate)(a, b)),
                                  getattr(gn, gate)(a, b)):
                ok2 = False
r.check("CliffordGates 2-qubit gates (cx, cz, sqrt_xx) all pairs", ok2)

# random circuit on both simulators
rng = np.random.default_rng(0)
tj, tn = TS_jax(n, batch_size=2), TS_torch(n, batch_size=2)
r.check_exact("initial tableau", tn.current_tableau, np.asarray(tj.current_tableau))
ops1 = ["h", "s", "sqrt_x"]
ops2 = ["cx", "cz", "sqrt_xx"]
for _ in range(60):
    if rng.random() < 0.5:
        op = ops1[rng.integers(3)]
        q = int(rng.integers(n))
        getattr(tj, op)(q)
        getattr(tn, op)(q)
    else:
        op = ops2[rng.integers(3)]
        a, b = rng.choice(n, size=2, replace=False)
        getattr(tj, op)(int(a), int(b))
        getattr(tn, op)(int(a), int(b))
r.check_exact("tableau after random 60-gate circuit", tn.current_tableau,
              np.asarray(tj.current_tableau))
r.check("simulator iterator", all(
    np.array_equal(np.asarray(a), b) for a, b in zip(tj, tn)))

# ---------------- icml CSS simulators ----------------
with repo_on_path(args.jax_repo, "icml2024-AI4science", icml=True):
    from simulators.tableau_simulator_css import TableauSimulatorCSS as CSS_jax
    from simulators.clifford_gates_css import CliffordGatesCSS as CGS_jax
with repo_on_path(args.torch_repo, "icml2024-AI4science", icml=True):
    from simulators.tableau_simulator_css import TableauSimulatorCSS as CSS_torch
    from simulators.clifford_gates_css import CliffordGatesCSS as CGS_torch

n, k = 7, 1
init_H = [1, 3, 5]
for bell in [False, True]:
    cj = CSS_jax(n, k, init_H, bell=bell)
    cn = CSS_torch(n, k, init_H, bell=bell)
    okX = np.array_equal(np.asarray(cj.H_X), cn.H_X)
    okZ = np.array_equal(np.asarray(cj.H_Z), cn.H_Z)
    r.check(f"CSS init (bell={bell}) H_X/H_Z", okX and okZ)

cj = CSS_jax(n, k, init_H)
cn = CSS_torch(n, k, init_H)
rng = np.random.default_rng(3)
for _ in range(40):
    a, b = rng.choice(n, size=2, replace=False)
    cj.cx(int(a), int(b))
    cn.cx(int(a), int(b))
r.check("CSS tableau after random CX circuit",
        np.array_equal(np.asarray(cj.H_X), cn.H_X)
        and np.array_equal(np.asarray(cj.H_Z), cn.H_Z))

okg = True
gj = CGS_jax(n)
gn = CGS_torch(n)
for a in range(n):
    for b in range(n):
        if a != b and not np.array_equal(np.asarray(gj.cx(a, b)), gn.cx(a, b)):
            okg = False
r.check("CliffordGatesCSS.cx all pairs", okg)

r.finish()
