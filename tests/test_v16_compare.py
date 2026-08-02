"""JAX(V1.6) vs PyTorch: exact GF(2) kernels, V1.6 reward/progress/termination,
v1.4 action canonicalization + pending-mask filtering, and symmetric GNN logits.

Baseline is the ``qdx-Size-Aware-GNN-V1-6-New-Reward`` branch snapshot
(default sibling directory ``qdx-JAX-V16``).
"""

import numpy as np
import torch

from compare_utils import parse_args, repo_on_path, Reporter, tree_max_diff, asnp

args = parse_args(__doc__)

import jax
import jax.numpy as jnp

r = Reporter("V1.6 exact GF(2) / reward / action space (JAX vs torch)")


def pauli_check_matrix(*paulis):
    n_qubits = len(paulis[0])
    matrix = np.zeros((len(paulis), 2 * n_qubits), dtype=np.uint8)
    for row, pauli in enumerate(paulis):
        for qubit, operator in enumerate(pauli):
            if operator in "XY":
                matrix[row, qubit] = 1
            if operator in "YZ":
                matrix[row, n_qubits + qubit] = 1
    return matrix


def to_np_tree(t):
    if isinstance(t, dict) or hasattr(t, "items"):
        return {k: to_np_tree(v) for k, v in t.items()}
    return asnp(t)


def to_torch_tree(t):
    if isinstance(t, dict) or hasattr(t, "items"):
        return {k: to_torch_tree(v) for k, v in t.items()}
    return torch.from_numpy(
        np.array(asnp(t), dtype=np.float32, copy=True)
    ).requires_grad_(True)


# ===================================================================== kernels
# Random-but-reproducible stabilizer/error batteries, checked field by field.
RNG = np.random.default_rng(20260719)
KERNEL_CASES = []
# (a) hand-written algebraic cases from the V1.6 unit tests
KERNEL_CASES.append(
    ("ZZI/IZZ vs ZIZ,ZII,XII",
     pauli_check_matrix("ZZI", "IZZ"),
     pauli_check_matrix("ZIZ", "ZII", "XII"),
     np.array([0.2, 0.3, 0.5], dtype=np.float32))
)
KERNEL_CASES.append(
    ("five-qubit code, weight-1 errors",
     pauli_check_matrix("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"),
     np.concatenate([
         pauli_check_matrix(*[f"{'I'*i}{p}{'I'*(4-i)}" for i in range(5)])
         for p in "XYZ"
     ]),
     (RNG.random(15) + 0.05).astype(np.float32))
)
# (b) randomized commuting stabilizer groups produced by real Clifford circuits
with repo_on_path(args.torch_repo):
    from qdx.simulators import TableauSimulator as TS_t
    from qdx.runtime_cache import build_exact_weight_error_operators

    for case_index in range(4):
        n = int(RNG.integers(3, 6))
        sim = TS_t(n)
        for _ in range(int(RNG.integers(3, 9))):
            if RNG.random() < 0.5:
                getattr(sim, RNG.choice(["h", "s", "sqrt_x"]))(int(RNG.integers(n)))
            else:
                a, b = RNG.choice(n, size=2, replace=False)
                getattr(sim, RNG.choice(["cx", "cz", "sqrt_xx"]))(int(a), int(b))
        tab = asnp(sim.current_tableau[0]).astype(np.uint8)
        stabs = tab[n + 1 :]
        errs = build_exact_weight_error_operators(n, 1 + int(RNG.integers(2)))
        probs = (RNG.random(errs.shape[0]) * 0.4 + 0.01).astype(np.float32)
        KERNEL_CASES.append((f"random circuit n={n} (#{case_index})", stabs, errs, probs))
        KERNEL_CASES.append(
            (f"random circuit n={n} (#{case_index}) [tableau]", tab, errs, probs)
        )

with repo_on_path(args.jax_repo):
    from qdx.gf2_distance import (
        jax_exact_gf2_kl,
        jax_tableau_kl,
        jax_softness_kl,
        gf2_rref as gf2_rref_j,
        gf2_row_space_mask as gf2_row_space_mask_j,
        symplectic_commutation_mask as commutation_j,
        verify_stabilizer_distance_gf2 as verify_j,
        stabilizer_check_matrix_from_gates as from_gates_j,
    )
    from qdx.runtime_cache import build_s_structure as build_s_structure_j

    jax_kernel = {}
    for name, matrix, errors, probs in KERNEL_CASES:
        if matrix.shape[0] == matrix.shape[1] and "[tableau]" in name:
            res = jax_tableau_kl(
                jnp.asarray(matrix), 1, jnp.asarray(errors), jnp.asarray(probs), 10.0
            )
        else:
            res = jax_exact_gf2_kl(
                jnp.asarray(matrix), jnp.asarray(errors), jnp.asarray(probs), 10.0
            )
        jax_kernel[name] = {
            "logical": np.asarray(res.logical_error_mask),
            "commutes": np.asarray(res.commutes_mask),
            "in_stab": np.asarray(res.in_stabilizer_mask),
            "count": int(res.error_count),
            "prob": float(res.logical_error_probability),
            "cost": float(res.error_cost),
            "reward": float(res.reward),
            "terminal": bool(res.terminal),
        }
    # per-weight statistics and softness cross-check on one case
    _, stabs5, errs5, probs5 = KERNEL_CASES[1]
    weights5 = np.ones(errs5.shape[0], dtype=np.int32)
    wv5 = np.asarray([1, 2], dtype=np.int32)
    byweight_j = jax_exact_gf2_kl(
        jnp.asarray(stabs5), jnp.asarray(errs5), jnp.asarray(probs5), 10.0,
        error_weights=jnp.asarray(weights5), weight_values=jnp.asarray(wv5))
    byweight_j = {
        "count_by_weight": np.asarray(byweight_j.error_count_by_weight),
        "total_by_weight": np.asarray(byweight_j.total_count_by_weight),
        "rate_by_weight": np.asarray(byweight_j.error_rate_by_weight),
    }
    soft_j = {
        s: int(jax_softness_kl(
            jnp.asarray(KERNEL_CASES[0][1]), jnp.asarray(KERNEL_CASES[0][2]),
            jnp.asarray(KERNEL_CASES[0][3]),
            jnp.asarray(build_s_structure_j(2, s)), 10.0).error_count)
        for s in (1, 2)
    }
    # host verifier
    verify_j_out = {}
    for label, cm, target, mw in [
        ("rep3", pauli_check_matrix("ZZI", "IZZ"), 2, None),
        ("five_qubit_target", pauli_check_matrix("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"), 3, None),
        ("five_qubit_exact", pauli_check_matrix("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"), 3, 3),
    ]:
        res = verify_j(cm, target_distance=target, max_weight=mw)
        verify_j_out[label] = res.to_dict()
    rref_j_reduced, rref_j_pivots = gf2_rref_j(
        np.asarray([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]], dtype=np.uint8))
    from_gates_j_out = np.asarray(from_gates_j(2, 1, [".h(1)"]))

with repo_on_path(args.torch_repo):
    from qdx.gf2_distance import (
        torch_exact_gf2_kl,
        torch_tableau_kl,
        torch_softness_kl,
        gf2_rref,
        gf2_row_space_mask,
        symplectic_commutation_mask,
        verify_stabilizer_distance_gf2,
        stabilizer_check_matrix_from_gates,
    )
    from qdx.runtime_cache import build_s_structure

    ok_fields = True
    detail = ""
    for name, matrix, errors, probs in KERNEL_CASES:
        if matrix.shape[0] == matrix.shape[1] and "[tableau]" in name:
            res = torch_tableau_kl(
                torch.from_numpy(np.ascontiguousarray(matrix)), 1,
                torch.from_numpy(np.ascontiguousarray(errors)),
                torch.from_numpy(probs), 10.0)
        else:
            res = torch_exact_gf2_kl(
                torch.from_numpy(np.ascontiguousarray(matrix)),
                torch.from_numpy(np.ascontiguousarray(errors)),
                torch.from_numpy(probs), 10.0)
        j = jax_kernel[name]
        same = (
            np.array_equal(asnp(res.logical_error_mask), j["logical"])
            and np.array_equal(asnp(res.commutes_mask), j["commutes"])
            and np.array_equal(asnp(res.in_stabilizer_mask), j["in_stab"])
            and int(res.error_count) == j["count"]
            and bool(res.terminal) == j["terminal"]
            and abs(float(res.error_cost) - j["cost"]) < 1e-5
            and abs(float(res.reward) - j["reward"]) < 1e-5
        )
        if not same:
            ok_fields = False
            detail = f"first mismatch: {name}"
            break
    r.check(f"exact KL kernels: all fields match over {len(KERNEL_CASES)} cases",
            ok_fields, detail)

    bw = torch_exact_gf2_kl(
        torch.from_numpy(np.ascontiguousarray(stabs5)),
        torch.from_numpy(np.ascontiguousarray(errs5)),
        torch.from_numpy(probs5), 10.0,
        error_weights=torch.from_numpy(weights5),
        weight_values=torch.from_numpy(wv5))
    r.check_value_equal("per-weight error counts",
                        asnp(bw.error_count_by_weight), byweight_j["count_by_weight"])
    r.check_value_equal("per-weight total counts",
                        asnp(bw.total_count_by_weight), byweight_j["total_by_weight"])
    r.check_close("per-weight error rates",
                  asnp(bw.error_rate_by_weight), byweight_j["rate_by_weight"],
                  rtol=1e-6, atol=1e-7)

    soft_t = {
        s: int(torch_softness_kl(
            torch.from_numpy(np.ascontiguousarray(KERNEL_CASES[0][1])),
            torch.from_numpy(np.ascontiguousarray(KERNEL_CASES[0][2])),
            torch.from_numpy(KERNEL_CASES[0][3]),
            torch.from_numpy(np.ascontiguousarray(build_s_structure(2, s))),
            10.0).error_count)
        for s in (1, 2)
    }
    r.check("softness KL counts match (s=1,2)", soft_t == soft_j,
            f"torch {soft_t} vs jax {soft_j}")

    rref_t_reduced, rref_t_pivots = gf2_rref(
        np.asarray([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]], dtype=np.uint8))
    r.check_value_equal("host gf2_rref basis", rref_t_reduced, rref_j_reduced)
    r.check("host gf2_rref pivots", rref_t_pivots == rref_j_pivots,
            f"{rref_t_pivots} vs {rref_j_pivots}")

    ok_verify = True
    for label, cm, target, mw in [
        ("rep3", pauli_check_matrix("ZZI", "IZZ"), 2, None),
        ("five_qubit_target", pauli_check_matrix("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"), 3, None),
        ("five_qubit_exact", pauli_check_matrix("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"), 3, 3),
    ]:
        got = verify_stabilizer_distance_gf2(
            cm, target_distance=target, max_weight=mw).to_dict()
        if got != verify_j_out[label]:
            ok_verify = False
            break
    r.check("host verify_stabilizer_distance_gf2 dicts identical", ok_verify)
    r.check_value_equal("stabilizer_check_matrix_from_gates",
                        stabilizer_check_matrix_from_gates(2, 1, [".h(1)"]),
                        from_gates_j_out)

# ============================================================ env V1.6 rewards
N, K, D, MAX_STEPS = 5, 1, 3, 12
GATES = ["cx", "h", "s", "sqrt_x", "cz", "sqrt_xx"]
ACTION_SEQ = [0, 5, 11, 3, 27, 8, 14, 2, 19, 6]

env_jax_out = {}
with repo_on_path(args.jax_repo):
    from qdx.envs.code_discovery import CodeDiscovery as CD_j
    from qdx.simulators.clifford_gates import CliffordGates as CG_j

    g = CG_j(N)
    gate_set = [getattr(g, name) for name in GATES]
    graph = [(i, j) for i in range(N) for j in range(N) if i != j]
    for method in ("existing", "gf2", "gf2_tableau"):
        env = CD_j(N, K, D, gate_set, graph=graph, max_steps=MAX_STEPS,
                   lbda=10.0, pI=0.9, softness=1, kl_method=method)
        _, state = env.reset(jax.random.PRNGKey(0), None)
        rec = {
            "num_actions": int(env.num_actions),
            "action_string_stim": list(env.action_string_stim),
            "P0": float(state.progress_score),
            "success0": bool(state.success),
            "rewards": [], "physical": [], "progress": [], "delta": [],
            "distance": [], "violations": [], "bonus": [], "done": [],
            "pending_sum": [], "tableau_sum": [],
        }
        for step, action in enumerate(ACTION_SEQ):
            _, state, reward, done, info = env.step_env(
                jax.random.PRNGKey(100 + step), state,
                jnp.asarray(action % env.num_actions), env.default_params)
            rec["rewards"].append(float(reward))
            rec["physical"].append(float(info["physical_reward"]))
            rec["progress"].append(float(info["progress_score"]))
            rec["delta"].append(float(info["progress_delta"]))
            rec["distance"].append(int(info["distance"]))
            rec["violations"].append(int(info["violations_at_distance"]))
            rec["bonus"].append(float(info["success_bonus"]))
            rec["done"].append(bool(done))
            rec["pending_sum"].append(int(np.asarray(state.pending_action_mask).sum()))
            rec["tableau_sum"].append(int(np.asarray(state.tableau).sum()))
            if done:
                break
        env_jax_out[method] = rec
    # commutation/cancellation relation tables
    env_ref = CD_j(N, K, D, gate_set, graph=graph, max_steps=MAX_STEPS,
                   lbda=10.0, pI=0.9, softness=1, kl_method="gf2_tableau")
    commute_j = np.asarray(env_ref._commute_table)
    cancel_j = np.asarray(env_ref._cancel_table)
    actions_j = np.asarray(env_ref.actions)

with repo_on_path(args.torch_repo):
    from qdx.envs.code_discovery import CodeDiscovery as CD_t
    from qdx.simulators.clifford_gates import CliffordGates as CG_t
    from qdx import torch_random as tr

    g = CG_t(N)
    gate_set = [getattr(g, name) for name in GATES]
    graph = [(i, j) for i in range(N) for j in range(N) if i != j]
    for method in ("existing", "gf2", "gf2_tableau"):
        env = CD_t(N, K, D, gate_set, graph=graph, max_steps=MAX_STEPS,
                   lbda=10.0, pI=0.9, softness=1, kl_method=method)
        _, state = env.reset(tr.PRNGKey(0), None)
        j = env_jax_out[method]
        r.check(f"[{method}] action space size + strings",
                int(env.num_actions) == j["num_actions"]
                and list(env.action_string_stim) == j["action_string_stim"],
                f"torch {int(env.num_actions)} vs jax {j['num_actions']}")
        r.check_close(f"[{method}] initial progress P0",
                      float(state.progress_score), j["P0"], rtol=1e-6, atol=1e-6)
        r.check(f"[{method}] initial success flag",
                bool(state.success) == j["success0"])

        rewards, physical, progress, delta = [], [], [], []
        distance, violations, bonus, dones = [], [], [], []
        pending_sum, tableau_sum = [], []
        for step, action in enumerate(ACTION_SEQ):
            _, state, reward, done, info = env.step_env(
                tr.PRNGKey(100 + step), state,
                action % env.num_actions, env.default_params)
            rewards.append(float(reward))
            physical.append(float(info["physical_reward"]))
            progress.append(float(info["progress_score"]))
            delta.append(float(info["progress_delta"]))
            distance.append(int(info["distance"]))
            violations.append(int(info["violations_at_distance"]))
            bonus.append(float(info["success_bonus"]))
            dones.append(bool(done))
            pending_sum.append(int(state.pending_action_mask.sum()))
            tableau_sum.append(int(state.tableau.sum()))
            if done:
                break

        r.check_close(f"[{method}] total reward per step", rewards, j["rewards"],
                      rtol=1e-5, atol=1e-5)
        r.check_close(f"[{method}] physical reward component", physical,
                      j["physical"], rtol=1e-5, atol=1e-5)
        r.check_close(f"[{method}] progress score", progress, j["progress"],
                      rtol=1e-6, atol=1e-6)
        r.check_close(f"[{method}] progress delta", delta, j["delta"],
                      rtol=1e-6, atol=1e-6)
        r.check_value_equal(f"[{method}] frontier distance", distance, j["distance"])
        r.check_value_equal(f"[{method}] violations at frontier", violations,
                            j["violations"])
        r.check_value_equal(f"[{method}] success bonus", bonus, j["bonus"])
        r.check(f"[{method}] done flags", dones == j["done"],
                f"torch {dones} vs jax {j['done']}")
        r.check_value_equal(f"[{method}] pending-mask popcount per step",
                            pending_sum, j["pending_sum"])
        r.check_value_equal(f"[{method}] tableau checksum per step",
                            tableau_sum, j["tableau_sum"])

    env_ref_t = CD_t(N, K, D, gate_set, graph=graph, max_steps=MAX_STEPS,
                     lbda=10.0, pI=0.9, softness=1, kl_method="gf2_tableau")
    r.check_value_equal("action matrices", asnp(env_ref_t.actions), actions_j)
    r.check_value_equal("commutation table C_ij", asnp(env_ref_t._commute_table),
                        commute_j)
    r.check_value_equal("cancellation table R_ij", asnp(env_ref_t._cancel_table),
                        cancel_j)

# ============================== graph observation + symmetric GNN logits
PADDING = {"n_max": 6, "stabilizers_max": 5, "hardware_edges_max": 30}
HP = dict(hidden_dim=32, gate_dim=8, num_gnn_layers=2)
GRAPH_ACTIONS = [0, 7, 3, 12, 5, 9]

with repo_on_path(args.jax_repo):
    from qdx.envs.graph_code_discovery import GraphCodeDiscovery as GCD_j
    from qdx.gnn.observation import GraphPadding as GP_j
    from qdx.gnn.model import GNNQDXActorCritic as GNN_j
    from qdx.simulators.clifford_gates import CliffordGates as CG_j2

    g = CG_j2(N)
    gate_set = [getattr(g, name) for name in GATES]
    env = GCD_j(N, K, D, gate_set, graph=graph, max_steps=MAX_STEPS, lbda=10.0,
                pI=0.9, softness=1, kl_method="gf2_tableau",
                graph_padding=GP_j(**PADDING))
    obs, state = env.reset(jax.random.PRNGKey(0), None)
    obs_seq_j = [obs]
    for step, action in enumerate(GRAPH_ACTIONS):
        obs, state, _, _, _ = env.step(
            jax.random.PRNGKey(200 + step), state,
            jnp.asarray(action % env.num_actions), None)
        obs_seq_j.append(obs)
    fields = [f for f in obs_seq_j[0].__dataclass_fields__]
    obs_np_j = [{f: np.asarray(getattr(o, f)) for f in fields} for o in obs_seq_j]

    net_j = GNN_j(num_gate_types=len(GATES), activation="tanh", **HP)
    params_j = net_j.init(jax.random.PRNGKey(11), env.graph_observation_template())
    params_j_np = to_np_tree(params_j)
    batch_j = jax.tree_util.tree_map(lambda *l: jnp.stack(l), *obs_seq_j)
    pi_j, v_j = net_j.apply(params_j, batch_j)
    logits_j = np.asarray(pi_j.logits)
    values_j = np.asarray(v_j)
    entropy_j = np.asarray(pi_j.entropy())
    is_sym_j = np.asarray(obs_seq_j[0].action_is_symmetric)

with repo_on_path(args.torch_repo):
    from qdx.envs.graph_code_discovery import GraphCodeDiscovery as GCD_t
    from qdx.gnn.observation import GraphPadding as GP_t, obs_stack
    from qdx.gnn.model import GNNQDXActorCritic as GNN_t
    from qdx.simulators.clifford_gates import CliffordGates as CG_t2
    from qdx import torch_random as tr2

    g = CG_t2(N)
    gate_set = [getattr(g, name) for name in GATES]
    env_t = GCD_t(N, K, D, gate_set, graph=graph, max_steps=MAX_STEPS, lbda=10.0,
                  pI=0.9, softness=1, kl_method="gf2_tableau",
                  graph_padding=GP_t(**PADDING))
    obs_t, state_t = env_t.reset(tr2.PRNGKey(0), None)
    obs_seq_t = [obs_t]
    for step, action in enumerate(GRAPH_ACTIONS):
        obs_t, state_t, _, _, _ = env_t.step(
            tr2.PRNGKey(200 + step), state_t, action % env_t.num_actions, None)
        obs_seq_t.append(obs_t)

    r.check("graph obs: field names/order match JAX dataclass",
            list(obs_seq_t[0]._fields) == fields,
            f"torch {list(obs_seq_t[0]._fields)[:4]}... vs jax {fields[:4]}...")

    int_fields = ["senders", "receivers", "relation_ids", "node_mask", "edge_mask",
                  "qubit_mask", "stabilizer_mask", "action_types", "action_gate_ids",
                  "action_is_symmetric", "action_first", "action_second",
                  "action_mask", "action_env_indices"]
    float_fields = ["node_features", "edge_features", "global_features"]
    ok_int, ok_float, maxdiff = True, True, 0.0
    for ot, oj in zip(obs_seq_t, obs_np_j):
        for name in int_fields:
            if not np.array_equal(asnp(getattr(ot, name)), oj[name]):
                ok_int = False
        for name in float_fields:
            d = float(np.max(np.abs(asnp(getattr(ot, name)).astype(np.float64)
                                    - oj[name].astype(np.float64))))
            maxdiff = max(maxdiff, d)
            if d > 1e-5:
                ok_float = False
    r.check("graph obs: integer/mask fields exact over rollout (incl. dynamic mask)",
            ok_int)
    r.check("graph obs: float features close over rollout", ok_float,
            f"max abs diff {maxdiff:.3g}")
    r.check_value_equal("graph obs: action_is_symmetric",
                        asnp(obs_seq_t[0].action_is_symmetric), is_sym_j)

    net_t = GNN_t(num_gate_types=len(GATES), activation="tanh", **HP)
    params_t_init = net_t.init(tr2.PRNGKey(11), env_t.graph_observation_template())
    d = tree_max_diff(params_j_np, to_np_tree(params_t_init))
    r.check("GNN init params match flax defaults", d < 5e-6, f"max abs diff {d:.3g}")

    params_t = to_torch_tree(params_j_np)
    batch_t = obs_stack(obs_seq_t)
    with torch.no_grad():
        pi_t, v_t = net_t.apply(params_t, batch_t)
    mask = asnp(batch_t.action_mask).astype(bool)
    r.check_close("GNN forward: valid logits (symmetric averaging)",
                  asnp(pi_t.logits)[mask], logits_j[mask], rtol=1e-4, atol=1e-5)
    r.check_close("GNN forward: values", asnp(v_t), values_j, rtol=1e-4, atol=1e-5)
    r.check_close("GNN forward: entropy", asnp(pi_t.entropy()), entropy_j,
                  rtol=1e-4, atol=1e-5)

    # order invariance of symmetric two-qubit logits (V1.4 property)
    single = obs_seq_t[0]
    sym_idx = int(np.argmax(asnp(single.action_is_symmetric)))
    swapped_first = single.action_first.clone()
    swapped_second = single.action_second.clone()
    swapped_first[sym_idx], swapped_second[sym_idx] = (
        single.action_second[sym_idx].clone(), single.action_first[sym_idx].clone())
    swapped = single._replace(action_first=swapped_first, action_second=swapped_second)
    with torch.no_grad():
        base_logits = net_t.apply(params_t, single)[0].logits
        swap_logits = net_t.apply(params_t, swapped)[0].logits
    r.check_close("symmetric action logit is order invariant",
                  float(base_logits[sym_idx]), float(swap_logits[sym_idx]),
                  rtol=1e-6, atol=1e-6)

# ================================================ validation distance verifier
GATE_CIRCUIT = ['.h(0)', '.cx(0, 1)', '.cx(0, 2)', '.h(3)', '.cx(3, 4)', '.cx(1, 4)']
with repo_on_path(args.jax_repo):
    from qdx.utils import distance_error_stats_up_to_target as dist_j
    dist_out_j = {
        m: dist_j(N, K, GATE_CIRCUIT, D, softness=None, kl_method=m)
        for m in ("existing", "gf2", "gf2_tableau")
    }
with repo_on_path(args.torch_repo):
    from qdx.utils import distance_error_stats_up_to_target as dist_t
    for method in ("existing", "gf2", "gf2_tableau"):
        got = dist_t(N, K, GATE_CIRCUIT, D, softness=None, kl_method=method)
        r.check(f"validation distance stats [{method}]", got == dist_out_j[method],
                f"torch {got[0]} vs jax {dist_out_j[method][0]}")

r.finish()
