"""JAX vs PyTorch: the four qdx environments.

Uses identical PRNG key streams and action sequences on both sides and
compares static tensors, observations, states, rewards, dones and infos.
Integer/boolean quantities must match exactly; float32 quantities must match
within round-off tolerances.
"""

import numpy as np

from compare_utils import parse_args, repo_on_path, Reporter

args = parse_args(__doc__)

r = Reporter("Environments (CodeDiscovery / Meta / Delta / Max)")

N, K, D = 5, 1, 3
MAX_STEPS = 8
N_ROLLOUT = 3 * MAX_STEPS

# Deterministic action sequence and keys shared by both sides
with repo_on_path(args.torch_repo):
    from qdx import torch_random as nr

master = nr.PRNGKey(2024)
step_keys = nr.split(nr.fold_in(master, 1), N_ROLLOUT)
reset_key = nr.fold_in(master, 2)


def rollout_jax(env_ctor_args, env_kwargs, gates_names):
    import jax
    import jax.numpy as jnp
    from qdx.simulators.clifford_gates import CliffordGates

    g = CliffordGates(N)
    gate_set = [getattr(g, name) for name in gates_names]
    env = env_ctor_args(gate_set, env_kwargs)
    A = env.action_space().n

    static = {
        "actions": np.asarray(env.actions),
        "Omega": np.asarray(env.Omega),
        "E_mu": np.asarray(env.E_mu),
        "S_struct": np.asarray(env.S_struct),
        "action_string": list(env.action_string),
        "action_string_stim": list(env.action_string_stim),
        "num_actions": A,
        "obs_shape": tuple(env.obs_shape),
    }
    if hasattr(env, "p_mu"):
        static["p_mu"] = np.asarray(env.p_mu)

    actions = [int(x) for x in np.asarray(
        nr.randint(nr.fold_in(master, 7), (N_ROLLOUT,), 0, A))]

    obs, state = env.reset(jnp.asarray(reset_key), None)
    traj = {"obs": [np.asarray(obs)], "reward": [], "done": [],
            "discount": [], "tableau": [], "extra": []}
    for t in range(N_ROLLOUT):
        obs, state, rew, done, info = env.step(
            jnp.asarray(step_keys[t]), state, actions[t], None)
        traj["obs"].append(np.asarray(obs))
        traj["reward"].append(float(rew))
        traj["done"].append(bool(done))
        traj["discount"].append(float(info["discount"]))
        traj["tableau"].append(np.asarray(state.tableau))
        extra = {}
        if hasattr(state, "cZ"):
            extra["cZ"] = float(state.cZ)
            extra["p_mu"] = np.asarray(state.p_mu)
        if hasattr(state, "x_t"):
            extra["x_t"] = float(state.x_t)
            extra["current_rew"] = float(state.current_rew)
            extra["init_rew"] = float(state.init_rew)
        if "max_reward" in info:
            extra["max_reward"] = float(info["max_reward"])
        traj["extra"].append(extra)
    return static, actions, traj


def rollout_torch(env_ctor_args, env_kwargs, gates_names):
    from qdx.simulators.clifford_gates import CliffordGates

    g = CliffordGates(N)
    gate_set = [getattr(g, name) for name in gates_names]
    env = env_ctor_args(gate_set, env_kwargs)
    A = env.action_space().n

    static = {
        "actions": np.asarray(env.actions),
        "Omega": np.asarray(env.Omega),
        "E_mu": np.asarray(env.E_mu),
        "S_struct": np.asarray(env.S_struct),
        "action_string": list(env.action_string),
        "action_string_stim": list(env.action_string_stim),
        "num_actions": A,
        "obs_shape": tuple(env.obs_shape),
    }
    if hasattr(env, "p_mu"):
        static["p_mu"] = np.asarray(env.p_mu)

    actions = [int(x) for x in nr.randint(nr.fold_in(master, 7), (N_ROLLOUT,), 0, A)]

    obs, state = env.reset(reset_key, None)
    traj = {"obs": [np.asarray(obs)], "reward": [], "done": [],
            "discount": [], "tableau": [], "extra": []}
    for t in range(N_ROLLOUT):
        obs, state, rew, done, info = env.step(step_keys[t], state, actions[t], None)
        traj["obs"].append(np.asarray(obs))
        traj["reward"].append(float(rew))
        traj["done"].append(bool(done))
        traj["discount"].append(float(info["discount"]))
        traj["tableau"].append(np.asarray(state.tableau))
        extra = {}
        if hasattr(state, "cZ"):
            extra["cZ"] = float(state.cZ)
            extra["p_mu"] = np.asarray(state.p_mu)
        if hasattr(state, "x_t"):
            extra["x_t"] = float(state.x_t)
            extra["current_rew"] = float(state.current_rew)
            extra["init_rew"] = float(state.init_rew)
        if "max_reward" in info:
            extra["max_reward"] = float(info["max_reward"])
        traj["extra"].append(extra)
    return static, actions, traj


def compare(env_name, sj, aj, tj, sn, an, tn, obs_exact=True):
    r.check_value_equal(f"{env_name}: action tensor", sn["actions"], sj["actions"])
    r.check_value_equal(f"{env_name}: Omega", sn["Omega"], sj["Omega"])
    r.check_value_equal(f"{env_name}: E_mu", sn["E_mu"], sj["E_mu"])
    r.check_value_equal(f"{env_name}: S_struct", sn["S_struct"], sj["S_struct"])
    if "p_mu" in sj:
        r.check_close(f"{env_name}: p_mu", sn["p_mu"], sj["p_mu"], rtol=1e-6, atol=1e-7)
    r.check(f"{env_name}: action strings", sn["action_string"] == sj["action_string"]
            and sn["action_string_stim"] == sj["action_string_stim"])
    r.check(f"{env_name}: num_actions/obs_shape",
            sn["num_actions"] == sj["num_actions"] and sn["obs_shape"] == sj["obs_shape"])
    r.check(f"{env_name}: identical action sequence used", an == aj)

    if obs_exact:
        r.check("%s: observations (exact)" % env_name,
                all(np.array_equal(a, b) for a, b in zip(tn["obs"], tj["obs"])))
    else:
        ok = all(a.shape == b.shape and np.allclose(a, b, rtol=1e-6, atol=1e-6)
                 for a, b in zip(tn["obs"], tj["obs"]))
        r.check("%s: observations (float tolerance)" % env_name, ok)
    r.check(f"{env_name}: tableaus (exact)",
            all(np.array_equal(a, b) for a, b in zip(tn["tableau"], tj["tableau"])))
    r.check(f"{env_name}: dones (exact)", tn["done"] == tj["done"],
            f"torch {sum(tn['done'])} vs jax {sum(tj['done'])} resets")
    r.check(f"{env_name}: discounts (exact)", tn["discount"] == tj["discount"])
    r.check_close(f"{env_name}: rewards", tn["reward"], tj["reward"], rtol=1e-5, atol=1e-5)
    for key in tj["extra"][0].keys():
        va = [e[key] for e in tn["extra"]]
        vb = [e[key] for e in tj["extra"]]
        if key == "p_mu":
            ok = all(np.allclose(a, b, rtol=1e-5, atol=1e-7) for a, b in zip(va, vb))
            r.check(f"{env_name}: state p_mu", ok)
        else:
            r.check_close(f"{env_name}: {key}", va, vb, rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------
# STANDARD
with repo_on_path(args.jax_repo):
    from qdx.envs.code_discovery import CodeDiscovery as CD
    sj, aj, tj = rollout_jax(
        lambda gs, kw: CD(N, K, D, gs, max_steps=MAX_STEPS, lbda=10, pI=0.9, softness=1),
        None, ["cx", "h"])
with repo_on_path(args.torch_repo):
    from qdx.envs.code_discovery import CodeDiscovery as CDt
    sn, an, tn = rollout_torch(
        lambda gs, kw: CDt(N, K, D, gs, max_steps=MAX_STEPS, lbda=10, pI=0.9, softness=1),
        None, ["cx", "h"])
compare("STANDARD", sj, aj, tj, sn, an, tn, obs_exact=True)

# ----------------------------------------------------------------------
# NOISE-AWARE (random cZ)
with repo_on_path(args.jax_repo):
    from qdx.envs.meta_code_discovery import MetaCodeDiscovery as MC
    sj, aj, tj = rollout_jax(
        lambda gs, kw: MC(N, K, D, gs, max_steps=MAX_STEPS, lbda=1, pI=0.9, softness=1),
        None, ["cx", "h", "s"])
with repo_on_path(args.torch_repo):
    from qdx.envs.meta_code_discovery import MetaCodeDiscovery as MCt
    sn, an, tn = rollout_torch(
        lambda gs, kw: MCt(N, K, D, gs, max_steps=MAX_STEPS, lbda=1, pI=0.9, softness=1),
        None, ["cx", "h", "s"])
compare("NOISE-AWARE", sj, aj, tj, sn, an, tn, obs_exact=False)

# ----------------------------------------------------------------------
# NOISE-AWARE (fixed cZ)
with repo_on_path(args.jax_repo):
    from qdx.envs.meta_code_discovery import MetaCodeDiscovery as MC
    sj, aj, tj = rollout_jax(
        lambda gs, kw: MC(N, K, D, gs, max_steps=MAX_STEPS, lbda=1, pI=0.9, softness=1,
                          random_cZ=False, cZ=1.3, pX=0.0419095),
        None, ["cx", "h", "s"])
with repo_on_path(args.torch_repo):
    from qdx.envs.meta_code_discovery import MetaCodeDiscovery as MCt
    sn, an, tn = rollout_torch(
        lambda gs, kw: MCt(N, K, D, gs, max_steps=MAX_STEPS, lbda=1, pI=0.9, softness=1,
                           random_cZ=False, cZ=1.3, pX=0.0419095),
        None, ["cx", "h", "s"])
compare("NOISE-AWARE-fixed-cZ", sj, aj, tj, sn, an, tn, obs_exact=False)

# ----------------------------------------------------------------------
# DELTA
with repo_on_path(args.jax_repo):
    from qdx.envs.delta_code_discovery import DeltaCodeDiscovery as DC
    sj, aj, tj = rollout_jax(
        lambda gs, kw: DC(N, K, D, gs, max_steps=MAX_STEPS, lbda=10, pI=0.9, softness=1),
        None, ["cx", "h"])
with repo_on_path(args.torch_repo):
    from qdx.envs.delta_code_discovery import DeltaCodeDiscovery as DCt
    sn, an, tn = rollout_torch(
        lambda gs, kw: DCt(N, K, D, gs, max_steps=MAX_STEPS, lbda=10, pI=0.9, softness=1),
        None, ["cx", "h"])
compare("DELTA", sj, aj, tj, sn, an, tn, obs_exact=True)

# ----------------------------------------------------------------------
# MAX
with repo_on_path(args.jax_repo):
    from qdx.envs.max_code_discovery import MaxCodeDiscovery as XC
    sj, aj, tj = rollout_jax(
        lambda gs, kw: XC(N, K, D, gs, max_steps=MAX_STEPS, lbda=10, pI=0.9, softness=1),
        None, ["cx", "h"])
with repo_on_path(args.torch_repo):
    from qdx.envs.max_code_discovery import MaxCodeDiscovery as XCt
    sn, an, tn = rollout_torch(
        lambda gs, kw: XCt(N, K, D, gs, max_steps=MAX_STEPS, lbda=10, pI=0.9, softness=1),
        None, ["cx", "h"])
compare("MAX", sj, aj, tj, sn, an, tn, obs_exact=False)

# ----------------------------------------------------------------------
# Utils (evaluation helpers)
gates_seq = ['.h(0)', '.cx(0, 1)', '.cx(0, 2)', '.h(3)', '.cx(3, 4)', '.cx(1, 4)']
with repo_on_path(args.jax_repo):
    from qdx.utils import Utils as U_jax
    uj = U_jax(N, K, gates_seq, softness=N)
    E2 = np.asarray(uj.error_operators(2))
    kl_j = [int(uj.check_KL(uj.error_operators(w))) for w in range(1, 4)]
    klz_j = [uj.check_KL_cZ(uj.error_operators(w), 1.3) for w in range(1, 4)]
    tab_j = np.asarray(uj.tableau)
    S_j = np.asarray(uj.S)
with repo_on_path(args.torch_repo):
    from qdx.utils import Utils as U_torch
    un = U_torch(N, K, gates_seq, softness=N)
    E2n = np.asarray(un.error_operators(2))
    kl_n = [int(un.check_KL(un.error_operators(w))) for w in range(1, 4)]
    klz_n = [un.check_KL_cZ(un.error_operators(w), 1.3) for w in range(1, 4)]
    tab_n = np.asarray(un.tableau)
    S_n = np.asarray(un.S)

r.check_value_equal("Utils: tableau", tab_n, tab_j)
r.check_value_equal("Utils: stabilizer group S", S_n, S_j)
r.check_value_equal("Utils: error_operators(2)", E2n, E2)
r.check("Utils: check_KL by weight", kl_n == kl_j, f"torch {kl_n} vs jax {kl_j}")
r.check("Utils: check_KL_cZ by weight", klz_n == klz_j, f"torch {klz_n} vs jax {klz_j}")

r.finish()
