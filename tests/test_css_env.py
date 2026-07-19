"""JAX vs PyTorch: the icml2024-AI4science CSS environment and utilities."""

import numpy as np

from compare_utils import parse_args, repo_on_path, Reporter

args = parse_args(__doc__)

r = Reporter("icml2024 CSS environment / UtilsCSS")

N, K, D = 6, 2, 3
INIT_H = [2, 4]
MAX_STEPS = 8
N_ROLLOUT = 3 * MAX_STEPS

# shared keys / actions
with repo_on_path(args.torch_repo):
    from qdx import torch_random as nr
master = nr.PRNGKey(555)
step_keys = nr.split(nr.fold_in(master, 1), N_ROLLOUT)
reset_key = nr.fold_in(master, 2)


def run_side(repo, is_jax):
    with repo_on_path(repo, "icml2024-AI4science", icml=True):
        from envs.code_discovery_env import CodeDiscoveryCSS
        from simulators.clifford_gates_css import CliffordGatesCSS
        from simulators.tableau_simulator_css import TableauSimulatorCSS
        from utils_css import UtilsCSS

        g = CliffordGatesCSS(N)
        env = CodeDiscoveryCSS(N, K, D, INIT_H, [g.cx], max_steps=MAX_STEPS,
                               lbda=0.1, pI=0.7, softness=1, bell=False)
        A = env.action_space().n
        static = {
            "actions": np.asarray(env.actions),
            "z_action_move": np.asarray(env.z_action_move),
            "E_mu": np.asarray(env.E_mu),
            "p_mu": np.asarray(env.p_mu),
            "S_struct_X": np.asarray(env.S_struct_X),
            "S_struct_Z": np.asarray(env.S_struct_Z),
            "A": A,
        }
        actions = [int(x) for x in np.asarray(
            nr.randint(nr.fold_in(master, 7), (N_ROLLOUT,), 0, A))]

        if is_jax:
            import jax.numpy as jnp
            rk = jnp.asarray(reset_key)
            sk = [jnp.asarray(k) for k in step_keys]
        else:
            rk = reset_key
            sk = list(step_keys)

        obs, state = env.reset(rk, None)
        traj = {"obs": [np.asarray(obs)], "SX": [], "SZ": [], "reward": [],
                "done": [], "discount": []}
        for t in range(N_ROLLOUT):
            obs, state, rew, done, info = env.step(sk[t], state, actions[t], None)
            traj["obs"].append(np.asarray(obs))
            traj["SX"].append(np.asarray(state.S_X))
            traj["SZ"].append(np.asarray(state.S_Z))
            traj["reward"].append(float(rew))
            traj["done"].append(bool(done))
            traj["discount"].append(float(info["discount"]))

        # UtilsCSS on a fixed circuit
        tab = TableauSimulatorCSS(N, K, init_H=INIT_H, bell=False)
        for (a, b) in [(2, 3), (4, 5), (0, 2), (1, 4), (2, 5)]:
            tab.cx(a, b)
        utils = UtilsCSS(N, K, tab, softness=1)
        kl = [int(utils.check_KL(utils.CSS_error_operators(w))) for w in range(1, 4)]
        emu2 = np.asarray(utils.CSS_error_operators(2))

        return static, actions, traj, kl, emu2


sj, aj, tj, klj, emuj = run_side(args.jax_repo, True)
sn, an, tn, kln, emun = run_side(args.torch_repo, False)

r.check("identical action sequence used", an == aj)
for k in ["actions", "z_action_move", "E_mu", "S_struct_X", "S_struct_Z"]:
    r.check_value_equal(f"static {k}", sn[k], sj[k])
r.check_close("static p_mu", sn["p_mu"], sj["p_mu"], rtol=1e-6, atol=1e-7)
r.check("action space size", sn["A"] == sj["A"])

r.check("observations exact",
        all(np.array_equal(a, b) for a, b in zip(tn["obs"], tj["obs"])))
r.check("S_X exact", all(np.array_equal(a, b) for a, b in zip(tn["SX"], tj["SX"])))
r.check("S_Z exact", all(np.array_equal(a, b) for a, b in zip(tn["SZ"], tj["SZ"])))
r.check("dones exact", tn["done"] == tj["done"],
        f"np {sum(tn['done'])} vs jax {sum(tj['done'])} resets")
r.check("discounts exact", tn["discount"] == tj["discount"])
r.check_close("rewards", tn["reward"], tj["reward"], rtol=1e-5, atol=1e-5)

r.check("UtilsCSS.check_KL by weight", kln == klj, f"np {kln} vs jax {klj}")
r.check_value_equal("UtilsCSS.CSS_error_operators(2)", emun, emuj)

r.finish()
