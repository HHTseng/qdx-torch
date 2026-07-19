"""JAX vs PyTorch: PRNG primitives must match bit-for-bit (normal/gumbel up to
1-ulp libm differences)."""

import numpy as np

from compare_utils import parse_args, repo_on_path, Reporter

args = parse_args(__doc__)

import jax
import jax.numpy as jnp
import flax
from flax.core import scope as flax_scope

with repo_on_path(args.torch_repo):
    from qdx import torch_random as nr

r = Reporter("PRNG primitives (JAX vs PyTorch threefry)")

# PRNGKey
for seed in [0, 1, 42, 43, 44, 1234, 2**31 - 1, 2**33 + 7]:
    r.check_exact(f"PRNGKey({seed})", nr.PRNGKey(seed), np.asarray(jax.random.PRNGKey(seed)))

key = jax.random.PRNGKey(42)
nkey = nr.PRNGKey(42)

# split
for num in [2, 3, 4, 16, 64, 100]:
    r.check_exact(f"split(num={num})", nr.split(nkey, num),
                  np.asarray(jax.random.split(key, num)))

# nested split chains
k_j, k_n = key, nkey
for i in range(5):
    k_j = jax.random.split(k_j)[1]
    k_n = nr.split(k_n)[1]
r.check_exact("chained splits", k_n, np.asarray(k_j))

# fold_in
for d in [0, 1, 7, 99, 123456789, 2**32 - 1]:
    r.check_exact(f"fold_in({d})", nr.fold_in(nkey, d),
                  np.asarray(jax.random.fold_in(key, jnp.uint32(d))))

# random bits
for shape in [(), (1,), (5,), (7, 3), (2, 4, 5), (1000,)]:
    r.check_exact(f"bits{shape}", nr.random_bits(nkey, shape),
                  np.asarray(jax.random.bits(key, shape, dtype=jnp.uint32)))

# uniform (incl. non-default ranges, which exercise the FMA path)
r.check_exact("uniform (0,1)", nr.uniform(nkey, (10000,)),
              np.asarray(jax.random.uniform(key, (10000,))))
r.check_exact("uniform (-1,2)", nr.uniform(nkey, (10000,), -1.0, 2.0),
              np.asarray(jax.random.uniform(key, (10000,), minval=-1.0, maxval=2.0)))
tiny = float(np.finfo(np.float32).tiny)
r.check_exact("uniform (tiny,1)", nr.uniform(nkey, (10000,), tiny, 1.0),
              np.asarray(jax.random.uniform(key, (10000,), minval=tiny, maxval=1.0)))

# normal: equal within 1 ulp (log1p implementation differences)
a = nr.normal(nkey, (100000,))
b = np.asarray(jax.random.normal(key, (100000,)))
r.check_close("normal (1-ulp tolerance)", a, b, rtol=0, atol=5e-7)
frac = np.mean(a != b)
r.check("normal bit-agreement > 98%", frac < 0.02, f"{100*(1-frac):.2f}% bit-identical")

# gumbel: equal within 1 ulp of log
g = nr.gumbel(nkey, (100000,))
gj = np.asarray(jax.random.gumbel(key, (100000,)))
r.check_close("gumbel (1-ulp tolerance)", g, gj, rtol=1e-5, atol=1e-6)

# categorical: decisions must agree
flips = 0
total = 0
for s in range(300):
    k = jax.random.PRNGKey(s)
    nk = nr.PRNGKey(s)
    logits = np.asarray(jax.random.normal(jax.random.fold_in(k, 999), (64, 66)))
    cj = np.asarray(jax.random.categorical(k, jnp.asarray(logits), shape=(1, 64)))
    cn = nr.categorical(nk, logits, shape=(1, 64))
    flips += int(np.sum(cj != cn))
    total += cj.size
r.check("categorical decisions identical", flips == 0, f"{flips}/{total} flips")

# permutation
for n in [1, 10, 320, 512, 2048]:
    r.check_value_equal(f"permutation({n})", nr.permutation(nkey, n),
                        np.asarray(jax.random.permutation(key, n)))

# randint / choice
r.check_value_equal("randint(0,21)", nr.randint(nkey, (1000,), 0, 21),
                    np.asarray(jax.random.randint(key, (1000,), 0, 21)))
r.check_value_equal("randint(5,300)", nr.randint(nkey, (1000,), 5, 300),
                    np.asarray(jax.random.randint(key, (1000,), 5, 300)))
for s in range(50):
    k = jax.random.PRNGKey(s); nk = nr.PRNGKey(s)
    cj = np.asarray(jax.random.choice(k, jnp.array(range(21))))
    cn = nr.choice(nk, np.array(range(21)))
    if int(cj) != int(cn):
        r.check(f"choice seed {s}", False, f"{cn} vs {cj}")
        break
else:
    r.check("choice scalar over 50 seeds", True)
r.check_value_equal("choice shaped", nr.choice(nkey, np.arange(84), shape=(500,)),
                    np.asarray(jax.random.choice(key, jnp.arange(84), shape=(500,))))

# flax parameter-RNG folding
for suffix in [("Dense_0", 1), ("Dense_0", 2), ("Dense_3", 1), ("Dense_5", 2)]:
    lazy = flax_scope.LazyRng.create(key, *suffix)
    r.check_exact(f"flax fold {suffix}", nr.flax_fold_in_static(nkey, suffix),
                  np.asarray(lazy.as_jax_rng()))

r.finish()
