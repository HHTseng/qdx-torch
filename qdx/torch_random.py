"""Re-implementation of the JAX PRNG (threefry2x32) and the sampling
primitives used throughout this repository (PyTorch port).

The PRNG itself runs on NumPy uint32 arithmetic on purpose: randomness needs
no autodiff or GPU, and NumPy integer ops let us reproduce ``jax.random``
bit-for-bit. Consumers convert the resulting arrays to ``torch`` tensors at
the boundary (see :mod:`qdx.torch_nn`).

The goal is bit-exact reproduction of ``jax.random`` (with the default
``jax_threefry_partitionable=True`` behaviour of recent JAX versions) so that
the PyTorch port of the library produces numerically matching results:

    PRNGKey / split / fold_in / random_bits   -> bit-exact
    uniform / gumbel / categorical / randint /
    choice / permutation                      -> bit-exact
    normal (float32, Giles erfinv polynomial) -> bit-exact up to libm ulp
    flax parameter RNG folding (SHA-1 paths)  -> bit-exact

Keys are represented exactly like legacy JAX keys: ``np.ndarray`` of shape
``(2,)`` (or ``(n, 2)`` for split results) with dtype ``uint32``.
"""

import hashlib
import math

import numpy as np

UINT32_MAX = np.uint32(0xFFFFFFFF)


# ---------------------------------------------------------------------------
# threefry2x32 block cipher (the core of JAX's default PRNG)
# ---------------------------------------------------------------------------

def _rotate_left(x, d):
    d = np.uint32(d)
    return (x << d) | (x >> np.uint32(32 - d))


def _threefry2x32(k1, k2, x1, x2):
    """Apply the Threefry-2x32 hash to pairs (x1, x2) with key (k1, k2).

    All inputs are uint32 arrays (x1 and x2 must have the same shape);
    the two output arrays have that same shape. Matches JAX's
    ``threefry2x32_p`` exactly.
    """
    rotations = ((13, 15, 26, 6), (17, 29, 16, 24))
    k1 = np.uint32(k1)
    k2 = np.uint32(k2)
    ks = (k1, k2, k1 ^ k2 ^ np.uint32(0x1BD11BDA))

    x0 = np.asarray(x1, dtype=np.uint32)
    x1_ = np.asarray(x2, dtype=np.uint32)

    with np.errstate(over="ignore"):
        x0 = x0 + ks[0]
        x1_ = x1_ + ks[1]

        def apply_rounds(a, b, rots):
            for r in rots:
                a = a + b
                b = _rotate_left(b, r)
                b = a ^ b
            return a, b

        x0, x1_ = apply_rounds(x0, x1_, rotations[0])
        x0 = x0 + ks[1]
        x1_ = x1_ + ks[2] + np.uint32(1)

        x0, x1_ = apply_rounds(x0, x1_, rotations[1])
        x0 = x0 + ks[2]
        x1_ = x1_ + ks[0] + np.uint32(2)

        x0, x1_ = apply_rounds(x0, x1_, rotations[0])
        x0 = x0 + ks[0]
        x1_ = x1_ + ks[1] + np.uint32(3)

        x0, x1_ = apply_rounds(x0, x1_, rotations[1])
        x0 = x0 + ks[1]
        x1_ = x1_ + ks[2] + np.uint32(4)

        x0, x1_ = apply_rounds(x0, x1_, rotations[0])
        x0 = x0 + ks[2]
        x1_ = x1_ + ks[0] + np.uint32(5)

    return x0, x1_


def threefry_2x32(key, count):
    """JAX's ``threefry_2x32``: hash a flat uint32 ``count`` vector."""
    key = np.asarray(key, dtype=np.uint32)
    count = np.asarray(count, dtype=np.uint32)
    odd_size = count.size % 2
    flat = count.ravel()
    if odd_size:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.uint32)])
    half = flat.size // 2
    b1, b2 = _threefry2x32(key[0], key[1], flat[:half], flat[half:])
    out = np.concatenate([b1, b2])
    if odd_size:
        out = out[:-1]
    return out.reshape(count.shape)


def _iota_2x32_shape(shape):
    """64-bit iota over ``shape`` as a (hi, lo) pair of uint32 arrays."""
    size = int(np.prod(shape)) if len(shape) else 1
    idx = np.arange(size, dtype=np.uint64)
    hi = (idx >> np.uint64(32)).astype(np.uint32).reshape(shape)
    lo = (idx & np.uint64(0xFFFFFFFF)).astype(np.uint32).reshape(shape)
    return hi, lo


# ---------------------------------------------------------------------------
# Key construction and manipulation
# ---------------------------------------------------------------------------

def PRNGKey(seed):
    """Equivalent of ``jax.random.PRNGKey`` (threefry, legacy uint32 keys).

    With x64 disabled (the JAX default used by this repo), the integer seed is
    canonicalized to int32, so the high key word is always zero.
    """
    seed = int(seed)
    return np.array([np.uint32(0), np.uint32(seed & 0xFFFFFFFF)], dtype=np.uint32)


def split(key, num=2):
    """Equivalent of ``jax.random.split`` (partitionable/'foldlike' mode)."""
    key = np.asarray(key, dtype=np.uint32)
    shape = (int(num),)
    c1, c2 = _iota_2x32_shape(shape)
    b1, b2 = _threefry2x32(key[0], key[1], c1, c2)
    return np.stack([b1, b2], axis=-1)


def fold_in(key, data):
    """Equivalent of ``jax.random.fold_in`` for uint32 ``data``."""
    key = np.asarray(key, dtype=np.uint32)
    data = np.uint32(data)
    seed_pair = np.array([np.uint32(0), data], dtype=np.uint32)
    return threefry_2x32(key, seed_pair)


def random_bits(key, shape):
    """32-bit random bits, matching partitionable ``jax.random.bits``."""
    key = np.asarray(key, dtype=np.uint32)
    shape = tuple(int(s) for s in shape)
    c1, c2 = _iota_2x32_shape(shape)
    b1, b2 = _threefry2x32(key[0], key[1], c1, c2)
    return b1 ^ b2


# ---------------------------------------------------------------------------
# Samplers (float32, matching JAX defaults with x64 disabled)
# ---------------------------------------------------------------------------

def _fma_f32(a, b, c):
    """float32 fused multiply-add, emulated in float64.

    XLA fuses ``a * b + c`` into a single-rounding FMA on CPU. The float64
    emulation is exact for the product (24-bit mantissas) and performs the
    single rounding on the sum, matching a true FMA except for astronomically
    rare double-rounding cases (~2^-29 per element).
    """
    return (np.asarray(a, np.float64) * np.asarray(b, np.float64)
            + np.asarray(c, np.float64)).astype(np.float32)


def uniform(key, shape=(), minval=0.0, maxval=1.0):
    """float32 uniform in [minval, maxval), bit-exact vs jax.random.uniform."""
    shape = tuple(int(s) for s in shape)
    minval = np.float32(minval)
    maxval = np.float32(maxval)
    bits = random_bits(key, shape)
    float_bits = (bits >> np.uint32(9)) | np.uint32(0x3F800000)
    floats = float_bits.view(np.float32) - np.float32(1.0)
    return np.maximum(minval, _fma_f32(floats, maxval - minval, minval))


_ERFINV_SMALL = np.array(
    [2.81022636e-08, 3.43273939e-07, -3.5233877e-06, -4.39150654e-06,
     0.00021858087, -0.00125372503, -0.00417768164, 0.246640727, 1.50140941],
    dtype=np.float32)

_ERFINV_BIG = np.array(
    [-0.000200214257, 0.000100950558, 0.00134934322, -0.00367342844,
     0.00573950773, -0.0076224613, 0.00943887047, 1.00167406, 2.83297682],
    dtype=np.float32)


def _erfinv_f32(x):
    """float32 inverse error function (Giles 2012), as used by XLA.

    Matches jax.lax.erf_inv up to 1-ulp differences stemming from the
    platform log1p implementation.
    """
    x = np.asarray(x, dtype=np.float32)
    w = -np.log1p(-x * x)
    small = w < np.float32(5.0)
    w_small = w - np.float32(2.5)
    with np.errstate(invalid="ignore"):
        w_big = np.sqrt(w) - np.float32(3.0)

    p_small = np.full_like(x, _ERFINV_SMALL[0])
    for c in _ERFINV_SMALL[1:]:
        p_small = _fma_f32(p_small, w_small, c)
    p_big = np.full_like(x, _ERFINV_BIG[0])
    for c in _ERFINV_BIG[1:]:
        p_big = _fma_f32(p_big, w_big, c)

    return np.where(small, p_small, p_big) * x


def normal(key, shape=()):
    """float32 standard normal, matching jax.random.normal."""
    lo = np.nextafter(np.float32(-1.0), np.float32(0.0))
    u = uniform(key, shape, lo, np.float32(1.0))
    return np.float32(np.sqrt(2)) * _erfinv_f32(u)


def gumbel(key, shape=()):
    """float32 standard Gumbel (JAX 'low' mode, the default)."""
    tiny = np.finfo(np.float32).tiny
    u = uniform(key, shape, tiny, np.float32(1.0))
    return -np.log(-np.log(u))


def truncated_normal(key, lower, upper, shape=()):
    """float32 truncated normal, matching jax.random.truncated_normal."""
    import math

    sqrt2 = np.float32(np.sqrt(2))
    lower = np.float32(lower)
    upper = np.float32(upper)
    a = np.float32(math.erf(np.float32(lower / sqrt2)))
    b = np.float32(math.erf(np.float32(upper / sqrt2)))
    u = uniform(key, shape, a, b)
    out = sqrt2 * _erfinv_f32(u)
    # jax clips to the open interval just inside (lower, upper)
    lo = np.nextafter(lower, np.float32(np.inf))
    hi = np.nextafter(upper, np.float32(-np.inf))
    return np.clip(out, lo, hi).astype(np.float32)


def lecun_normal_init(key, shape):
    """flax's default Dense kernel init: variance_scaling(1.0, 'fan_in',
    'truncated_normal') for a kernel of shape (fan_in, fan_out)."""
    fan_in = shape[0]
    variance = np.float32(1.0) / np.float32(max(1, fan_in))
    # jax divides by the truncated-normal stddev correction constant
    stddev = np.float32(np.sqrt(variance)) / np.float32(0.87962566103423978)
    return (truncated_normal(key, -2.0, 2.0, shape) * stddev).astype(np.float32)


def categorical(key, logits, axis=-1, shape=None):
    """Gumbel-max categorical sampling, matching jax.random.categorical."""
    logits = np.asarray(logits, dtype=np.float32)
    batch_shape = tuple(np.delete(logits.shape, axis))
    if shape is None:
        shape = batch_shape
    shape = tuple(shape)
    shape_prefix = shape[:len(shape) - len(batch_shape)]

    ax = axis
    if ax >= 0:
        ax -= logits.ndim
    logits_shape = list(shape[len(shape) - len(batch_shape):])
    logits_shape.insert(ax % logits.ndim, logits.shape[axis])

    g = gumbel(key, tuple(shape_prefix) + tuple(logits_shape))
    expanded = logits.reshape((1,) * len(shape_prefix) + logits.shape)
    return np.argmax(g + expanded, axis=ax).astype(np.int32)


def permutation(key, x):
    """Matching jax.random.permutation for 1-D arrays or integer inputs."""
    if np.ndim(x) == 0:
        x = np.arange(int(x), dtype=np.int32)
    else:
        x = np.asarray(x).copy()
    size = x.size
    exponent = 3
    uint32max = np.iinfo(np.uint32).max
    num_rounds = int(np.ceil(exponent * np.log(max(1, size)) / np.log(uint32max)))
    for _ in range(num_rounds):
        keys = split(key)
        key, subkey = keys[0], keys[1]
        sort_keys = random_bits(subkey, x.shape)
        order = np.argsort(sort_keys, kind="stable")
        x = np.take(x, order)
    return x


def randint(key, shape, minval, maxval):
    """Matching jax.random.randint for int32 outputs."""
    shape = tuple(int(s) for s in shape)
    minval_i = int(minval)
    maxval_i = int(maxval)
    keys = split(key)
    k1, k2 = keys[0], keys[1]
    higher_bits = random_bits(k1, shape)
    lower_bits = random_bits(k2, shape)

    span = np.uint32(maxval_i - minval_i) if maxval_i > minval_i else np.uint32(1)
    with np.errstate(over="ignore"):
        multiplier = np.uint32(np.uint32(2 ** 16) % span)
        multiplier = np.uint32((multiplier * multiplier) % span)
        random_offset = (higher_bits % span) * multiplier + (lower_bits % span)
        random_offset = random_offset % span
    return (np.int32(minval_i) + random_offset.astype(np.int32)).reshape(shape)


def choice(key, a, shape=()):
    """Matching jax.random.choice(key, a[, shape]) for a 1-D array ``a``
    (uniform probabilities, with replacement)."""
    a = np.asarray(a)
    n_inputs = a.shape[0]
    ind = randint(key, shape, 0, n_inputs)
    if shape == ():
        return a[int(ind)]
    return np.take(a, ind, axis=0).reshape(shape)


# ---------------------------------------------------------------------------
# flax parameter-RNG derivation (LazyRng folding via SHA-1)
# ---------------------------------------------------------------------------

def flax_fold_in_static(key, suffix):
    """Replicates flax.core.scope._fold_in_static (flax_fix_rng_separator=False).

    ``suffix`` is a tuple of str/int path elements, e.g. ('Dense_0', 1).
    """
    if not suffix:
        return np.asarray(key, dtype=np.uint32)
    m = hashlib.sha1()
    for x in suffix:
        if isinstance(x, str):
            m.update(x.encode("utf-8"))
        elif isinstance(x, (int, np.integer)):
            x = int(x)
            m.update(x.to_bytes((x.bit_length() + 7) // 8, byteorder="big"))
        else:
            raise ValueError(f"Expected int or string, got: {x}")
    d = m.digest()
    hash_int = int.from_bytes(d[:4], byteorder="big")
    return fold_in(key, np.uint32(hash_int))
