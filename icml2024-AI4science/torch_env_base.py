"""PyTorch re-implementation of the small slice of gymnax used by this repo.

Replicates ``gymnax.environments.environment.Environment`` (including the
auto-reset key-splitting semantics of ``step``), the ``spaces`` classes, and
the ``FlattenObservationWrapper`` / ``LogWrapper`` from
``gymnax.wrappers.purerl`` — operating eagerly on torch tensors.

Observations and tableaux are torch tensors; per-episode scalar bookkeeping
(returns/lengths) is done with NumPy float32/int32 scalars to reproduce the
jitted original's float32 arithmetic exactly. Keys are threefry uint32 pairs
from :mod:`qdx.torch_random`.
"""

from dataclasses import dataclass

import numpy as np
import torch

import torch_random


# ---------------------------------------------------------------------------
# spaces
# ---------------------------------------------------------------------------

class Discrete:
    """Minimal analogue of gymnax spaces.Discrete."""

    def __init__(self, num_categories, dtype=torch.int32):
        self.n = num_categories
        self.shape = ()
        self.dtype = dtype


class Box:
    """Minimal analogue of gymnax spaces.Box."""

    def __init__(self, low, high, shape, dtype=torch.float32):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype


class Dict:
    """Minimal analogue of gymnax spaces.Dict."""

    def __init__(self, spaces):
        self.spaces = spaces


class spaces:
    """Namespace mirroring ``gymnax.environments.spaces``."""

    Discrete = Discrete
    Box = Box
    Dict = Dict


# ---------------------------------------------------------------------------
# Environment base class
# ---------------------------------------------------------------------------

class Environment:
    """Abstract base class mirroring gymnax's Environment semantics."""

    @property
    def default_params(self):
        return None

    def step(self, key, state, action, params=None):
        """Performs step transitions in the environment (with auto-reset)."""
        if params is None:
            params = self.default_params

        # Step (same key-splitting order as gymnax)
        keys = torch_random.split(key)
        key_step, key_reset = keys[0], keys[1]
        obs_st, state_st, reward, done, info = self.step_env(
            key_step, state, action, params
        )
        obs_re, state_re = self.reset_env(key_reset, params)

        # Auto-reset environment based on termination
        if done:
            state = state_re
            obs = obs_re
        else:
            state = state_st
            obs = obs_st

        return obs, state, reward, done, info

    def reset(self, key, params=None):
        """Performs resetting of environment."""
        if params is None:
            params = self.default_params
        obs, state = self.reset_env(key, params)
        return obs, state

    def step_env(self, key, state, action, params):
        raise NotImplementedError

    def reset_env(self, key, params):
        raise NotImplementedError

    def get_obs(self, state, params=None, key=None):
        raise NotImplementedError

    def is_terminal(self, state, params):
        raise NotImplementedError

    def discount(self, state, params):
        """Return a discount of zero if the episode has terminated."""
        return np.float32(0.0) if self.is_terminal(state, params) else np.float32(1.0)

    @property
    def name(self):
        return type(self).__name__

    @property
    def num_actions(self):
        raise NotImplementedError

    def action_space(self, params=None):
        raise NotImplementedError

    def observation_space(self, params=None):
        raise NotImplementedError

    def state_space(self, params=None):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Wrappers (gymnax.wrappers.purerl)
# ---------------------------------------------------------------------------

class GymnaxWrapper:
    """Base class for wrappers."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenObservationWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def observation_space(self, params=None):
        space = self._env.observation_space(params)
        assert isinstance(space, Box), "Only Box spaces are supported for now."
        return Box(
            low=space.low,
            high=space.high,
            shape=(int(np.prod(space.shape)),),
            dtype=space.dtype,
        )

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return torch.reshape(obs, (-1,)), state

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return torch.reshape(obs, (-1,)), state, reward, done, info


@dataclass(frozen=True)
class LogEnvState:
    env_state: object
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def reset(self, key, params=None):
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, np.float32(0.0), np.int32(0),
                            np.float32(0.0), np.int32(0))
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        done_f = np.float32(done)
        done_i = np.int32(done)
        new_episode_return = np.float32(state.episode_returns + np.float32(reward))
        new_episode_length = np.int32(state.episode_lengths + 1)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=np.float32(new_episode_return * (1 - done_f)),
            episode_lengths=np.int32(new_episode_length * (1 - done_i)),
            returned_episode_returns=np.float32(
                state.returned_episode_returns * (1 - done_f)
                + new_episode_return * done_f),
            returned_episode_lengths=np.int32(
                state.returned_episode_lengths * (1 - done_i)
                + new_episode_length * done_i),
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        return obs, state, reward, done, info
