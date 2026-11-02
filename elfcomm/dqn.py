from __future__ import annotations

import copy
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TensorQNetwork(nn.Module):
    """State-conditioned CP decision tensor, flattened in NumPy C order."""

    def __init__(self, state_dim, action_shape, rank=8, hidden=128, additive_heads=False):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_shape = tuple(int(n) for n in action_shape)
        self.rank = int(rank)
        if self.state_dim < 1 or not self.action_shape or any(n < 1 for n in self.action_shape):
            raise ValueError("State and action dimensions must be positive.")
        if self.rank < 1 or hidden < 1:
            raise ValueError("Tensor rank and hidden size must be positive.")
        self.action_count = math.prod(self.action_shape)
        self.body = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.weights_head = nn.Linear(hidden, rank)
        self.value_head = nn.Linear(hidden, 1)
        self.axis_heads = nn.ModuleList(nn.Linear(hidden, n) for n in self.action_shape) if additive_heads else nn.ModuleList()
        self.factors = nn.ParameterList([nn.Parameter(torch.empty(n, rank)) for n in self.action_shape])
        for factor in self.factors:
            nn.init.normal_(factor, mean=0.0, std=0.7)

    def action_basis(self):
        basis = self.factors[0]
        for factor in self.factors[1:]:
            basis = (basis[:, None, :] * factor[None, :, :]).reshape(-1, self.rank)
        return basis

    def forward(self, states):
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if states.ndim != 2 or states.shape[-1] != self.state_dim:
            raise ValueError("Unexpected state shape.")
        features = self.body(states)
        values = self.value_head(features) + self.weights_head(features) @ self.action_basis().T
        if self.axis_heads:
            grid = values.reshape(-1, *self.action_shape)
            for axis, head in enumerate(self.axis_heads):
                advantage = head(features)
                advantage = advantage - advantage.mean(dim=1, keepdim=True)
                shape = [features.shape[0]] + [1] * len(self.action_shape)
                shape[axis + 1] = self.action_shape[axis]
                grid = grid + advantage.reshape(shape)
            values = grid.reshape(-1, self.action_count)
        return values


class _ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.items = []
        self.position = 0
        self.group_keys = []
        self.group_offsets = []
        self.groups = {}

    def __len__(self):
        return len(self.items)

    def append(self, transition, group=None):
        index = self.position
        if len(self.items) < self.capacity:
            self.items.append(transition)
            self.group_keys.append(group)
            self.group_offsets.append(0)
        else:
            old_group = self.group_keys[index]
            old_pool = self.groups[old_group]
            offset = self.group_offsets[index]
            last = old_pool.pop()
            if offset < len(old_pool):
                old_pool[offset] = last
                self.group_offsets[last] = offset
            if not old_pool:
                del self.groups[old_group]
            self.items[self.position] = transition
            self.group_keys[index] = group
        pool = self.groups.setdefault(group, [])
        self.group_offsets[index] = len(pool)
        pool.append(index)
        self.position = (self.position + 1) % self.capacity

    def sample(self, size, rng):
        return [self.items[i] for i in rng.sample(range(len(self.items)), size)]

    def sample_balanced(self, size, rng, mixture):
        # Mix uniform replay with power/feasibility strata; importance weights
        # correct the changed sampling probabilities rather than changing rewards.
        pools = list(self.groups.values())
        count = len(self.items)
        indices, weights = [], []
        for _ in range(size):
            if rng.random() < mixture:
                index = rng.choice(rng.choice(pools))
            else:
                index = rng.randrange(count)
            probability = ((1 - mixture) / count
                           + mixture / (len(pools) * len(self.groups[self.group_keys[index]])))
            indices.append(index)
            weights.append(1.0 / (count * probability))
        return [self.items[i] for i in indices], weights


class DQNAgent:
    def __init__(self, state_dim, action_shape, cfg, seed, device="cpu"):
        self.state_dim = int(state_dim)
        self.action_shape = tuple(int(n) for n in action_shape)
        self.action_count = math.prod(self.action_shape)
        self.cfg = dict(cfg)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.gamma = float(cfg.get("gamma", 0.95))
        self.batch_size = int(cfg.get("batch_size", 64))
        self.warmup_steps = int(cfg.get("warmup_steps", 256))
        self.target_tau = float(cfg.get("target_tau", 0.01))
        self.gradient_clip = float(cfg.get("gradient_clip", 5.0))
        self.replay_balance = float(cfg.get("replay_balance_fraction", 0.0))
        if not 0 <= self.replay_balance < 1:
            raise ValueError("replay_balance_fraction must lie in [0, 1)")
        if not 0 <= self.gamma <= 1 or self.batch_size < 1 or self.warmup_steps < 0:
            raise ValueError("Invalid DQN gamma, batch size, or warmup.")
        if not 0 < self.target_tau <= 1 or self.gradient_clip <= 0:
            raise ValueError("Invalid target update or gradient clip.")
        self.replay = _ReplayBuffer(cfg.get("replay_capacity", 20000))
        if self.replay.capacity < max(self.batch_size, self.warmup_steps):
            raise ValueError("Replay capacity must accommodate batch size and warmup.")
        self.exploration_rng = np.random.default_rng(self.seed + 11)
        self.replay_rng = random.Random(self.seed + 29)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.online = TensorQNetwork(self.state_dim, self.action_shape, int(cfg.get("tensor_rank", 8)), int(cfg.get("hidden_size", 128)), bool(cfg.get("additive_value_heads", False))).to(self.device)
        self.target = copy.deepcopy(self.online)
        self.target.eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=float(cfg.get("learning_rate", 3e-4)))
        self.step_count = 0
        self.transition_count = 0
        self.last_action_source = "greedy"

    def _state(self, value):
        result = np.asarray(value, dtype=np.float32)
        if result.shape != (self.state_dim,) or not np.all(np.isfinite(result)):
            raise ValueError("State must have the configured dimension and finite values.")
        return result

    def _mask(self, value):
        result = np.asarray(value, dtype=bool)
        if result.shape != (self.action_count,):
            raise ValueError("Mask must have one value per flattened action.")
        return result

    def act(self, state, mask, epsilon=0.0, exploration_anchor=None):
        state_array = self._state(state)
        mask_array = self._mask(mask)
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1.")
        legal = np.flatnonzero(mask_array)
        if not legal.size:
            self.last_action_source = "outage"
            return -1
        if epsilon > 0 and self.exploration_rng.random() < epsilon:
            options = self.cfg.get("structured_exploration", {})
            if (options.get("enabled", False) and exploration_anchor is not None
                    and len(self.action_shape) == 4
                    and 0 <= exploration_anchor < self.action_count and mask_array[exploration_anchor]
                    and self.exploration_rng.random() >= options.get("global_fraction", 0.5)):
                coordinates = list(np.unravel_index(exploration_anchor, self.action_shape))
                axis = (1 if self.exploration_rng.random() < 0.5
                        else int(self.exploration_rng.choice([0, 2, 3])))
                candidates = []
                for value in range(self.action_shape[axis]):
                    if value == coordinates[axis]:
                        continue
                    if axis == 0 and abs(value - coordinates[axis]) > options.get("frequency_neighbor_radius", 3):
                        continue
                    trial = coordinates.copy()
                    trial[axis] = value
                    index = int(np.ravel_multi_index(tuple(trial), self.action_shape))
                    if mask_array[index]:
                        candidates.append(index)
                if candidates:
                    self.last_action_source = "local_exploration"
                    return int(self.exploration_rng.choice(candidates))
            self.last_action_source = "global_exploration"
            return int(self.exploration_rng.choice(legal))
        self.last_action_source = "greedy"
        with torch.no_grad():
            q = self.online(torch.as_tensor(state_array, device=self.device))[0]
            return int(q.masked_fill(~torch.as_tensor(mask_array, device=self.device), -torch.inf).argmax().item())

    def remember(self, state, action, reward, next_state, done, next_mask, feasible=None):
        if not 0 <= action < self.action_count:
            raise ValueError("Skip outage (-1) transitions; only actual actions enter replay.")
        if not math.isfinite(reward):
            raise ValueError("Reward must be finite.")
        self.replay.append((self._state(state).copy(), int(action), float(reward), self._state(next_state).copy(),
                            bool(done), np.packbits(self._mask(next_mask))),
                           group=(int(np.unravel_index(action, self.action_shape)[1]), feasible))
        self.transition_count += 1

    def _bootstrap_values(self, next_states, dones, next_masks):
        with torch.no_grad():
            usable = (~dones) & next_masks.any(dim=1)
            values = torch.zeros(next_states.shape[0], device=self.device)
            if usable.any():
                selected = next_states[usable]
                online_q = self.online(selected).masked_fill(~next_masks[usable], -torch.inf)
                actions = online_q.argmax(dim=1, keepdim=True)
                values[usable] = self.target(selected).gather(1, actions).squeeze(1)
            return values

    def learn(self):
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return None
        if self.replay_balance:
            samples, weights = self.replay.sample_balanced(self.batch_size, self.replay_rng, self.replay_balance)
        else:
            samples = self.replay.sample(self.batch_size, self.replay_rng)
            weights = [1.0] * self.batch_size
        state, action, reward, next_state, done, next_mask = zip(*samples)
        states = torch.as_tensor(np.stack(state), dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack(next_state), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(action, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(done, dtype=torch.bool, device=self.device)
        next_masks = np.stack([np.unpackbits(mask, count=self.action_count) for mask in next_mask])
        next_masks = torch.as_tensor(next_masks, dtype=torch.bool, device=self.device)
        targets = rewards + self.gamma * self._bootstrap_values(next_states, dones, next_masks)
        predictions = self.online(states).gather(1, actions[:, None]).squeeze(1)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        loss = (F.smooth_l1_loss(predictions, targets, reduction="none") * weights).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("DQN loss became nonfinite.")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.gradient_clip)
        self.optimizer.step()
        with torch.no_grad():
            for target_parameter, online_parameter in zip(self.target.parameters(), self.online.parameters()):
                target_parameter.lerp_(online_parameter, self.target_tau)
        self.step_count += 1
        return float(loss.item())

    def reset_replay(self):
        self.replay = _ReplayBuffer(self.replay.capacity)

    def save(self, path, include_replay=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": 3, "state_dim": self.state_dim, "action_shape": self.action_shape,
                   "cfg": self.cfg, "seed": self.seed, "online": self.online.state_dict(),
                   "target": self.target.state_dict(), "optimizer": self.optimizer.state_dict(),
                   "step_count": self.step_count, "transition_count": self.transition_count,
                   "exploration_rng": self.exploration_rng.bit_generator.state,
                   "replay_rng": self.replay_rng.getstate()}
        if include_replay:
            payload["replay_items"] = self.replay.items
            payload["replay_groups"] = self.replay.group_keys
            payload["replay_position"] = self.replay.position
        with Path(path).open("wb") as stream:
            torch.save(payload, stream)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint["state_dim"] != self.state_dim or tuple(checkpoint["action_shape"]) != self.action_shape:
            raise ValueError("Checkpoint state/action dimensions do not match.")
        for key, default in (("tensor_rank", 8), ("hidden_size", 128), ("additive_value_heads", False)):
            if checkpoint["cfg"].get(key, default) != self.cfg.get(key, default):
                raise ValueError("Checkpoint tensor architecture does not match.")
        replay_items = checkpoint.get("replay_items", [])
        if len(replay_items) > self.replay.capacity:
            raise ValueError("Checkpoint replay does not fit.")
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step_count = int(checkpoint["step_count"])
        self.transition_count = int(checkpoint["transition_count"])
        self.exploration_rng.bit_generator.state = checkpoint["exploration_rng"]
        self.replay_rng.setstate(checkpoint["replay_rng"])
        replay_groups = checkpoint.get("replay_groups", [None] * len(replay_items))
        if len(replay_groups) != len(replay_items):
            raise ValueError("Checkpoint replay groups do not match replay items")
        self.replay = _ReplayBuffer(self.replay.capacity)
        for transition, group in zip(replay_items, replay_groups):
            self.replay.append(transition, group=group)
        self.replay.position = int(checkpoint.get("replay_position", 0)) % self.replay.capacity
