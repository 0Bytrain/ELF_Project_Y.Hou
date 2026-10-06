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

    def __init__(self, state_dim, action_shape, rank=8, hidden=128):
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
        return self.value_head(features) + self.weights_head(features) @ self.action_basis().T


class _ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.items = []
        self.position = 0

    def __len__(self):
        return len(self.items)

    def append(self, transition):
        if len(self.items) < self.capacity:
            self.items.append(transition)
        else:
            self.items[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, size, rng):
        return [self.items[i] for i in rng.sample(range(len(self.items)), size)]


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
            self.online = TensorQNetwork(self.state_dim, self.action_shape, int(cfg.get("tensor_rank", 8)), int(cfg.get("hidden_size", 128))).to(self.device)
        self.target = copy.deepcopy(self.online)
        self.target.eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=float(cfg.get("learning_rate", 3e-4)))
        self.step_count = 0
        self.transition_count = 0

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

    def act(self, state, mask, epsilon=0.0):
        state_array = self._state(state)
        mask_array = self._mask(mask)
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1.")
        legal = np.flatnonzero(mask_array)
        if not legal.size:
            return -1
        if epsilon > 0 and self.exploration_rng.random() < epsilon:
            return int(self.exploration_rng.choice(legal))
        with torch.no_grad():
            q = self.online(torch.as_tensor(state_array, device=self.device))[0]
            return int(q.masked_fill(~torch.as_tensor(mask_array, device=self.device), -torch.inf).argmax().item())

    def remember(self, state, action, reward, next_state, done, next_mask):
        if not 0 <= action < self.action_count:
            raise ValueError("Skip outage (-1) transitions; only actual actions enter replay.")
        if not math.isfinite(reward):
            raise ValueError("Reward must be finite.")
        self.replay.append((self._state(state).copy(), int(action), float(reward), self._state(next_state).copy(), bool(done), self._mask(next_mask).copy()))
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
        state, action, reward, next_state, done, next_mask = zip(*self.replay.sample(self.batch_size, self.replay_rng))
        states = torch.as_tensor(np.stack(state), dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack(next_state), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(action, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(done, dtype=torch.bool, device=self.device)
        next_masks = torch.as_tensor(np.stack(next_mask), dtype=torch.bool, device=self.device)
        targets = rewards + self.gamma * self._bootstrap_values(next_states, dones, next_masks)
        predictions = self.online(states).gather(1, actions[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(predictions, targets)
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

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 1, "state_dim": self.state_dim, "action_shape": self.action_shape, "cfg": self.cfg, "seed": self.seed, "online": self.online.state_dict(), "target": self.target.state_dict(), "optimizer": self.optimizer.state_dict(), "step_count": self.step_count, "transition_count": self.transition_count, "exploration_rng": self.exploration_rng.bit_generator.state, "replay_rng": self.replay_rng.getstate(), "replay_items": self.replay.items, "replay_position": self.replay.position}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint["state_dim"] != self.state_dim or tuple(checkpoint["action_shape"]) != self.action_shape:
            raise ValueError("Checkpoint state/action dimensions do not match.")
        for key, default in (("tensor_rank", 8), ("hidden_size", 128)):
            if checkpoint["cfg"].get(key, default) != self.cfg.get(key, default):
                raise ValueError("Checkpoint tensor architecture does not match.")
        if len(checkpoint["replay_items"]) > self.replay.capacity:
            raise ValueError("Checkpoint replay does not fit.")
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step_count = int(checkpoint["step_count"])
        self.transition_count = int(checkpoint["transition_count"])
        self.exploration_rng.bit_generator.state = checkpoint["exploration_rng"]
        self.replay_rng.setstate(checkpoint["replay_rng"])
        self.replay.items = checkpoint["replay_items"]
        self.replay.position = int(checkpoint["replay_position"]) % self.replay.capacity
