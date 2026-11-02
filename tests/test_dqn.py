import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from elfcomm.dqn import DQNAgent, TensorQNetwork


class _FixedQ(nn.Module):
    def __init__(self, q):
        super().__init__()
        self.register_buffer("q", torch.tensor(q, dtype=torch.float32))

    def forward(self, states):
        return self.q.expand(states.shape[0], -1)


class TensorDQNTests(unittest.TestCase):
    def make_agent(self, seed=4):
        return DQNAgent(5, (2, 2, 2, 2), {"batch_size": 4, "warmup_steps": 4, "replay_capacity": 32, "hidden_size": 16, "tensor_rank": 3}, seed)

    def populate(self, agent):
        rng = np.random.default_rng(123)
        for index in range(12):
            state = rng.normal(size=5).astype(np.float32)
            mask = rng.random(16) > 0.3
            if index % 3 == 0:
                mask[:] = False
            agent.remember(state, index % 16, float(index % 3 - 1), state * 0.9, index % 4 == 0, mask)

    def test_tensor_matches_explicit_cp_and_c_order(self):
        network = TensorQNetwork(5, (2, 3, 2, 4), rank=3, hidden=8)
        states = torch.randn(2, 5)
        values = network(states)
        self.assertEqual(tuple(values.shape), (2, 48))
        features = network.body(states)
        weights = network.weights_head(features)
        baseline = network.value_head(features)
        for indices in np.ndindex(network.action_shape):
            flat = np.ravel_multi_index(indices, network.action_shape)
            product = torch.ones(3)
            for factor, index in zip(network.factors, indices):
                product = product * factor[index]
            expected = baseline[:, 0] + (weights * product).sum(dim=-1)
            torch.testing.assert_close(values[:, flat], expected)

    def test_mask_and_evaluation_do_not_advance_rng(self):
        agent = self.make_agent()
        state = np.zeros(5, dtype=np.float32)
        mask = np.zeros(16, dtype=bool)
        self.assertEqual(agent.act(state, mask), -1)
        mask[7] = True
        rng_before = copy.deepcopy(agent.exploration_rng.bit_generator.state)
        for _ in range(10):
            self.assertEqual(agent.act(state, mask), 7)
        self.assertEqual(rng_before, agent.exploration_rng.bit_generator.state)
        for _ in range(10):
            self.assertEqual(agent.act(state, mask, epsilon=1.0), 7)

    def test_double_dqn_masks_argmax_and_blocks_terminal_bootstrap(self):
        agent = self.make_agent()
        agent.online = _FixedQ([1.0, 100.0] + [0.0] * 14)
        agent.target = _FixedQ([7.0, 80.0] + [0.0] * 14)
        states = torch.zeros(3, 5)
        masks = torch.zeros(3, 16, dtype=torch.bool)
        masks[0, 0] = True
        masks[1, :2] = True
        dones = torch.tensor([False, True, False])
        values = agent._bootstrap_values(states, dones, masks)
        torch.testing.assert_close(values, torch.tensor([7.0, 0.0, 0.0]))
        self.assertFalse(values.requires_grad)

    def test_learning_checkpoint_and_exact_resume(self):
        agent = self.make_agent()
        self.assertIsNone(agent.learn())
        self.populate(agent)
        before = [parameter.detach().clone() for parameter in agent.online.parameters()]
        loss = agent.learn()
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, agent.online.parameters())))
        self.assertTrue(all(parameter.grad is None for parameter in agent.target.parameters()))
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "agent.pt"
            agent.save(checkpoint, include_replay=True)
            restored = self.make_agent(seed=99)
            restored.load(checkpoint)
            states = torch.ones(3, 5)
            torch.testing.assert_close(agent.online(states), restored.online(states), rtol=0, atol=0)
            mask = np.ones(16, dtype=bool)
            for _ in range(5):
                self.assertEqual(agent.act(np.zeros(5), mask, 1.0), restored.act(np.zeros(5), mask, 1.0))
            self.assertEqual(agent.learn(), restored.learn())
            for a, b in zip(agent.online.parameters(), restored.online.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_default_checkpoint_omits_replay(self):
        agent = self.make_agent()
        self.populate(agent)
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "agent.pt"
            agent.save(checkpoint)
            restored = self.make_agent(seed=99)
            restored.load(checkpoint)
            self.assertEqual(len(restored.replay), 0)
            self.assertIsNone(restored.learn())

    def test_replay_copies_input_arrays(self):
        agent = self.make_agent()
        state = np.ones(5, dtype=np.float32)
        mask = np.ones(16, dtype=bool)
        agent.remember(state, 0, 1.0, state, False, mask)
        state[:] = 0
        mask[:] = False
        self.assertTrue(agent.replay.items[0][0].all())
        self.assertTrue(agent.replay.items[0][-1].all())


if __name__ == "__main__":
    unittest.main()
