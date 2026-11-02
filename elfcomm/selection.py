from __future__ import annotations

import numpy as np


class StableDecisionController:
    def __init__(self, cfg):
        self.settings = cfg.get("decision", {})
        self.previous_action = None
        self.pending_action = None
        self.pending_count = 0

    def select(self, env, agent, state, mask, candidate=None):
        candidate = int(agent.act(state, mask, epsilon=0.0) if candidate is None else candidate)
        previous = self.previous_action
        if candidate < 0:
            self.previous_action = None
            self.pending_action, self.pending_count = None, 0
            return -1, "no_feasible_action"
        if previous is None or not mask[previous]:
            self.previous_action = candidate
            self.pending_action, self.pending_count = None, 0
            return candidate, "initial_or_previous_infeasible"
        old, new = env.actions[previous], env.actions[candidate]
        selected, reason = previous, "retain_feasible_configuration"
        if new.tx_power_kw < old.tx_power_kw:
            selected, reason = candidate, "lower_feasible_power"
            self.pending_action, self.pending_count = None, 0
        elif new.tx_power_kw == old.tx_power_kw:
            gain = env.action_metrics(candidate)["goodput_bps_estimate"] - env.action_metrics(previous)["goodput_bps_estimate"]
            if gain > self.settings.get("same_power_goodput_gain_bps", 0.01):
                self.pending_count = self.pending_count + 1 if self.pending_action == candidate else 1
                self.pending_action = candidate
                reason = "confirming_same_power_improvement"
                if self.pending_count >= self.settings.get("same_power_confirmation_rounds", 1):
                    selected, reason = candidate, "confirmed_same_power_material_improvement"
                    self.pending_action, self.pending_count = None, 0
            else:
                self.pending_action, self.pending_count = None, 0
        else:
            self.pending_action, self.pending_count = None, 0
        self.previous_action = selected
        return selected, reason


def learning_rate_at(settings, phase_episode, phase_length):
    start = float(settings["learning_rate"])
    end = float(settings.get("learning_rate_final", start))
    fraction = min(1.0, max(0.0, (phase_episode - 1) / max(1, phase_length - 1)))
    return end + 0.5 * (start - end) * (1.0 + np.cos(np.pi * fraction))
