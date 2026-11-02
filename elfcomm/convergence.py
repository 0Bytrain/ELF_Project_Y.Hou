from __future__ import annotations

import math

import numpy as np


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _flag(value):
    return str(value).lower() in ("true", "1", "1.0")


def tail_statistics(rows, window=10, tolerance=0.15, key="ber_estimate",
                    episodes_up_to=None, episodes_from=None, target_ber=None):
    if window < 2 or tolerance < 0:
        raise ValueError("Need window >= 2 and nonnegative stability tolerance")
    selected = [r for r in rows if r.get("evaluation_event") != "before_update"]
    if episodes_up_to is not None:
        selected = [r for r in selected if float(r["episode"]) <= episodes_up_to]
    if episodes_from is not None:
        selected = [r for r in selected if float(r["episode"]) >= episodes_from]
    # Do not search for a conveniently stable historical interval or discard
    # outages/missing samples from the final consecutive evaluation window.
    recent = selected[-window:]
    values = [_number(r.get(key)) for r in recent]
    result = {"window": window, "n": len(selected), "mean": None, "min": None,
              "max": None, "spread": None, "converged": False, "resolved": False,
              "target_met": False, "start": None, "end": None, "reason": "insufficient_samples"}
    if len(recent) < window or any(v is None for v in values):
        return result
    result.update(start=float(recent[0]["episode"]), end=float(recent[-1]["episode"]))
    mean = float(np.mean(values))
    result.update(mean=mean, min=min(values), max=max(values))
    no_outage = all((_number(r.get("outage_fraction", 0.0)) == 0) for r in recent)
    result["target_met"] = bool(target_ber is not None and no_outage
                                and all(v <= target_ber for v in values)
                                and all((_number(r.get("ber_target_met_fraction_estimate", 1)) == 1)
                                        for r in recent))
    limited = key == "ber_estimate" and any(
        v == 0 or _flag(r.get("ber_resolution_limited", False))
        or (_number(r.get("ber_resolution_limited_fraction", 0)) or 0) > 0
        for r, v in zip(recent, values))
    result["resolved"] = not limited
    if limited:
        result["reason"] = "below_table_resolution"
        return result
    if not no_outage:
        result["reason"] = "outage_in_tail"
        return result
    spread = (max(values) - min(values)) / mean if mean > 0 else None
    stable = spread is not None and spread <= tolerance
    result.update(spread=spread, converged=bool(stable), reason="stable_tail" if stable else "unstable_tail")
    return result


def policy_tail_statistics(rows, target_ber, window=20, settings=None):
    settings = settings or {}
    selected = [r for r in rows if r.get("evaluation_event") not in ("before_update", "untrained")]
    recent = selected[-window:]
    result = {"window": window, "n": len(selected), "converged": False,
              "target_met": False, "reason": "insufficient_samples", "mean": None,
              "min": None, "max": None, "resolved": False}
    if len(recent) < window:
        return result
    ber = [_number(r.get("ber_estimate")) for r in recent]
    power = [_number(r.get("mean_tx_power_kw")) for r in recent]
    rewards = [_number(r.get("mean_reward")) for r in recent]
    if any(v is None for values in (ber, power, rewards) for v in values):
        result["reason"] = "missing_observations"
        return result
    target_met = all(v <= target_ber and _number(r.get("outage_fraction")) == 0
                     and _number(r.get("ber_target_met_fraction_estimate")) == 1
                     for r, v in zip(recent, ber))
    signatures = [r.get("policy_signature") for r in recent]
    switch_fraction = (sum(a != b for a, b in zip(signatures, signatures[1:])) / (window - 1)
                       if all(signatures) else 1.0)
    power_stable = max(power) - min(power) <= settings.get("power_tolerance_kw", 0.0)
    reward_stable = max(rewards) - min(rewards) <= settings.get("reward_tolerance", 0.02)
    configuration_stable = switch_fraction <= settings.get("max_configuration_switch_fraction", 0.05)
    gaps = [_number(r.get("power_optimality_gap_kw")) for r in recent]
    minimum_verified = all(g is not None and abs(g) <= 1e-9 for g in gaps)
    resolved = not any(v == 0 or (_number(r.get("ber_resolution_limited_fraction")) or 0) > 0
                       for r, v in zip(recent, ber))
    exploration_finished = all(_number(r.get("training_epsilon")) == 0 for r in recent)
    exploration_ok = exploration_finished or not settings.get("require_exploration_finished", False)
    converged = target_met and power_stable and reward_stable and configuration_stable and minimum_verified and exploration_ok
    if not target_met:
        reason = "ber_target_or_transmission_not_met"
    elif not minimum_verified:
        reason = "minimum_model_power_not_reached_or_unknown"
    elif not exploration_ok:
        reason = "training_exploration_not_finished_in_observation_window"
    elif not (power_stable and reward_stable and configuration_stable):
        reason = "policy_still_changing"
    else:
        reason = "estimated_minimum_power_policy_stable" if resolved else "estimated_policy_stable_but_ber_unresolved"
    result.update(converged=converged, target_met=target_met, resolved=resolved,
                  estimated_policy_converged=converged,
                  exploration_finished=exploration_finished,
                  reliability_certified=False,
                  mean=float(np.mean(ber)), min=min(ber), max=max(ber),
                  start=float(recent[0]["episode"]), end=float(recent[-1]["episode"]),
                  mean_tx_power_kw=float(np.mean(power)), power_stable=power_stable,
                  reward_stable=reward_stable, configuration_switch_fraction=switch_fraction,
                  minimum_power_verified_in_model=minimum_verified,
                  reason=reason)
    return result


def candidate_lock_statistics(rows, target_ber, settings):
    options = dict(settings, max_configuration_switch_fraction=1.0)
    stats = policy_tail_statistics(rows, target_ber,
                                   settings.get("candidate_lock_evaluations", 5), options)
    stats["ready"] = stats.pop("converged")
    return stats


def phase_assessment(rows, shift_episode, target_ber, window=10, tolerance=0.15, policy_settings=None):
    if policy_settings is not None:
        pre = policy_tail_statistics([r for r in rows if float(r["episode"]) <= shift_episode],
                                     target_ber, window, policy_settings)
        post = policy_tail_statistics([r for r in rows if float(r["episode"]) > shift_episode],
                                      target_ber, window, policy_settings)
        return {"pre": pre, "post": post,
                "both_phases_target_met_estimate": pre["target_met"] and post["target_met"],
                "criterion_met": pre["converged"] and post["converged"]}
    pre = tail_statistics(rows, window, tolerance, episodes_up_to=shift_episode,
                          target_ber=target_ber)
    post = tail_statistics(rows, window, tolerance, episodes_from=shift_episode + 1,
                           target_ber=target_ber)
    return {"pre": pre, "post": post,
            "both_phases_target_met_estimate": pre["target_met"] and post["target_met"],
            "criterion_met": pre["converged"] and post["converged"]
                             and pre["target_met"] and post["target_met"]}
