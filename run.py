from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from elfcomm.config import ROOT, load_config, quick_config, resolve_path, save_json, validate_config


def parse_args():
    parser = argparse.ArgumentParser(description="ELF relative-link Polar / tensor DQN simulator")
    parser.add_argument("command", choices=["run", "build-table", "train", "decide", "validate", "sensitivity", "plot", "demo", "antenna-check"])
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    parser.add_argument("--quick", action="store_true", help="N64/L2 tiny Monte Carlo smoke profile, not reliability evidence")
    parser.add_argument("--table", help="BER table path; separates incompatible configurations")
    parser.add_argument("--checkpoint")
    parser.add_argument("--scenario", help="JSON containing environment overrides and occupied_bands")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--min-frames", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--noise-asd", type=float, help="Environmental ASD at the 30 Hz reference frequency in pT/sqrt(Hz); receiver RMS remains independent")
    parser.add_argument("--depth", type=float)
    parser.add_argument("--distance", type=float, help="Horizontal distance in metres (formula model requires >= 1200000)")
    parser.add_argument("--ice-thickness", type=float)
    parser.add_argument("--ice-temperature", type=float)
    parser.add_argument("--water-conductivity", type=float)
    parser.add_argument("--signal-scale-db", type=float,
                        help="Scale reference_signal_rms_pt and the field anchor by 10^(dB/20) for sensitivity")
    parser.add_argument("--source-model", choices=["anchor", "current_moment"],
                        help="Override link.source_model for source-model comparison")
    parser.add_argument("--antenna-unit", choices=["si", "engineering"],
                        help="Override antenna.unit_convention for the optional current_moment model")
    return parser.parse_args()


def scenario_from_args(cfg, args):
    env = copy.deepcopy(cfg["link"]["reference_environment"])
    # None means "let the environment sample random bands".  An explicit empty
    # list means "manual spectrum is empty"; the two must not be conflated, or
    # every CLI training run silently disables random occupancy.
    configured_bands = copy.deepcopy(cfg["environment"]["occupied_bands"])
    bands = configured_bands if configured_bands else None
    if args.scenario:
        source = json.loads(resolve_path(args.scenario).read_text(encoding="utf-8-sig"))
        env.update(source.get("environment", {}))
        if "occupied_bands" in source:
            bands = copy.deepcopy(source["occupied_bands"])
    for name, key in [("noise_asd", "environment_noise_asd_pt_rt_hz"), ("depth", "receiver_depth_m"),
                      ("distance", "horizontal_distance_m"), ("ice_thickness", "ice_thickness_m"),
                      ("ice_temperature", "ice_temperature_c"), ("water_conductivity", "seawater_conductivity_s_m")]:
        value = getattr(args, name)
        if value is not None:
            env[key] = value
    for key, (low, high) in cfg["environment"]["ranges"].items():
        if not low <= env[key] <= high:
            raise SystemExit(f"Scenario {key}={env[key]} is outside configured range [{low}, {high}]")
    return env, bands


def scenario_is_explicit(args):
    """True when the user actually supplied a scenario/CLI environment.

    Without this distinction run.py always supplies the reference environment,
    which would accidentally turn random-environment training into fixed-channel
    training even when no scenario was requested.
    """
    keys = ("noise_asd", "depth", "distance", "ice_thickness", "ice_temperature",
            "water_conductivity")
    return bool(args.scenario or any(getattr(args, key) is not None for key in keys))


def decision(cfg, table, agent, scenario, bands):
    from elfcomm.environment import CommunicationEnvironment

    env = CommunicationEnvironment(cfg, table, seed=cfg["seed"] + 200000)
    state, mask = env.reset(environment=scenario, occupied_bands=bands, dynamic=False)
    from elfcomm.selection import StableDecisionController
    action, selection_reason = StableDecisionController(cfg).select(env, agent, state, mask)
    _, _, _, _, result = env.step(action)
    channel = env.channel
    gain_frequency = (env.actions[action].frequency_hz if action >= 0
                      else cfg["link"]["reference_frequency_hz"])
    calibration_mode = ("relative_attenuation_anchor" if channel.source_model == "anchor"
                        else "current_moment_source")
    result.update(calibration_mode=calibration_mode,
                  source_model=channel.source_model,
                  propagation_model=channel.propagation_model,
                  reference_signal_rms_pt=channel.reference_signal_rms_pt,
                  reference_snr_db=channel.reference_snr_db,
                  reference_tx_power_kw=cfg["link"]["reference_tx_power_kw"],
                  distance_gain_frequency_hz=float(gain_frequency),
                  distance_gain_db=float(channel.distance_gain_db(gain_frequency,
                                                                  scenario["horizontal_distance_m"])),
                  legal_action_count=int(env.spectral_mask.sum()), feasible_action_count=int(mask.sum()),
                  selection_reason=selection_reason, run_label=cfg.get("run_label", "relative_link_simulation"))
    if action >= 0:
        ber_ok = bool(result["meets_ber_target_estimate"])
        rate_ok = bool(result["meets_rate_target_estimate"])
        if cfg["objective"].get("mode") == "ber_constrained_min_power":
            status = ("estimated_ber_target_met_power_preference_unverified" if ber_ok
                      else "estimated_ber_target_not_met")
        elif ber_ok and rate_ok:
            status = "estimated_targets_met_unverified"
        elif ber_ok:
            status = "estimated_ber_target_met_rate_target_not_met"
        elif rate_ok:
            status = "estimated_rate_target_met_ber_target_not_met"
        else:
            status = "estimated_targets_not_met"
        result["decision_status"] = status
    else:
        result["decision_status"] = ("ber_target_unreachable_no_transmission" if env.spectral_mask.any()
                                     else "no_legal_action_no_transmission")
    return result


def antenna_check(cfg):
    """Compare the anchor and the optional current-moment source model."""
    import numpy as np

    from elfcomm.antenna import AntennaModel
    from elfcomm.channel import magnetic_transfer

    link = cfg["link"]
    antenna = AntennaModel(link)
    env = link["reference_environment"]
    local = abs(magnetic_transfer(link["reference_frequency_hz"], env, link))
    sensor_psd = link["receiver_noise_rms_pt"] ** 2 / link["reference_noise_bandwidth_hz"]
    noise_psd = sensor_psd + env["environment_noise_asd_pt_rt_hz"] ** 2

    def row(unit, field_uv):
        air_pt = field_uv * 1e-6 / 299792458.0 * 1e12
        rx_pt = air_pt * local
        return {"unit": unit, "e_db_uv_m": float(20 * np.log10(field_uv)),
                "b_air_pt": float(air_pt), "rx_field_rms_pt": float(rx_pt),
                "snr_db": float(10 * np.log10(rx_pt ** 2 / (noise_psd * link["reference_noise_bandwidth_hz"])))}

    reference = {
        "frequency_hz": link["reference_frequency_hz"],
        "distance_m": env["horizontal_distance_m"],
        "power_kw": link["reference_tx_power_kw"],
        "local_transfer": float(local),
        "anchor": {"unit": "anchor", "e_db_uv_m": link["field_anchor_e_db_uv_m"],
                   "b_air_pt": float(10 ** (link["field_anchor_e_db_uv_m"] / 20) * 1e-6 / 299792458.0 * 1e12),
                   "rx_field_rms_pt": float(link["reference_signal_rms_pt"]),
                   "snr_db": float(10 * np.log10(link["reference_signal_rms_pt"] ** 2 / (noise_psd * link["reference_noise_bandwidth_hz"])))},
        "si": row("si", antenna.reference_field_uv_m("si")),
        "engineering": row("engineering", antenna.reference_field_uv_m("engineering")),
        "feed_current_a": float(antenna.feed_current_a(link["reference_frequency_hz"], link["reference_tx_power_kw"])),
        "input_resistance_ohm": float(antenna.input_resistance_ohm(link["reference_frequency_hz"])),
        "radiation_resistance_ohm": float(antenna.radiation_resistance_ohm(link["reference_frequency_hz"])),
        "radiation_efficiency": float(antenna.radiation_efficiency(link["reference_frequency_hz"])),
        "radiated_power_w": float(antenna.radiated_power_w(link["reference_frequency_hz"], link["reference_tx_power_kw"])),
        "effective_length_m": float(antenna.effective_length_m),
    }
    frequency_rows = []
    for frequency in cfg["actions"]["frequency_hz"]:
        frequency_rows.append({
            "frequency_hz": float(frequency),
            "feed_current_a": float(antenna.feed_current_a(frequency, link["reference_tx_power_kw"])),
            "current_moment_a_m": float(antenna.current_moment_a_m(frequency, link["reference_tx_power_kw"])),
            "input_resistance_ohm": float(antenna.input_resistance_ohm(frequency)),
            "radiation_resistance_ohm": float(antenna.radiation_resistance_ohm(frequency)),
            "radiation_efficiency": float(antenna.radiation_efficiency(frequency)),
            "radiated_power_w": float(antenna.radiated_power_w(frequency, link["reference_tx_power_kw"])),
            "e_si_db_uv_m": float(20 * np.log10(antenna.field_uv_m(frequency, env["horizontal_distance_m"], unit_convention="si"))),
            "e_engineering_db_uv_m": float(20 * np.log10(antenna.field_uv_m(frequency, env["horizontal_distance_m"], unit_convention="engineering"))),
        })
    distance_rows = []
    for distance in (100000.0, 1200000.0, 1950000.0, 2950000.0):
        distance_rows.append({
            "distance_m": distance,
            "e_si_db_uv_m": float(20 * np.log10(antenna.field_uv_m(link["reference_frequency_hz"], distance, unit_convention="si"))),
            "e_engineering_db_uv_m": float(20 * np.log10(antenna.field_uv_m(link["reference_frequency_hz"], distance, unit_convention="engineering"))),
        })
    return {
        "source_model": link.get("source_model", "anchor"),
        "antenna": link["antenna"],
        "reference": reference,
        "frequency_sweep": frequency_rows,
        "distance_sweep": distance_rows,
        "unit_note": "si and engineering are both converted to the same uV/m field; the engineering branch applies 1/sqrt(1000)",
    }


def validate_decision(cfg, result):
    from elfcomm.ber import simulate_link

    if not result["transmitted"]:
        return {"decision": result, "status": "no_transmission", "reliability_certified": False}
    settings = dict(cfg["validation"])
    settings["min_frames"] = settings["max_frames"]
    print(f"Independent {result.get('coding_scheme', 'Polar')} validation: {result['modulation']}, "
          f"Es/N0={result['esn0_db']:.2f} dB, max {settings['max_frames']} frames", flush=True)
    measured = simulate_link(result.get("coding_scheme", "Polar"), result["modulation"],
                             result["code_rate"], result["esn0_db"], **cfg["polar"],
                             seed=cfg["seed"] + 900001, **settings)
    conservative_upper = measured["fer_upper95"]
    certified = conservative_upper <= cfg["objective"]["target_ber"]
    return {"decision": result, "measurement": measured,
            "measured_goodput_bps": result["payload_bits"] * (1 - measured["fer"]) / result["frame_duration_s"],
            "conservative_ber_upper95_via_fer": conservative_upper,
            "reliability_certified": bool(certified),
            "certificate_scope": "selected_action_including_coding_scheme_code_rate_tx_power_and_waveform_mode_in_fixed_simulated_environment_and_receiver_model_only",
            "status": "target_supported_by_conservative_bound" if certified else "target_not_certified_with_this_sample_budget"}


def main():
    args = parse_args()
    if args.command == "run" and not args.scenario:
        raise SystemExit("run 命令必须通过 --scenario 指定占用频段文件")      #不满足if条件就显示这段报错。run必须带--scenario
    cfg = load_config(args.config)                                      #加载默认配置，若不给出路径就走默认路径
    if args.quick or args.command == "demo":                            #若是demo指令或者quick模式就加载quick配置
        cfg = quick_config(cfg)
    if args.output:
        cfg["output_dir"] = args.output
    if args.table:
        cfg["ber_table"]["path"] = args.table
    elif args.output and args.command in ("build-table", "demo", "run") and not cfg["ber_table"].get("existing_only", False):
        cfg["ber_table"]["path"] = str(Path(args.output) / "ber_table_cc.json")
    if args.episodes:
        cfg["training"]["episodes"] = args.episodes
    if args.source_model:
        cfg["link"]["source_model"] = args.source_model
    if args.antenna_unit:
        cfg["link"]["antenna"]["unit_convention"] = args.antenna_unit
    budget = cfg["ber_table"] if args.command == "build-table" else cfg["validation"]
    if args.min_frames is not None:
        budget["min_frames"] = args.min_frames
    if args.max_frames is not None:
        budget["max_frames"] = args.max_frames
    if args.signal_scale_db is not None:
        if cfg["link"].get("source_model", "anchor") != "anchor":
            raise SystemExit("--signal-scale-db applies only to source_model='anchor'")
        # 同时缩放参考信号与锚点，保持锚点自检一致
        cfg["link"]["reference_signal_rms_pt"] *= 10 ** (args.signal_scale_db / 20)
        cfg["link"]["field_anchor_e_db_uv_m"] += args.signal_scale_db
    validate_config(cfg)
    output = resolve_path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "antenna-check":
        result = antenna_check(cfg)
        save_json(output / "antenna_check.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"Antenna check saved to {output / 'antenna_check.json'}")
        return
    if args.command == "run":
        from elfcomm.performance import load_or_build_table
        scenario, bands = scenario_from_args(cfg, args)                     #场景+占用频段
        table_path = resolve_path(args.table or cfg["ber_table"]["path"])
        table = load_or_build_table(cfg, table_path)                       #有表直接用，无表自动建
        from elfcomm.training import train, load_agent
        train(cfg, table, scenario=scenario if scenario_is_explicit(args) else None,
              occupied_bands=bands)
        agent = load_agent(cfg, table)                                       #加载最好的模型
        result = decision(cfg, table, agent, scenario, bands)                #做决策
        save_json(output / "decision.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"All artifacts saved to {output}")
        return
    if args.command == "plot":
        from elfcomm.plots import plot_training
        plot_training(output, cfg["training"]["ema_alpha"], cfg["objective"]["target_ber"])
        print(f"Plots saved to {output}")
        return
    from elfcomm.performance import build_table, load_or_build_table
    if args.command in ("build-table", "demo"):
        table = build_table(cfg)
        from elfcomm.plots import plot_table
        plot_table(table, output)
        save_json(output / "table_config.json", cfg)
        if args.command == "build-table":
            print(f"BER table saved: {table.path}")
            return
    else:
        table = load_or_build_table(cfg)
    if args.command in ("train", "demo"):
        from elfcomm.training import train
        scenario, bands = scenario_from_args(cfg, args)
        train(cfg, table, scenario=scenario if scenario_is_explicit(args) else None,
              occupied_bands=bands)
        if args.command == "train":
            print(f"Training artifacts saved: {output}")
            return
    from elfcomm.training import load_agent, write_csv
    agent = load_agent(cfg, table, args.checkpoint)
    scenario, bands = scenario_from_args(cfg, args)
    result = decision(cfg, table, agent, scenario, bands)
    if args.command == "sensitivity":
        rows = []
        for scale_db in [-6, -3, 0, 3, 6]:
            changed = copy.deepcopy(cfg)
            changed["link"]["reference_signal_rms_pt"] *= 10 ** (scale_db / 20)
            changed["link"]["field_anchor_e_db_uv_m"] += scale_db
            row = decision(changed, table, agent, scenario, bands)
            row["signal_scale_db"] = scale_db
            rows.append(row)
        write_csv(output / "reference_signal_sensitivity.csv", rows)
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    save_json(output / "decision.json", result)
    if args.command in ("validate", "demo"):
        report = validate_decision(cfg, result)
        save_json(output / "validation_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
