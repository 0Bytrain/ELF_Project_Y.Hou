from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from elfcomm.config import ROOT, load_config, quick_config, resolve_path, save_json, validate_config


def parse_args():
    parser = argparse.ArgumentParser(description="ELF relative-link Polar / tensor DQN simulator")
    parser.add_argument("command", choices=["run", "build-table", "train", "decide", "validate", "sensitivity", "plot", "demo"])
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    parser.add_argument("--quick", action="store_true", help="N64/L2 tiny Monte Carlo smoke profile, not reliability evidence")
    parser.add_argument("--table", help="BER table path; separates incompatible configurations")
    parser.add_argument("--checkpoint")
    parser.add_argument("--scenario", help="JSON containing environment overrides and occupied_bands")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--min-frames", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--noise-asd", type=float, help="Environmental ASD at 5Hz in pT/sqrt(Hz); receiver RMS remains independent")
    parser.add_argument("--depth", type=float)
    parser.add_argument("--distance", type=float)
    parser.add_argument("--ice-thickness", type=float)
    parser.add_argument("--ice-temperature", type=float)
    parser.add_argument("--water-conductivity", type=float)
    parser.add_argument("--reference-snr", type=float, help="Explicit sensitivity experiment; assumed reference SNR in 1Hz")
    return parser.parse_args()


def scenario_from_args(cfg, args):
    env = copy.deepcopy(cfg["link"]["reference_environment"])
    bands = copy.deepcopy(cfg["environment"]["occupied_bands"])
    if args.scenario:
        source = json.loads(resolve_path(args.scenario).read_text(encoding="utf-8-sig"))
        env.update(source.get("environment", {}))
        bands = source.get("occupied_bands", bands)
    for name, key in [("noise_asd", "environment_noise_asd_pt_rt_hz"), ("depth", "receiver_depth_m"),
                      ("distance", "horizontal_distance_m"), ("ice_thickness", "ice_thickness_m"),
                      ("ice_temperature", "ice_temperature_c"), ("water_conductivity", "seawater_conductivity_s_m")]:
        value = getattr(args, name)
        if value is not None:
            env[key] = value
    return env, bands


def decision(cfg, table, agent, scenario, bands):
    from elfcomm.environment import CommunicationEnvironment

    env = CommunicationEnvironment(cfg, table, seed=cfg["seed"] + 200000)
    state, mask = env.reset(environment=scenario, occupied_bands=bands, dynamic=False)
    action = agent.act(state, mask, epsilon=0.0)
    _, _, _, _, result = env.step(action)
    result.update(calibration_mode="relative_assumption", reference_snr_db=cfg["link"]["reference_snr_db"],
                  fixed_tx_power_kw_metadata=cfg["link"]["fixed_tx_power_kw"],
                  legal_action_count=int(mask.sum()), run_label=cfg.get("run_label", "relative_link_simulation"))
    if action >= 0:
        result["decision_status"] = "estimated_target_met_unverified" if result["meets_ber_target_estimate"] else "estimated_target_not_met"
    else:
        result["decision_status"] = "no_legal_action_no_transmission"
    return result


def validate_decision(cfg, result):
    from elfcomm.ber import simulate_link

    if not result["transmitted"]:
        return {"decision": result, "status": "no_transmission", "reliability_certified": False}
    settings = dict(cfg["validation"])
    settings["min_frames"] = settings["max_frames"]
    print(f"Independent Polar validation: {result['modulation']}, Es/N0={result['esn0_db']:.2f} dB, "
          f"max {settings['max_frames']} frames", flush=True)
    measured = simulate_link(result["modulation"], result["code_rate"], result["esn0_db"], **cfg["polar"],
                             seed=cfg["seed"] + 900001, **settings)
    conservative_upper = measured["fer_upper95"]
    certified = conservative_upper <= cfg["objective"]["target_ber"]
    return {"decision": result, "measurement": measured,
            "measured_goodput_bps": result["payload_bits"] * (1 - measured["fer"]) / result["frame_duration_s"],
            "conservative_ber_upper95_via_fer": conservative_upper,
            "reliability_certified": bool(certified),
            "certificate_scope": "selected_action_in_fixed_simulated_environment_and_receiver_model_only",
            "status": "target_supported_by_conservative_bound" if certified else "target_not_certified_with_this_sample_budget"}


def main():
    args = parse_args()
    if args.command == "run" and not args.scenario:
        raise SystemExit("run 命令必须通过 --scenario 指定占用频段文件")
    cfg = load_config(args.config)
    if args.quick or args.command == "demo":
        cfg = quick_config(cfg)
    if args.output:
        cfg["output_dir"] = args.output
    if args.table:
        cfg["ber_table"]["path"] = args.table
    if args.episodes:
        cfg["training"]["episodes"] = args.episodes
    budget = cfg["ber_table"] if args.command == "build-table" else cfg["validation"]
    if args.min_frames is not None:
        budget["min_frames"] = args.min_frames
    if args.max_frames is not None:
        budget["max_frames"] = args.max_frames
    if args.reference_snr is not None:
        cfg["link"]["reference_snr_db"] = args.reference_snr
    validate_config(cfg)
    output = resolve_path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "run":
        from elfcomm.performance import PerformanceTable, build_table
        scenario, bands = scenario_from_args(cfg, args)
        table_path = resolve_path(args.table or cfg["ber_table"]["path"])
        if table_path.exists():
            table = PerformanceTable(cfg, table_path)
        else:
            print("BER table not found; building it automatically...", flush=True)
            table = build_table(cfg, table_path)
        from elfcomm.training import train, load_agent
        train(cfg, table, scenario=scenario, occupied_bands=bands)
        agent = load_agent(cfg, table)
        result = decision(cfg, table, agent, scenario, bands)
        save_json(output / "decision.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"All artifacts saved to {output}")
        return
    if args.command == "plot":
        from elfcomm.plots import plot_training
        plot_training(output, cfg["training"]["ema_alpha"], cfg["objective"]["target_ber"])
        print(f"Plots saved to {output}")
        return
    from elfcomm.performance import PerformanceTable, build_table
    if args.command in ("build-table", "demo"):
        table = build_table(cfg)
        from elfcomm.plots import plot_table
        plot_table(table, output)
        save_json(output / "table_config.json", cfg)
        if args.command == "build-table":
            print(f"BER table saved: {table.path}")
            return
    else:
        table = PerformanceTable(cfg)
    if args.command in ("train", "demo"):
        from elfcomm.training import train
        scenario, bands = scenario_from_args(cfg, args)
        train(cfg, table, scenario=scenario, occupied_bands=bands)
        if args.command == "train":
            print(f"Training artifacts saved: {output}")
            return
    from elfcomm.training import load_agent, write_csv
    agent = load_agent(cfg, table, args.checkpoint)
    scenario, bands = scenario_from_args(cfg, args)
    result = decision(cfg, table, agent, scenario, bands)
    if args.command == "sensitivity":
        rows = []
        for snr_db in [5, 10, 15, 20]:
            changed = copy.deepcopy(cfg)
            changed["link"]["reference_snr_db"] = snr_db
            rows.append(decision(changed, table, agent, scenario, bands))
        write_csv(output / "reference_snr_sensitivity.csv", rows)
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
