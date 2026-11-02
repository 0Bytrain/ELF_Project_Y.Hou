from __future__ import annotations

import shutil
import sys
import json

import plot_shift_recovery
from elfcomm.config import fingerprint, load_config, resolve_path, save_json
from elfcomm.performance import load_or_build_table
from elfcomm.training import train


CONFIGS = (
    "configs/shift_noise.json",
    "configs/shift_distance.json",
    "configs/shift_depth.json",
)


def main(config=None):
    paths = CONFIGS if config is None else (config,)
    configs = [load_config(resolve_path(path)) for path in paths]
    if len(configs) > 1:
        shared = ("seed", "fixed_environment", "actions", "link", "environment", "objective", "training",
                  "polar", "ber_table", "validation")

        def baseline_contract(cfg):
            return {**{key: cfg[key] for key in shared},
                    "change_schedule": {key: value for key, value in cfg["channel_shift"].items()
                                        if key != "environment"}}

        baseline = fingerprint(baseline_contract(configs[0]))
        if any(fingerprint(baseline_contract(cfg)) != baseline for cfg in configs[1:]):
            raise ValueError("Controlled experiments must share baseline environment, spectrum, seed and training settings")
        if len({cfg["output_dir"] for cfg in configs}) != len(configs):
            raise ValueError("Each controlled experiment needs its own output directory")
    for cfg in configs:
        output = resolve_path(cfg["output_dir"])
        experiment = cfg.get("controlled_experiment", {})
        variable = experiment.get("variable")
        before = cfg["fixed_environment"]
        after = cfg["channel_shift"]["environment"]
        print(f"Experiment: {experiment.get('label', cfg.get('run_label'))}", flush=True)
        if variable:
            print(f"Only {variable}: {before[variable]} -> {after[variable]} {experiment['unit']}", flush=True)
        print(f"Estimated BER target in both phases: <= {cfg['objective']['target_ber']:g}; "
              "then minimize transmit power", flush=True)
        print(f"Manual spectrum: {cfg['environment']['occupied_bands']}", flush=True)
        print(f"At most {cfg['training']['episodes']} episodes; change after "
              f"{cfg['channel_shift']['episode']}; no within-episode drift in controlled experiments", flush=True)
        table = load_or_build_table(cfg)
        train(cfg, table)
        plot_shift_recovery.main(["--directory", str(output)])
        figure_name = plot_shift_recovery.figure_filename(cfg)
        if experiment:
            parent = output.parent
            root_name = f"training_ber_curve_{experiment['name']}.png"
            named_figure = parent / root_name
            shutil.copy2(output / figure_name, named_figure)
            index = parent / "experiment_index.json"
            previous = json.loads(index.read_text(encoding="utf-8")) if index.exists() else []
            results = [r for r in previous if r["name"] not in (experiment["name"], "conductivity")
                       and r.get("variable") != "seawater_conductivity_s_m"]
            results.append({"name": experiment["name"], "variable": variable,
                            "before": before[variable], "after": after[variable],
                            "unit": experiment["unit"], "target_ber": cfg["objective"]["target_ber"],
                            "output_directory": str(output), "figure": str(named_figure)})
            save_json(index, results)
        print(f"Finished: {output / figure_name}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
