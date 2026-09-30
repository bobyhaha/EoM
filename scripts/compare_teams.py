"""Run four matched-task conditions; label the singleton as an EoM-like ablation."""

import argparse
from pathlib import Path
import json
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hayekmas.adapters.teams.runtime import run
from hayekmas.adapters.teams.report import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="global_configs/teams_demo.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="7,17,29")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds):
        parser.error("seeds must be distinct")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    raw = json.loads(Path(args.config).read_text())
    # A comparison's configured USD budget is shared by partitioning it across cells.
    if raw.get("model", {}).get("api") == "openrouter":
        raw["budget"] = {**raw["budget"], "max_usd": raw["budget"]["max_usd"] / (4 * len(seeds))}
    results, aggregates = [], []
    for condition in ("individual", "random_fixed", "self_selected_fixed", "dynamic"):
        cell = []
        for seed in seeds:
            config = {**raw, "teams": {**raw.get("teams", {}), "condition": condition, "seed": seed}}
            result = run(config, out=out / f"{condition}-{seed}", plots=not args.no_plots)
            result["seed"] = seed
            results.append(result)
            cell.append(result["mean_score"])
            print(f"{condition} seed={seed} mean_score={cell[-1]:.3f}")
            write_json(out / "runs.json", results)
        aggregates.append(
            {
                "condition": condition,
                "independent_seeds": len(seeds),
                "mean_score": statistics.mean(cell),
                "seed_standard_deviation": statistics.stdev(cell) if len(cell) > 1 else None,
            }
        )
    write_json(
        out / "comparison.json",
        {
            "conditions": aggregates,
            "note": "Matched task seeds; inference costs differ and are reported per run. Singleton is EoM-like, not an original-paper reproduction.",
        },
    )


if __name__ == "__main__":
    main()
