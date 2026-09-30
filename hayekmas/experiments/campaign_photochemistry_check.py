"""Check arithmetic and time dimensions in a saved photochemistry task."""

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from .campaign_client import atomic_json


def export(root):
    repository = Path(__file__).resolve().parents[2]
    dataset = repository / "third_party/benchmarks/frontier-science-research/data/research_test.jsonl"
    task = [json.loads(line) for line in dataset.read_text().splitlines()][8]
    result_path = next((root / "original/evaluation/epoch-1").glob("09-*/result.json"))
    result = json.loads(result_path.read_text())
    assert result["task_id"] == task["task_group_id"]
    flux, enhancement = Decimal("2e20"), Decimal("4e4")
    cross_sections = {785: Decimal("3e-18"), 532: Decimal("1e-18")}
    direct_rates = {785: Decimal("0.1"), 532: Decimal("0.02")}
    excitation = {w: flux * cross_sections[w] * enhancement for w in cross_sections}
    rubric_recipe = {w: excitation[w] * direct_rates[w] for w in cross_sections}
    linear_baseline_scaling = {w: enhancement * direct_rates[w] for w in cross_sections}
    baseline_population_times_yield = {w: direct_rates[w] / (flux * cross_sections[w]) for w in cross_sections}
    assert excitation == {785: Decimal("2.4e7"), 532: Decimal("8e6")}
    assert rubric_recipe[785] / rubric_recipe[532] == 15
    assert linear_baseline_scaling[785] / linear_baseline_scaling[532] == 5
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "task_id": result["task_id"],
        "selection": "Post hoc selection of an explicitly numerical task for a units check; observed answer and grade were known. Not representative sampling.",
        "scope": "Arithmetic and dimensional consistency only. No experiment, replacement grading, or aggregate-score correction.",
        "original_score_unchanged": result["score"],
        "reference": "https://goldbook.iupac.org/terms/view/Q04991",
        "effective_flux": str(flux * enhancement),
        "excitation_rates_per_second": {w: str(v) for w, v in excitation.items()},
        "rubric_recipe_numerical_values": {w: str(v) for w, v in rubric_recipe.items()},
        "rubric_recipe_ratio": "15",
        "time_dimension_check": {
            "excitation_rate_exponent": -1,
            "given_direct_rate_exponent": -1,
            "product_exponent": -2,
            "required_rate_exponent": -1,
            "consistent_as_written": False,
        },
        "conditional_linear_model": {
            "assumptions": "The given direct rates refer to the same ensemble, unsaturated charge-driven chemistry, and the enhancement changes excitation rate only; population and wavelength-specific yield remain fixed when enhancement is applied.",
            "population_times_yield_inferred_from_each_baseline": {w: str(v) for w, v in baseline_population_times_yield.items()},
            "enhanced_decomposition_rates_per_second": {w: str(v) for w, v in linear_baseline_scaling.items()},
            "ratio": "5",
            "status": "Illustrative conditional model, not a new benchmark target. Additional pathways or changed yields require more information.",
        },
        "source_hashes": {
            str(result_path.relative_to(root)): hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "dataset": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "task_rubric": hashlib.sha256(task["answer"].encode()).hexdigest(),
            "analysis_code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    atomic_json(root / "photochemistry-units-review.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(export(parser.parse_args().root)["time_dimension_check"], indent=2))
