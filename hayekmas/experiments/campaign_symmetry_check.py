"""A deterministic algebra check for the held-out C5/TICA task; no grading."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .campaign_client import atomic_json


def eigenvalues(lagged, covariance):
    values, vectors = np.linalg.eigh(covariance)
    assert min(values) > 0
    inverse_sqrt = (vectors * (1 / np.sqrt(values))) @ vectors.conj().T
    return np.linalg.eigvalsh(inverse_sqrt @ lagged @ inverse_sqrt)


def check():
    # A constructed example, not receptor trajectory data or a benchmark answer.
    p, n = 2, 5
    within = np.array([[2.0, 0.4], [0.4, 1.0]])
    diagonal = np.array([[0.6, 0.12], [0.12, 0.25]])
    next_block = np.array([[0.03, 0.06], [-0.02, 0.04]])
    second_block = np.array([[0.02, 0.0], [0.01, -0.01]])
    blocks = [diagonal, next_block, second_block, second_block.T, next_block.T]
    covariance = np.kron(np.eye(n), within)
    lagged = np.block([[blocks[(j - i) % n] for j in range(n)] for i in range(n)])
    koopman = np.linalg.solve(covariance, lagged)
    rotation = np.kron(np.roll(np.eye(n), 1, axis=0), np.eye(p))
    fourier = np.exp(-2j * np.pi * np.outer(np.arange(n), np.arange(n)) / n) / np.sqrt(n)
    transform = np.kron(fourier, np.eye(p))
    transformed_covariance = transform @ covariance @ transform.conj().T
    transformed_lagged = transform @ lagged @ transform.conj().T
    mask = np.kron(np.eye(n), np.ones((p, p)))
    split = []
    for k in range(n):
        section = slice(k * p, (k + 1) * p)
        split.extend(eigenvalues(transformed_lagged[section, section], transformed_covariance[section, section]))
    full = eigenvalues(lagged, covariance)
    embedding = np.kron(np.ones((n, 1)) / np.sqrt(n), np.eye(p))
    collapsed = embedding.T @ koopman @ embedding
    block_row_sum = sum(koopman[:p, k * p:(k + 1) * p] for k in range(n))
    group_sum = sum(np.linalg.matrix_power(rotation, k) @ koopman @ np.linalg.matrix_power(rotation.T, k) for k in range(n))
    transition = koopman.T
    innovation_covariance = covariance - transition @ covariance @ transition.T
    errors = {
        "lagged_covariance_symmetry": np.linalg.norm(lagged - lagged.T),
        "koopman_metric_self_adjoint": np.linalg.norm(covariance @ koopman - koopman.T @ covariance),
        "cyclic_commutator": np.linalg.norm(rotation @ koopman - koopman @ rotation),
        "fourier_off_block_norm": np.linalg.norm(transformed_lagged * (1 - mask)),
        "full_vs_sector_eigenvalues_max_error": max(abs(full - np.sort(split))),
        "invariant_projection_vs_row_sum": np.linalg.norm(collapsed - block_row_sum),
        "group_sum_minus_five_times_koopman": np.linalg.norm(group_sum - 5 * koopman),
    }
    assert all(value < 1e-12 for value in errors.values())
    assert np.linalg.eigvalsh(innovation_covariance).min() > 0
    assert np.linalg.norm(koopman - koopman.T) > 0.01
    assert max(full) < 1
    return {
        "constructed_example": True,
        "numpy_version": np.__version__,
        "covariance": covariance.tolist(),
        "lagged_covariance": lagged.tolist(),
        "full_generalized_eigenvalues": full.tolist(),
        "numerical_errors": {name: float(value) for name, value in errors.items()},
        "koopman_euclidean_asymmetry_norm": float(np.linalg.norm(koopman - koopman.T)),
        "minimum_innovation_covariance_eigenvalue": float(np.linalg.eigvalsh(innovation_covariance).min()),
        "full_group_sum_shape": list(group_sum.shape),
        "invariant_sector_shape": list(collapsed.shape),
    }


def export(root):
    repository = Path(__file__).resolve().parents[2]
    dataset = repository / "third_party/benchmarks/frontier-science-research/data/research_test.jsonl"
    task = [json.loads(line) for line in dataset.read_text().splitlines()][3]
    result_path = next((root / "teams/evaluation/epoch-1").glob("04-*/result.json"))
    result = json.loads(result_path.read_text())
    assert task["task_group_id"] == result["task_id"]
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "selection": "Post hoc review of the primary task where only the team submitted. Chosen after results were known; not representative sampling.",
        "task_id": result["task_id"],
        "original_team_score_unchanged": result["score"],
        "scope": "Constructed linear-algebra sanity check, not receptor simulation, expert scientific validation, or a replacement score.",
        "primary_source": "https://www.nature.com/articles/s41467-024-53170-z",
        "primary_source_mirror": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11489734/",
        "calculation": check(),
        "source_hashes": {
            str(result_path.relative_to(root)): hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "dataset": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "task_rubric": hashlib.sha256(task["answer"].encode()).hexdigest(),
            "analysis_code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    atomic_json(root / "symmetry-algebra-review.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    report = export(parser.parse_args().root)
    print(json.dumps(report["calculation"]["numerical_errors"], indent=2))
