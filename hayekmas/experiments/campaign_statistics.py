"""Paired task resampling; uncertainty is conditional on the trained populations."""

import random


def percentile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def paired_intervals(tasks, required=19, samples=20000, seed=20260930):
    if len(tasks) != required:
        return None
    score_differences = [r["teams"] - r["original"] for r in tasks]
    pass_differences = [int(r["teams_pass"]) - int(r["original_pass"]) for r in tasks]
    rng = random.Random(seed)
    score_means, pass_means = [], []
    for _ in range(samples):
        indices = rng.choices(range(len(tasks)), k=len(tasks))
        score_means.append(sum(score_differences[i] for i in indices) / len(tasks))
        pass_means.append(sum(pass_differences[i] for i in indices) / len(tasks))
    return {
        "method": "paired task percentile bootstrap",
        "confidence": 0.95,
        "resamples": samples,
        "seed": seed,
        "score_difference": [percentile(score_means, 0.025), percentile(score_means, 0.975)],
        "pass_rate_difference": [percentile(pass_means, 0.025), percentile(pass_means, 0.975)],
        "interpretation": "Conditional on these trained populations and observed judge outputs; does not measure training-seed or provider variance.",
    }
