"""Export reproducible research figures from the public report snapshot; no API calls."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"original": "#3e62bd", "teams": "#168174"}
NAMES = {"original": "Original EoM", "teams": "Voluntary teams"}


def export_grading_variability(data, output, timestamp):
    """Plot every recorded grade, preserving primary scores and missingness."""
    if not any(v.get("grader_recheck", {}).get("started") for v in data["arms"].values()):
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 10), layout="constrained")
    for ax, arm in zip(axes, ("original", "teams"), strict=True):
        check = data["arms"][arm]["grader_recheck"]
        repetitions = check["replicates"]
        source = repetitions[0]["tasks"]
        for position, task in enumerate(source):
            if not task["has_final_answer"]:
                ax.text(0.025, position, "No submitted answer · fixed zero", fontsize=8, color="#76828c", va="center")
                continue
            ax.scatter(
                task["primary_score"],
                position - 0.24,
                marker="D",
                s=32,
                facecolors="none",
                edgecolors="#202b37",
                zorder=3,
            )
            for repetition, shift in zip(repetitions, (-0.06, 0.12, 0.30), strict=True):
                grade = next(row for row in repetition["tasks"] if row["task_id"] == task["task_id"])
                if grade["score"] is not None:
                    ax.scatter(grade["score"], position + shift, color=COLORS[arm], s=23, zorder=3)
                elif grade["grade_attempted"]:
                    ax.text(1.015, position + shift, "invalid", fontsize=6, va="center", color="#a34f35")
        valid = sum(rep["valid_grades"] for rep in repetitions)
        expected = check["expected_grades"]
        ax.set(
            title=f"{NAMES[arm]} · {valid}/{expected} new grades valid",
            xlabel="Rubric score (0–1)",
            xlim=(-0.02, 1.13),
            ylim=(len(source) - 0.4, -0.6),
            yticks=range(len(source)),
            yticklabels=[f"{task['index']:02d} · {task['task_id'][:8]}" for task in source],
        )
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.grid(axis="both", alpha=0.13)
        ax.tick_params(axis="y", labelsize=8)
        ax.axvline(0.5, color="#9b9d9f", linewidth=0.8, linestyle=":")
    axes[0].scatter([], [], marker="D", s=32, facecolors="none", edgecolors="#202b37", label="Primary grade")
    axes[0].scatter([], [], color=COLORS["original"], s=23, label="New grades 1–3 (top to bottom)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Same answers, repeated model grading · {timestamp}", fontsize=13)
    fig.supxlabel(
        "Post hoc: unchanged primary answers and saved judge prompts. Unrun grades are omitted; invalid grades remain missing.\n"
        "Primary scores stay fixed. Same-model variation is not expert validation; repeated grades are not independent tasks.",
        fontsize=9,
    )
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        tmp = output / f"grading-variability.tmp.{suffix}"
        fig.savefig(tmp, dpi=160, metadata={"Creator": "EoM campaign report"} if suffix in ("pdf", "svg") else None)
        tmp.replace(output / f"grading-variability.{suffix}")
    plt.close(fig)


def export_repeat_stability(data, output, timestamp):
    """Show all task/run cells; uncompleted results remain visibly missing."""
    task_rows = {
        r["index"]: r
        for values in data["arms"].values()
        for r in values["episodes"]
        if r["phase"] == "evaluation" and r["epoch"] == 1
    }
    if not task_rows:
        return
    indices = sorted(task_rows)
    positions = {index: position for position, index in enumerate(indices)}
    labels = [
        f"{index:02d} · {task_rows[index]['subject'][:4]} · {task_rows[index]['task_id'][:8]}" for index in indices
    ]
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("#e8edf1")
    fig, axes = plt.subplots(1, 2, figsize=(12, 9), layout="constrained")
    for ax, arm in zip(axes, ("original", "teams"), strict=True):
        scores = np.full((len(indices), 3), np.nan)
        missing_final = np.zeros(scores.shape, dtype=bool)
        for row in data["arms"][arm]["episodes"]:
            if row["epoch"] != 1 or row["phase"] not in ("evaluation", "replication"):
                continue
            column = row.get("repeat", 0) if row["phase"] == "replication" else 0
            if column not in (0, 1, 2) or row["index"] not in positions:
                continue
            assert row["task_id"] == task_rows[row["index"]]["task_id"]
            position = positions[row["index"]]
            scores[position, column] = row["score"] or 0
            missing_final[position, column] = not row["has_final_answer"]
        mesh = ax.imshow(np.ma.masked_invalid(scores), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for position in range(len(indices)):
            for column in range(3):
                score = scores[position, column]
                label = "—" if np.isnan(score) else f"{score:.3f}" + ("†" if missing_final[position, column] else "")
                ax.text(
                    column,
                    position,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#33454e" if np.isnan(score) or score > 0.55 else "white",
                )
        counts = np.isfinite(scores).sum(axis=0)
        ax.set(
            title=NAMES[arm],
            yticks=range(len(indices)),
            yticklabels=labels,
            xticks=range(3),
            xticklabels=[
                f"{name}\n{count}/19 complete"
                for name, count in zip(("Primary", "Repeat 1", "Repeat 2"), counts, strict=True)
            ],
        )
        ax.tick_params(axis="both", length=0, labelsize=9)
    fig.colorbar(mesh, ax=axes, shrink=0.65, label="Rubric score (0–1)", pad=0.025)
    fig.suptitle(f"Evaluation variability from fixed trained populations · {timestamp}", fontsize=13)
    fig.supxlabel(
        "† No final answer: scored zero. Gray: not completed. Same 19 tasks; repeats are not independent task samples.\n"
        "Primary remains fixed. Model, scheduling and judge variability may all affect outcomes.",
        fontsize=9,
    )
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        tmp = output / f"repeat-stability.tmp.{suffix}"
        fig.savefig(tmp, dpi=160, metadata={"Creator": "EoM campaign report"} if suffix in ("pdf", "svg") else None)
        tmp.replace(output / f"repeat-stability.{suffix}")
    plt.close(fig)


def export(root):
    data = json.loads((root / "dashboard-data.json").read_text())
    plt.rcParams.update(
        {"font.family": "sans-serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for arm, values in data["arms"].items():
        episodes = [r for r in values["episodes"] if r["phase"] == "training"]
        if not episodes:
            continue
        xs = np.arange(1, len(episodes) + 1)
        scores = [r["score"] or 0 for r in episodes]
        missing = [not r["has_final_answer"] for r in episodes]
        axes[0, 0].plot(xs, scores, color=COLORS[arm], alpha=0.75, label=NAMES[arm])
        axes[0, 0].scatter(xs[missing], np.array(scores)[missing], marker="x", color=COLORS[arm], s=30)
        axes[0, 1].plot(xs, [r["cumulative_cost"] for r in episodes], color=COLORS[arm], label=NAMES[arm])
        axes[1, 0].plot(
            xs, np.cumsum([r["has_final_answer"] for r in episodes]) / xs, color=COLORS[arm], label=NAMES[arm]
        )
    axes[0, 0].set(title="Training rubric scores · × means no final answer", ylabel="Score", ylim=(-0.04, 1.04))
    axes[0, 1].set(title="Cumulative billed API cost, including preflight", ylabel="US dollars")
    axes[1, 0].set(title="Training final-answer submission rate", ylabel="Cumulative fraction", ylim=(-0.04, 1.04))
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("Completed training task")
        ax.grid(alpha=0.15)
        if ax.lines:
            ax.legend(frameon=False)
    ax = axes[1, 1]
    primary = next((p for p in data["paired"] if p["epoch"] == 1), None)
    if primary:
        tasks = primary["tasks"]
        ax.scatter([r["original"] for r in tasks], [r["teams"] for r in tasks], c=COLORS["teams"], s=45, alpha=0.7)
        ax.plot([0, 1], [0, 1], "--", c="#999999", linewidth=1)
        ax.set(
            title=f"Primary held-out task pairs: {len(tasks)}/19",
            xlabel="Original EoM score",
            ylabel="Team score",
            xlim=(-0.04, 1.04),
            ylim=(-0.04, 1.04),
        )
        ax.text(
            0.04,
            0.96,
            f"Paired mean difference: {primary['paired_score_difference']:+.3f}",
            transform=ax.transAxes,
            va="top",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Held-out evaluation pending\nTraining outcomes do not establish test performance.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    timestamp = datetime.fromtimestamp(data["updated_at"], timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(
        f"EoM / Teams · GPT-6 Luna · {timestamp}\nSystem comparison; unequal population and economic mechanisms",
        fontsize=14,
    )
    output = root / "figures"
    output.mkdir(exist_ok=True)
    export_repeat_stability(data, output, timestamp)
    export_grading_variability(data, output, timestamp)
    for suffix in ("png", "svg", "pdf"):
        tmp = output / f"comparison.tmp.{suffix}"
        fig.savefig(tmp, dpi=160, metadata={"Creator": "EoM campaign report"} if suffix in ("pdf", "svg") else None)
        tmp.replace(output / f"comparison.{suffix}")
    plt.close(fig)

    original = [r for r in data["arms"]["original"]["episodes"] if r["phase"] == "training"]
    team_training = [r for r in data["arms"]["teams"]["episodes"] if r["phase"] == "training"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    if original:
        xs = range(len(original) + 1)
        populations = [original[0]["population_before"]] + [r["population_after"] for r in original]
        roles = sorted({a.get("type") or "Unknown" for agents in populations for a in agents})
        role_counts = [
            [sum((a.get("type") or "Unknown") == role for a in agents) for agents in populations] for role in roles
        ]
        axes[0, 0].stackplot(xs, role_counts, labels=[r.replace("ResearchAgent", "") for r in roles], alpha=0.8)
        axes[0, 0].legend(ncol=3, frameon=False, fontsize=8, loc="upper left")
        axes[0, 0].set(
            title="Original EoM population by inherited role", ylabel="Active agents", xlabel="Completed training task"
        )
        births = [r["diagnostics"]["births"] for r in original]
        removals = [-r["diagnostics"]["removed_agents"] for r in original]
        axes[1, 0].bar(range(1, len(original) + 1), births, label="Added", color=COLORS["original"])
        axes[1, 0].bar(range(1, len(original) + 1), removals, label="Removed", color="#b7744a")
        axes[1, 0].axhline(0, color="#999999", linewidth=0.7)
        axes[1, 0].set(
            title="Original EoM population turnover", ylabel="Agents per task", xlabel="Completed training task"
        )
        axes[1, 0].legend(frameon=False)
    if team_training:
        xs = range(1, len(team_training) + 1)
        sizes = sorted({size for r in team_training for size in r["metrics"]["team_sizes"]})
        group_counts = [[r["metrics"]["team_sizes"].count(size) for r in team_training] for size in sizes]
        axes[0, 1].stackplot(
            xs, group_counts, labels=["Singleton" if s == 1 else f"Size {s}" for s in sizes], alpha=0.8
        )
        axes[0, 1].set(
            title="Voluntary groups by membership size", ylabel="Bidding groups", xlabel="Completed training task"
        )
        axes[0, 1].legend(frameon=False, loc="upper left")
        for key, label, color in [
            ("bid_paid_total", "Total paid", COLORS["teams"]),
            ("bid_transfer_total", "Transferred to previous winners", "#3e62bd"),
            ("bid_burn_total", "First-bid burns", "#b7744a"),
        ]:
            axes[1, 1].plot(xs, [r["metrics"][key] for r in team_training], label=label, color=color)
        axes[1, 1].set(
            title="Team payment flows · cumulative economy units",
            ylabel="Economy units (not API dollars)",
            xlabel="Completed training task",
        )
        axes[1, 1].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle(
        f"Population evolution and team economics · {timestamp}\nDifferent mechanisms and units; these plots describe behavior, not causal effects",
        fontsize=14,
    )
    for suffix in ("png", "svg", "pdf"):
        tmp = output / f"evolution-overview.tmp.{suffix}"
        fig.savefig(tmp, dpi=160, metadata={"Creator": "EoM campaign report"} if suffix in ("pdf", "svg") else None)
        tmp.replace(output / f"evolution-overview.{suffix}")
    plt.close(fig)

    episodes = [r for r in data["arms"]["teams"]["episodes"] if r["phase"] == "training"]
    if not episodes:
        return
    names = list(dict.fromkeys(a["name"] for r in episodes for a in r["population_after"]))
    tags = sorted({a["team"] for r in episodes for a in r["population_after"] if a.get("team")})
    tag_map = {tag: i + 1 for i, tag in enumerate(tags)}
    values = np.array(
        [
            [
                next((tag_map.get(a.get("team"), 0) for a in r["population_after"] if a["name"] == name), 0)
                for r in episodes
            ]
            for name in names
        ]
    )
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), layout="constrained")
    palette = ["#e8edf1"] + [plt.cm.tab20(i % 20) for i in range(len(tags))]
    cmap = matplotlib.colors.ListedColormap(palette)
    axes[0].imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=len(tags) + 0.5)
    axes[0].set(yticks=range(len(names)), yticklabels=names, title="Team membership after each training task")
    for x in range(len(episodes)):
        for y in range(len(names)):
            value = values[y, x]
            axes[0].text(
                x,
                y,
                tags[value - 1].replace("team-", "T") if value else "solo",
                ha="center",
                va="center",
                fontsize=max(5, 9 - len(episodes) // 15),
            )
    tick_indices = np.unique(np.linspace(0, len(episodes) - 1, min(12, len(episodes)), dtype=int))
    axes[0].set_xticks(tick_indices, tick_indices + 1)
    for index, name in enumerate(names):
        wealth = [next((a["wealth"] for a in r["population_after"] if a["name"] == name), np.nan) for r in episodes]
        axes[1].plot(range(1, len(episodes) + 1), wealth, label=name, color=plt.cm.tab20(index % 20))
    axes[1].set(title="Personal wealth · team economy units", ylabel="Wealth", xlabel="Completed training task")
    axes[1].legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1), fontsize=8, frameon=False)
    axes[1].set_title("Personal wealth · team economy units", pad=40)
    axes[1].grid(alpha=0.15)
    fig.suptitle(f"Voluntary formation and individual outcomes · {timestamp}", fontsize=14)
    for suffix in ("png", "svg", "pdf"):
        tmp = output / f"team-dynamics.tmp.{suffix}"
        fig.savefig(tmp, dpi=160, metadata={"Creator": "EoM campaign report"} if suffix in ("pdf", "svg") else None)
        tmp.replace(output / f"team-dynamics.{suffix}")
    plt.close(fig)

    counts = [r.get("negotiation", {}).get("counts", {}) for r in episodes]
    eligible = np.array([c.get("revisable_personal_pledges", 0) for c in counts])
    revised = np.array([c.get("revised_personal_pledges", 0) for c in counts])
    winners = np.array([c.get("winning_teams", 0) for c in counts])
    unequal = np.array([c.get("unequal_winning_teams", 0) for c in counts])
    zeros = np.array([c.get("winning_teams_with_zero_money_contributor", 0) for c in counts])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    xs = np.arange(1, len(episodes) + 1)
    fractions = np.divide(revised, eligible, out=np.full(len(xs), np.nan), where=eligible > 0)
    axes[0].plot(xs, fractions, marker="o", markersize=4, color=COLORS["teams"])
    axes[0].set(
        title="Personal pledge revisions · each training task",
        ylabel="Fraction changed after first pledge",
        xlabel="Completed training task",
        ylim=(-0.03, 1.03),
    )
    totals = np.cumsum(winners)
    categories = np.array([winners - unequal, unequal - zeros, zeros]).cumsum(axis=1)
    shares = np.divide(categories, totals, out=np.zeros(categories.shape, dtype=float), where=totals > 0)
    axes[1].stackplot(
        xs,
        shares,
        labels=["Equal pledges", "Unequal, all positive", "Includes a zero-money contributor"],
        colors=["#b6d8d2", "#438f91", "#d1a56d"],
    )
    axes[1].set(
        title="Winning-team allocations · cumulative shares",
        ylabel="Fraction of winning teams",
        xlabel="Completed training task",
        ylim=(0, 1),
    )
    axes[1].legend(loc="lower left", frameon=True, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle(f"Negotiated personal contributions · {timestamp}", fontsize=14)
    fig.supxlabel(
        "Initial pledges can already respond to teammates. Zero money contribution does not imply zero intellectual contribution.",
        fontsize=9,
    )
    for suffix in ("png", "svg", "pdf"):
        tmp = output / f"negotiation-dynamics.tmp.{suffix}"
        fig.savefig(tmp, dpi=160, metadata={"Creator": "EoM campaign report"} if suffix in ("pdf", "svg") else None)
        tmp.replace(output / f"negotiation-dynamics.{suffix}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    export(parser.parse_args().root)
