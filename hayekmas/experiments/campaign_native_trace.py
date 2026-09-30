"""Expose native EoM's recorded lifecycle without changing its engine.

The text logger rounds monetary values. These observations are display-only;
they are not an exact accounting ledger or a replacement for checkpoints.
"""

from collections import Counter
import re


def native_trace(path):
    if not path.exists():
        return {"events": [], "credit_checks": [], "counts": {}}
    events = []
    checks = []
    trial, step = 1, None
    credit = None
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        text = raw.strip()
        replay = re.match(r"🔁 REPLAY trial (\d+)/(\d+)", text)
        next_step = re.match(r"┌─ Step (\d+)/(\d+)", text)
        if replay:
            trial, step, credit = int(replay[1]), None, None
        if next_step:
            step, credit = int(next_step[1]), None
        if text.startswith("🔍 CREDIT CHECK:"):
            credit = {"trial": trial, "step": step, "line": line_number, "agents": []}
            checks.append(credit)
        balance = re.fullmatch(r"(.+?) (\d+): \$(-?[\d.]+) (✓ solvent|💀 BANKRUPT)", text)
        if credit is not None and balance:
            credit["agents"].append(
                {
                    "name": balance[1],
                    "id": int(balance[2]),
                    "rounded_wealth": float(balance[3]),
                    "bankrupt": balance[4].endswith("BANKRUPT"),
                }
            )
        kind = None
        for pattern, candidate in (
            (r"🔁 REPLAY trial", "replay"),
            (r"┌─ Step", "step"),
            (r"🏆 WINNER:", "winner"),
            (r"💰 Payment:", "payment"),
            (r"Balances?:", "balance"),
            (r"💀 REMOVING", "removed"),
            (r"(?:🧬|💀) SPAWNING:", "birth"),
            (r"🔄 Restoring surviving agents", "restore"),
            (r"🛤️\s+Path reward", "path_reward"),
            (r"🔍 CREDIT CHECK:", "credit_check"),
            (r"✅ No bankruptcies - episode ACCEPTED", "accepted"),
            (r"⚠️\s+No active agents", "no_active_agents"),
        ):
            if re.match(pattern, text):
                kind = candidate
                break
        if kind:
            events.append({"kind": kind, "trial": trial, "step": step, "line": line_number, "text": text})
    return {
        "events": events,
        "credit_checks": checks,
        "counts": dict(Counter(e["kind"] for e in events)),
        "precision": "Native text-log money values are rounded; use committed checkpoints for exact end-of-task wealth.",
        "scope": "All native replay trials in the committed API attempt. Provider-interrupted attempts remain in their separate artifact directories.",
    }
