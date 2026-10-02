# Live interaction dashboard

From the repository root, serve an existing priority12 study without changing its experiment code:

```sh
.venv/bin/python scripts/serve_priority12_dashboard.py \
  --root runs/k10-priority12-20261002 --port 8791
```

Open http://127.0.0.1:8791/. The server binds only to localhost. It reads the plan, summary, completed results, and in-progress JSONL event logs. It does not use the API key or make model calls. Its root page overrides the supervisor's basic generated index without modifying the experiment or its source fingerprints.

Select a configuration and task, then scrub or play recorded rounds. Team colors and membership history come from committed membership events. Gold rings identify the winning members. Select an agent to filter conversations or a message to highlight its recorded recipients. Independent agents' own scratchpad messages have no recipient edges. Opening wealth is available in each node's tooltip. Filters separate formation, solving, bidding, and drafts/submissions. Expand messages to read their full text. Follow live returns to the newest available round.

Original EoM has no team communication log; the completed native action history and answer are shown instead. Missing grades are explicitly labeled and never converted to zero. These are development-task runs with fresh populations, not held-out evaluations of trained populations.
