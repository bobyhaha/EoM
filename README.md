<h1 align="center">Economy of Minds: Emerging Multi-Agent Intelligence with Economic Interactions</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2606.02859"><img src="https://img.shields.io/badge/arXiv-2606.02859-b31b1b.svg" alt="arXiv"></a>
  <a href="https://arxiv.org/pdf/2606.02859"><img src="https://img.shields.io/badge/pdf-2606.02859-yellow.svg" alt="pdf"></a>
  <img src="https://img.shields.io/badge/python->=3.10-blue.svg" alt="python">
  <a href="https://zhentingqi.github.io/internal/projects/EoM/"><img src="https://img.shields.io/badge/project-webpage-brightgreen.svg" alt="project webpage"></a>
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="license">
</p>

## Overview

### Voluntary-team prototype

The new round-team protocol adds discussion before activation/bidding, voluntary
team changes between decision rounds, and an episode path reward of **R/N to
each member for every contributing round** (no division by team size).
Run `.venv/bin/python main.py global_configs/teams_rounds_demo.json` for an
offline demonstration and open `runs/teams-rounds-demo/replay.html`.
See the first section of **[TEAMS.md](TEAMS.md)** for the exact rules, differences
from the earlier experiment, and how to inspect each round's credit allocation.
Existing configs default to `interaction_protocol: legacy` for reproducibility.
To prepare the requested k=10/20/50/100 study and its cost scenarios without API
calls, run `.venv/bin/python -m hayekmas.experiments.population_study --out runs/population-study-plan`.
Open the generated `index.html`. These are unlaunched study specifications;
they are not runnable inputs for the older campaign supervisor.
The older team layer uses negotiated personal contributions,
joint costly auctions, a shared scratchpad, equal final-team rewards, and paid reflection.
Every run includes an interactive replay of team formation and conversations.
See **[TEAMS.md](TEAMS.md)** for the mechanism, setup, live viewing and experiments.
Original adapters and the original individual engine are retained. The team
extension also compares against a direct single agent and independent pass@k
on the bundled FrontierScience-Research tasks; see the scientific comparison
commands in **[TEAMS.md](TEAMS.md)**.

### Current original-EoM versus teams campaign

The 12-hour GPT-6 Luna campaign saves everything under
`runs/eom-vs-teams-12h/`. Its saved deadline is September 30, 2026 at
15:10:53 UTC (08:10:53 Pacific), with a cumulative $25 ceiling for each arm.
Preflight work, interrupted attempts, and replayed work all remain charged
against those ceilings. The primary experiment trains on the bundled 40 tasks,
then evaluates a fresh copy of the trained checkpoint on each of 19 held-out
tasks. A second training epoch has a separately specified start-time gate.
Population training changes agent state and editable prompts; it does not
fine-tune GPT-6 Luna's model weights. The report counts actual strategy edits
and population changes separately from merely offering agents that option.

The live viewer is **http://127.0.0.1:8769/** while its local server is running.
Choose a system and an episode to inspect agent wealth, membership, actions,
and conversations. The team selector also offers the current live task.
Click either system's score in a comparison table to jump directly to that
task's trajectory in the correct primary evaluation or repeat.
Completed team episodes have a detailed round replay. Results are not evidence
that teams help until the held-out comparison is complete.
For original EoM, expand **Original EoM auctions and lifecycle** in an episode
to see payment chains, credit checks, bankruptcies, births, and replay resets.
Those native log amounts are rounded; exact final wealth is in the checkpoint.
For teams, expand **Contribution negotiations and payments** to compare each
member's first and final pledge and see which group won. The negotiation figure
tracks revisions and unequal monetary contributions over training. Per-auction
checks verify binding pledges, transfers to previous winners, equal rewards,
and every individual's settled wealth from the raw event log.
The generated `snapshot.html` is a portable interactive copy that opens without
a server. It displays its saved timestamp and never makes API or network calls.
Keep the result directory beside it for links to raw artifacts and figures.
`RESEARCH_FINDINGS.txt` provides a readable report with primary results,
separate repeats, cumulative billed and reserved costs, training dynamics,
protocol limitations, and audit status. It refreshes alongside figure exports;
`research-findings.json` contains the same structured findings. Incomplete
repeats remain explicitly incomplete, even after the research window ends.
`FIRST_TASK_WALKTHROUGH.txt` explains the first held-out task's two trajectories;
its companion JSON retains exact messages, decisions, and source hashes.
`TASK_OUTCOMES.txt` lists all held-out tasks with primary and repeated scores,
submission failures, answer paths, and exact saved primary grader explanations.
Those explanations are model judgments, not independent expert validation.
`SUBMISSION_REVIEW.txt` separates failures before any submission from failures
after intermediate work, and partitions each paired score gap by which systems
submitted. These are descriptive subsets, not causal estimates. Regenerate it
with `.venv/bin/python -m hayekmas.experiments.campaign_submission_review --root runs/eom-vs-teams-12h`.
`ARITHMETIC_SPOT_CHECK.txt` independently recomputes the first task's numerical
rubric targets and compares them with its original primary answer. It checks
one answer's arithmetic/rubric alignment, not the benchmark's scientific validity.
`SYMMETRY_ALGEBRA_REVIEW.txt`, `PHOTOCHEMISTRY_UNITS_REVIEW.txt`, and
`RUBRIC_NOTATION_REVIEW.txt` document selected post hoc checks against primary
sources and independent calculations. They flag specific mathematical ambiguities
without replacing grades or estimating benchmark-wide error. Reproduce the first
two calculations with `campaign_symmetry_check` and `campaign_photochemistry_check`
(using the same `--root` argument as the other report commands).
`SOURCE_CONSISTENCY_REVIEW.txt` records source hashes and historical reconstruction
limits; `FOLLOWUP_STUDY_DESIGN.txt` proposes a future controlled study, including
single-agent and pass-at-k baselines. That proposal is not an additional run.
`TEAM_PARTICIPATION.txt` and `figures/team-participation.png` show every team
agent's recorded proposals, selected actions, and payments during training.
Candidate authorship is not exclusive intellectual credit or a measure of quality.
Regenerate them after all 40 primary training tasks with
`.venv/bin/python -m hayekmas.experiments.campaign_participation --root runs/eom-vs-teams-12h`.

To serve saved results again, run from this repository:

```bash
.venv/bin/python -m http.server 8769 --bind 127.0.0.1 --directory runs/eom-vs-teams-12h
```

These commands regenerate reports from saved files without API calls:

```bash
.venv/bin/python -m hayekmas.experiments.campaign_report --root runs/eom-vs-teams-12h --figures
.venv/bin/python -m hayekmas.experiments.campaign_audit --root runs/eom-vs-teams-12h
```

Use `--watch --figures` on the report command for continuous updates; run only
one report writer per result directory. **Paid execution for this campaign is
closed.** All planned studies finished, and its `STOP` file deliberately blocks
new requests. Keep it in place and use a new campaign directory for future
experiments. `closed-ledger-manifest.json` seals the six request, response, and
cost logs; the audit detects changes. The unresolved original-arm cost receipt
still retains its conservative reservation.

Artifacts include `plan.json` and `RESEARCH_NOTES.txt` for the protocol and
repairs; `episodes.csv` for task outcomes and costs; `audit.json` for provenance
and completeness checks; `figures/` for PNG, SVG, and PDF exports; and each arm's
`api_usage.jsonl`, raw request/response logs, checkpoints, and per-task
`result.json`, `answer.md`, and `trajectory.log`. Committed task files are the
report's authoritative source. Team task folders also contain event logs and
`replay.html`.
Team-specific payment/message fields are blank for original EoM when equivalent
telemetry was not recorded; they must not be interpreted as zero transfers.
API costs with missing receipts retain a conservative reservation. CSV and
episode views report billed amounts and unconfirmed reservations separately;
both count against the spending limit, including after a worker restart.

The following describes the recorded execution sequence; both repeats and the
two separate diagnostics are now complete. Do not reactivate this closed run.
Optional repeat evaluations were initially disabled. After both
primary workers finish, the campaign can select `worker_mode: "replication"`
in `plan.json` and restart the stopped workers through the supervisor. This
runs at most two additional evaluations of the fixed first-epoch checkpoint,
with fresh copies for all 19 tasks and separately declared scheduling seeds.
Each arm must have at least two hours remaining to start a new repeat. Results
live under `arm/replications/repeat-N/`; the dashboard and CSV keep them separate
from primary results. They use the same cumulative ledgers and deadlines.
Repeated observations on these tasks are not additional independent test tasks.

A separate optional interface diagnostic is prepared for after the primary and
repeat evaluations. It compares unchanged and clarified team instructions on
the first three training tasks, using fresh populations. Its $1 total allowance
is included in the existing team ceiling and cannot replenish on restart.
Activation uses `worker_mode: "diagnostic"` and only the stopped team worker;
the runner requires complete primary studies and at least 45 minutes to start.
Results and replays stay under `teams/interface-diagnostic/`. These exploratory
training results never enter the primary or repeated held-out comparisons.

A separate post hoc grading reliability check reuses each submitted primary
answer's exact saved judge prompt three times, after the solver repeats and
interface diagnostic. Its additional allowance is $0.50 per arm within the
existing ceilings. Primary scores and solver outputs remain fixed. Unsubmitted
answers retain zero without a call; invalid grading syntax remains missing and
prevents a full-run regraded mean. The design is saved in
`grader-recheck-design.json`; outputs are under `arm/grader-recheck/` and appear
in a separate viewer/report section. This measures variability of the same
model judge, not independent scientific correctness. Activation uses
`worker_mode: "grader_recheck"` only after workers have exited and the start
gates pass; the saved deadline and cumulative ledgers still apply.

This is a comparison of complete systems. The original engine retains its
native roles, rent, reproduction, bankruptcy, fixed bids, and reward settings.
Teams use twelve generalists, voluntary membership, negotiated personal
contributions, equal reward shares, and optional paid reflection. Population
sizes, wealth scales, evolution, and test-time economics differ. Both arms use
the same task split, model, and rubric grader; equal dollar ceilings do not
mean equal token consumption. See the protocol for differences from the paper.
Native original prompt mutation also receives training rubrics; the team's
reflection interface uses bounded history summaries. This supervision
difference is preserved and disclosed. Held-out references belong only in
grader requests, and the audit checks recorded requests for full-rubric copies.
Both arms use the text-only ResearchWorld environment: their actions add
reasoning to shared notes, and an LLM grades the final answer. No browser,
literature-retrieval, or code-execution tool is supplied to either arm. A message
such as "I'll verify this" records an intention, not evidence that an external
source was consulted or a calculation was executed.

How can a population of agents self-orchestrate and self-adapt into stronger
collective intelligence **without centralized control**? Standard approaches
introduce a central orchestrator to create agents, assign specializations, and
coordinate actions — but this bottlenecks planning at a single coordination
gate and makes learning increasingly inefficient as the system scales.

Inspired by Friedrich Hayek's theory that **prices are a decentralized
coordination signal**, we present **Economy of Minds (EoM)**, a system in which
a population of agents compete via *auctions* for the right to act, exchange
payments through *peer-to-peer transactions*, and accumulate or lose wealth
based on environmental rewards. These simple economic signals induce
decentralized credit assignment and drive planning without global orchestration
or explicit communication protocols. The population *evolves through economic
selection*: agents that consistently contribute to successful trajectories
accumulate wealth and are mutated via **exploitation**, while ineffective agents
go bankrupt and are replaced via **exploration**. Initialized with weak agents,
the economy produces emergent multi-step reasoning strategies and outperforms
stronger monolithic baselines across five agentic tasks — mathematical
reasoning, financial research, scientific research, accelerator design, and
distributed-system optimization.



https://github.com/user-attachments/assets/bf8a5819-75aa-4452-836e-8f6b71967677



## Repository

The repository is organized around a small core engine and domain adapters:

- `main.py`: the global launcher. It reads one JSON config and dispatches to
  the configured adapter runtime.
- `global_configs/`: top-level run configs for the available domains.
- `hayekmas/base/`: the generic auction, training, evaluation, population,
  reward, and configuration machinery.
- `hayekmas/adapters/`: domain-specific environments, agents, prompts, and
  runtime config loaders.
- `hayekmas/utils/`: logging, LLM client construction, data, and visualization
  helpers.
- `third_party/benchmarks/`: benchmark assets used by the adapters.

## Supported Adapters

- `arch_dse_world`: accelerator design-space exploration for ResNet-50 layer
  mapping on a Gemmini-style systolic array. Config:
  `global_configs/train_arch_dse_world.json`.
- `cloudcast`: code evolution for a multi-cloud broadcast routing program.
  Config: `global_configs/train_cloudcast.json`.
- `researchworld`: scientific research reasoning with a rubric-based LLM
  judge. Config: `global_configs/train_research.json`.

## Execution Flow

1. `main.py` loads a JSON config from `global_configs/`.
2. The config's `domain` key selects an adapter runtime.
3. The adapter runtime deep-merges its adapter config, if present, with the
   global config.
4. The runtime builds the configured LLM client, environment, agents, and EoM
   engine.
5. Training or evaluation runs according to the config's `mode`.

## Installation

For general installation:

```bash
pip install -e ".[litellm]"
cd third_party/smolagents
pip install -e ".[toolkit]"
```

Please go to [CloudCast](#cloudcast),
[Frontier-Science-Research](#frontier-science-research), and
[Arch DSE World](#arch-dse-world) for domain-specific installations.

## Running

Run a config through the global launcher:

```bash
python main.py global_configs/train_cloudcast.json
python main.py global_configs/train_research.json
python main.py global_configs/train_arch_dse_world.json
```

Model/API selection lives in each config's `model` section. Common choices:

- `litellm`: cloud APIs and OpenAI-compatible gateways.
- `localhost`: a local OpenAI-compatible server such as vLLM or SGLang.
- `together`: Together AI chat completions.
- `demo`: deterministic toy client for lightweight local checks.
- `vllm` / `sglang`: direct local inference backends.

For `litellm`, set provider-specific environment variables such as
`OPENAI_API_KEY`, or use an OpenAI-compatible gateway with:

```bash
export OPENAI_BASE_URL=<your-base-url>
export OPENAI_API_KEY=<your-key>
```

For `localhost`, set `model.api_base` in the config or:

```bash
export LOCALHOST_BASE_URL=http://127.0.0.1:8000/v1
export LOCALHOST_API_KEY=not-needed
```

If `model.name` is empty, the localhost client attempts to detect the model
from `/v1/models`.

## CloudCast

The `cloudcast` adapter is a code-evolution task. A society of agents edits
a single Python file, `initial_program.py`, that defines a multi-cloud
broadcast routing algorithm. The verifier runs the program on five inter-
and intra-cloud scenarios using a Skyplane-derived cost and throughput
grid and returns the total egress cost; the score is
`max(0, 1 - cost / 1035)`, where `1035` is the cost of the Dijkstra
single-path seed. The workspace persists across episodes.

The task is from [ADRS](https://arxiv.org/pdf/2510.06189).

### Roles

Six fixed roles, defined in `agent.py`:

- `PlannerCloudcastAgent` (`planner`) — proposes the next sub-goal.
- `ReaderCloudcastAgent` (`reader`) — reads files in the workspace.
- `ImplementerCloudcastAgent` (`implementer`) — edits `initial_program.py`.
- `BuilderCloudcastAgent` (`builder`) — runs build / import checks.
- `EvaluatorCloudcastAgent` (`evaluator`) — calls the verifier mid-episode.
- `FinalizerCloudcastAgent` (`finalizer`) — submits the program with `final_answer`.

The auction selects one acting role per step. The last
`mas.terminal.start_on_step_from_end` steps of an episode are restricted
to agents carrying the tags in `mas.terminal.candidate_agent_tags`
(`["terminal"]` by default — only `Finalizer` qualifies).

### Files

- `hayekmas/adapters/cloudcast/` — adapter code (`agent.py`, `env.py`,
  `runtime.py`, `prompts.py`, `tools.py`, `task.py`).
- `third_party/benchmarks/cloudcast-broadcast-opt/` — task directory:
  `instruction.md`, `environment/initial_program.py` with the EVOLVE
  block, `environment/profiles/` cost and throughput grids, and the
  verifier under `tests/`. Runs offline.

Configuration in `hayekmas/adapters/cloudcast/configs/train.json`:

- `run.preserve_workspace_across_episodes` — keep the edited program
  across episodes.
- `run.num_episodes`, `run.max_steps` — episode and step budgets.
- `mas.engine.{min_num_agents, max_num_agents}` — population bounds.
- `mas.terminal.{enabled, start_on_step_from_end, candidate_agent_tags}`
  — restrict the final steps of an episode to the tagged agents.
- `mas.reward.{regression_multiplier, broken_program_penalty,
  path_reward_per_unique_author}` — reward shaping specific to this
  adapter.

## Frontier-Science-Research

The `researchworld` adapter uses tasks from OpenAI's [FrontierScience-Research benchmark](https://openai.com/index/frontierscience/), which targets scientific research reasoning in physics, chemistry and biology.
There are no external tools and the answer is graded by a rubric-based LLM judge.

### Roles

The `researchworld` also uses five specialized agents. All five are defined in `agent.py`; each has a `FROZEN_SYSTEM_PROMPT`(role identity, never mutated) and a `TRAINABLE_SYSTEM_PROMPT`
(strategy, evolved by the Hayek birth loop):

- `LiteratureResearchAgent` (`literature`) — surfaces definitions / theorems / standard formulas; no new derivation.
- `PlannerResearchAgent` (`planner`) — outlines the sub-parts (a), (b), (c)… and the tactic for each.
- `DeriverResearchAgent` (`deriver`) — the workhorse: one concrete derivation/calculation per turn.
- `VerifierResearchAgent` (`verifier`) — sanity-checks the latest contribution (signs, units, limits).
- `AnswerResearchAgent` (`answer`) — emit `<final_answer>…</final_answer>`; emitting it terminates the episode.

Wakeup rules: an agent never acts twice in the same role back-to-back; `literature`/`planner` may self-start an empty episode; `answer` only wakes after a `deriver`/`verifier` turn.


### Rubric reward

The LLM-judger reads the problem, rubric, and candidate answer and replies with `SCORE:` (clamped to`[0, 1]`) and `REASON:`. A task passes when `score >= judge.threshold` (default `0.7`). 

The same model that drives the agents also acts as the judge (`env.llm_fn`). 

Researchworld responsibilities are split by concern:

- `agent.py`: the five research agents, `ResearchAction`, and birth/serialization logic
- `env.py`: `ResearchEnv`, JSONL task loading, and the rubric-based LLM judge
- `runtime.py`: two-layer config parsing, train/eval entrypoints, and periodic-test logic

## Arch DSE World

The `arch_dse_world` adapter runs accelerator design-space exploration: a
ResNet-50 mapping search on a Gemmini-style systolic array, evaluated by
Timeloop + Accelergy (the DOSA paper's pipeline). Relevant files:

- `hayekmas/adapters/arch_dse_world/` holds the adapter code (agent, env,
  runtime) and the bundled simulator helper.
- `hayekmas/adapters/arch_dse_world/simulator/` holds the workspace template
  and ResNet-50 workload YAMLs.
- `hayekmas/adapters/arch_dse_world/configs/` holds adapter-level configs.
- `scripts/arch_dse_world/setup_arch_dse_simulator.sh` installs the Timeloop +
  Accelergy + DOSA simulator backend.
- `scripts/arch_dse_world/dosa_bounded_edps.json` stores cached DOSA baseline
  values.
- `scripts/arch_dse_world/launch_24jobs_perlayer.sh` is the optional per-layer
  launcher for SLURM-style cluster runs.

### Installation

This domain needs two external pieces because the reward comes from a real
hardware simulator.

First, configure an LLM. The default global config uses Together AI:

```bash
export TOGETHER_API_KEY=...
```

For practical throughput, you can instead serve the model yourself with vLLM or
SGLang and set `model.api` to `localhost`, with `LOCALHOST_BASE_URL` or
`model.api_base` pointing at your server. `litellm` with `OPENAI_BASE_URL` and
`OPENAI_API_KEY` also works. In every case, `model.name` must match the exact
model id your backend serves.

Second, install the simulator backend:

```bash
bash scripts/arch_dse_world/setup_arch_dse_simulator.sh
```

The script creates a self-contained conda environment, clones DOSA, builds
Timeloop + Accelergy, and prints the environment variables to export. A run
then looks like:

```bash
export TOGETHER_API_KEY=...
export DSE_CONDA_ENV=/path/to/arch_dse_sim   # printed by the setup script
export DOSA_ROOT=/path/to/dosa               # printed by the setup script
export ARCHGYM_SCRATCH="$(mktemp -d)"
python main.py global_configs/train_arch_dse_world.json
```

Gurobi is not required to run `arch_dse_world`: the eval path only runs
Timeloop on a given hardware/mapping pair. Gurobi is only needed if you
regenerate DOSA's own mapping-search baseline; cached baseline values are in
`scripts/arch_dse_world/dosa_bounded_edps.json`.

For cluster-scale per-layer experiments, use
`scripts/arch_dse_world/launch_24jobs_perlayer.sh` as the starting point and
override the environment variables it documents for your scheduler setup.

## Citation

```bibtex
@misc{qi2026economymindsemergingmultiagent,
      title={Economy of Minds: Emerging Multi-Agent Intelligence with Economic Interactions}, 
      author={Zhenting Qi and Huangyuan Su and Ao Qu and Chenyu Wang and Yu Yao and Han Zheng and Kushal Chattopadhyay and Guowei Xu and Zihan Wang and Weirui Ye and Vijay Janapa Reddi and Ju Li and Paul Pu Liang and Himabindu Lakkaraju and Sham Kakade and Yilun Du},
      year={2026},
      eprint={2606.02859},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.02859}, 
}
```

## License

MIT
