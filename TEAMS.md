# Voluntary teams on Economy of Minds

This is the full [upstream EoM repository](https://github.com/zhentingqi/EoM),
based on commit `e7faa7233c58ee3a8c88ef2e7971d083c98d30a8`, with a team adapter.
It implements the narrow prototype from the
[shared design](https://chatgpt.com/share/6abc4cbd-a124-83ea-9755-019abb92cd39),
the supplied implementation notes, and the subsequent requirements for
**communicated contribution decisions** and a **shared team scratchpad**.

## k=10 funding ablation

The October 1 funding study uses fresh populations on three development tasks,
not the full population-training protocol below. It crosses `objective_mode`
(`wealth` or `society`) with `team_bid_rule` (`voluntary` or `fixed`). Agents
have no assigned roles. Each task starts with ten agents; evolution and
reflection are disabled, the horizon is ten rounds, and lambda is zero.
The exact reproducible defaults are in
[`global_configs/k10_bid_ablation.json`](global_configs/k10_bid_ablation.json).

A voluntary team bid is the sum of individually authorized pledges. With the
fixed rule, `act=true` and `authorize_base_bid=true` authorize a maximum charge
of `team_base_bid * bid_cost_rate`. A strict majority must consent and be able
to afford that maximum. The actual total bid, 0.1 by default, is shared equally
among consenting yes voters if the team wins. Other members pay zero. Tied teams
are selected by the seeded lottery. This is a fixed-price consent mechanism,
not an automatic subsidy or a minimum added to voluntary pledges.

To launch a new four-condition study from the repository directory:

```bash
.venv/bin/python -m hayekmas.experiments.bid_ablation \
  --launch --root runs/my-k10-study --hours 3 --seed 7 --cell-cap 2
```

The example allows at most $2 per condition, $8 for this launch. These are
ceilings, not spending targets; separate launches need separate aggregate
budget accounting. The program uses `OPENROUTER_API_KEY` when set, otherwise
asks for it without echoing it. It refuses to overwrite an existing study or
automatically restart a paid cell. Use a new output directory for each run.
The launcher snapshots the resolved protocol and source hashes in `plan.json`;
the bundled training dataset supplies the same three task IDs for all cells.

Inspect `index.html`, per-task `replay.html`, `events.jsonl`, `result.json`, and
the per-condition request and cost ledgers. To serve a study locally:

```bash
.venv/bin/python -m http.server 8790 --bind 127.0.0.1 --directory runs/my-k10-study
```

Open `http://127.0.0.1:8790/`. Use another port if this session's server is
already running. The replay exposes membership, bids, payment transfers,
public work, private team messages, and historical reward credits. A
`STOP` file in the study directory prevents subsequent provider calls;
an already-started call may finish. Interrupted tasks are not scored as zero.

The session also includes separately labeled feedback, membership-clarification,
and three-round follow-ups, independent samples and reference-free selection,
an original-engine reference, and repeated grading. These are not pooled into
the primary factorial comparison. The fixed-bid mechanism does not by itself
establish a solution-quality advantage over single agents or original EoM.

## Population evolution (prepared full study)

The prepared study now uses **k initial agents and a 2k living-population cap**
for both original EoM and teams. Enable the team extension with:

```json
{
  "interaction_protocol": "rounds",
  "condition": "dynamic",
  "evolution_enabled": true,
  "num_agents": 10,
  "population_cap_multiplier": 2,
  "birth_interval": 5,
  "num_births_per_interval": 2,
  "p_a": 0.0,
  "p_b": 1.0,
  "periodical_good_p": 0.5,
  "rent": 0.0,
  "rent_interval": 5
}
```

After each training episode, settle historical R/N-per-member credit and optional
reflection, charge configured rent when due, then remove agents with **wealth < 0**.
Zero wealth alone is not bankruptcy. With affordable pledges, nonnegative task
rewards and rent=0, team bankruptcy may never occur. We do not introduce an
unrequested failure penalty or silently treat zero as bankruptcy.

For each bankruptcy, use the upstream p_a/p_b alternatives: mutate the richest
survivor, repair the removed agent using available training failure context, or
make no birth. Every five completed training tasks, attempt two additional births,
independently choosing success mutation or failure repair with probability 0.5.
All births share the 2k cap; a full population causes skips without a model call.
Prepared study membership-context limits scale to max(6144, 128*k) reference
tokens so identities for up to 2k agents fit; money ballots remain at 3072.
No minimum-population replenishment is configured. An empty population stops.

Mutation reuses the upstream good/bad birth specifications through the bounded
team policy client (JSON strategy response, update_tokens cap, medium reasoning
in solve-first). It updates only editable strategies; it never assigns specialist
roles or changes the model weights. Malformed/short strategies fall back to the
source prompt; provider failures are logged without minting a child. New agents
have unique names, recorded ancestry and fresh initial wealth. They start solo,
receive no historical credit, and must obtain consent to join teams. Existing
members keep their historical payments even if their team later dissolves.

Birth endowments, rent and signed removed balances have separate accounting
entries; initial endowments are not rewards. Replays show births, removals,
population counts and strategy inheritance. Checkpoints support changed population
sizes and continue agent naming/RNG state. Test copies and `training=False` disable
rent, births, removals and reflection. Runtime `split: "test"` selects this mode;
test auctions and rewards still operate inside the disposable episode.

This shares population evolution mechanisms but is **not an exact reproduction**
of original HayekMAS: teams do not replay completed tasks after bankruptcy or
preserve specialist roles. Original EoM retains these native behaviors. Report
actual population counts, turnover and spending alongside task scores.

The offline estimator models successful +2 periodic growth every five training
tasks up to 2k, with separate assumed replacement-mutation calls; it also stores
an at-cap cost sensitivity. Final test populations may differ between arms.
Old fixed-population estimates and old checkpoints are not valid substitutes.
No paid runs are launched by preparation. To prepare and inspect locally:

```bash
.venv/bin/python -m hayekmas.adapters.teams.runtime --config global_configs/teams_rounds_demo.json --out runs/my-evolution-demo --no-plots
.venv/bin/python -m hayekmas.experiments.population_study --out runs/my-evolution-study
```

The scripted demo starts with six agents; after task five there are eight, and
after task ten there are ten (cap twelve). It validates mechanics, not emergence.
Evolution defaults to false for compatibility with archived protocols/configs;
it is explicitly enabled in the current round demo and prepared team study cells.

## New protocol: teams within each decision round

Use `teams.interaction_protocol: "rounds"`, with `round_schedule: "compact"`
(the default round schedule). This is a separately selectable
mechanism; saved legacy results are not rerun or relabeled. On October 1, 2026,
the upstream HEAD was `d272266f7d6a0c302b7af5d20f1326b63c5656fb`. The local
original `base/mas.py`, `base/config.py`, `base/population.py`, and
`researchworld/agent.py` match those upstream files byte for byte.

In original EoM, ordinary steps are **activation/wake-up, bidding among active
agents, selection, payment, action**, repeated until termination. Terminal mode
can bypass wake-up and restrict the final auction to answer agents. Its default
path reward pays each occurrence in the realized action path; an optional mode
deduplicates authors. See [the upstream loop and path rewards](https://github.com/zhentingqi/EoM/blob/d272266f7d6a0c302b7af5d20f1326b63c5656fb/hayekmas/base/mas.py).

An **episode** is one task. A **decision round** is one opportunity to auction
and publish work within that episode. For backward-compatible storage,
`config.rounds` still means the number of episodes, `config.max_steps` bounds
decision rounds per episode, event `round` identifies the episode, and event
`step` identifies its decision round (both zero based).

1. Every decision round starts with one short public membership proposal per
   agent: stay or leave, optionally invite one person, and explain briefly.
   Departures are applied before invitations. Only solo recipients with valid
   invitations get an additional acceptance/decline call, at most one each.
   This allows a late invitation to be accepted without three full passes.
   Team membership changes only by consent; a departure and new invitation
   can be proposed together.
2. Every member gets one free-form message on the team's private scratchpad.
   Members can discuss readiness, the problem, and proposed personal amounts.
3. Each member makes one binding `act` vote **for the team** and authorizes
   their own pledge. Everyone sees the completed discussion; current ballots
   stay hidden until all members decide. A **strict majority of all members**
   must vote yes. Ties and invalid ballots do not activate the team; all its
   pledges are then canceled. Otherwise a positive sum enters the auction.
   `act: false` makes that member's pledge zero. Voting yes with zero money is
   allowed. No member can spend another's money. All members of an accepted
   contribution, including no voters and zero-money members, earn path rewards.
4. The winning team uses its shared discussion to propose and vote on a public
   contribution. Intermediate rounds add work to the common solution; the
   final round proposes a complete answer. No permanent roles, leader,
   mandatory critique, or independent-draft duties are assigned.
5. After the winning action, the next round starts with a new membership window.
   Existing teams may stay intact; they are not forcibly dissolved. Invitations
   expire at each membership window. Membership
   remains frozen during discussion, the auction and submission. Dynamic
   membership ignores the legacy every-five-tasks schedule.
6. One final answer is graded. The environment returns reward R, and credits
   are paid using the recorded team membership of **each accepted round**.

If N rounds published accepted work (including the final answer), then:

\[
\Delta w_i^{\mathrm{reward}} = \sum_{r=1}^{N}\mathbf{1}[i\in T_r]\frac{R}{N}.
\]

There is **no division by team size**. For R=12 and three contributing rounds,
each membership occurrence receives 4. If round 1 uses A+B and rounds 2 and 3
use A+C, A receives 12, B receives 4, and C receives 8. Joining C's team later
does not give C any of B's past credit. A zero-money member receives the same
share. Failed rounds and private work from losing teams are excluded from N.
Accepted means submitted to the environment, not independently proven useful.

Total issued reward is `(R/N) * sum(team_sizes_on_the_path)`, which can exceed
R. `reward` records environment R, `reward_issued` records actual new wealth,
and `credit_path` lists every round, saved members and payouts. `reward_income`
aggregates each person's receipts. The conservation check uses actual issued
wealth. Bid payments remain separate: winners pay their own pledges, later
bids transfer equally to the previous winning round's recorded members, and
the first winning bid burns. Switching cannot redirect a past team's payment.

An empty intermediate round does not end the episode. Calls are reserved for
discussion, the binding ballots and finalization; tight budgets may skip a
membership window or finish early, with explicit events. If no team activates
with a positive bid at the deadline, **no team is forced to act**: there is no
final answer and reward is zero. A winning team's real final proposal can still
survive malformed candidate votes through a logged candidate lottery. This
does not override a team's act/abstain decision or fabricate an answer.

`round_schedule: "full"` retains the previous round schedule for reproducibility:
one opening message, three configured membership passes, two configured private
discussion turns and two configured bid turns per member. It uses individual
activation and a free team lottery if the deadline is unfunded. Compact changes
those rules explicitly; full-versus-compact is therefore a bundled mechanism
comparison, not a clean single-variable ablation. Legacy EoM is unchanged.
`formation_turns` and `bidding_turns` control full/legacy schedules; compact has
one proposal plus targeted replies and one ballot regardless of those settings.
`discussion_turns` still controls the winner's substantive intermediate work
(two turns by default); compact pre-bid chat always has one turn.

Round teaming requires `bidding_mode: negotiated`, `collaboration_mode:
discussion`, and `finalization_enabled: true`. The `dynamic` condition offers
membership changes; the other conditions remain useful fixed/singleton
ablations. The comparison runner enables within-task formation for the new
protocol even after training, starts every test from a fresh trained copy, and
continues to disable paid strategy reflection during tests.

Run the new protocol without an API key:

```bash
.venv/bin/python -m hayekmas.adapters.teams.runtime \
  --config global_configs/teams_rounds_demo.json \
  --out runs/my-round-team-demo --no-plots
```

Open `runs/my-round-team-demo/replay.html`. Play advances one recorded event;
the membership timeline now contains each decision round. Read the activation,
regrouping and **Episode path reward** events to see exact decisions and credit.
The demo uses scripted arithmetic, not a scientific performance evaluation.
For a real run, copy your model/dataset/budget settings into a new config and
select this protocol; do not resume the sealed campaign. No paid evaluation of
this new mechanism has been performed as part of this change.

### Population size

The new demo uses six scripted agents to make individual events easy to inspect.
The adapter's general default remains 12 agents, with at most four in a team.
Six agents is useful for mechanism checks, but does not establish how teaming
scales. The requested full study uses k = 10, 20, 50 and 100, with all 40 training
tasks and 19 test tasks. Preparation and cost estimation do not authorize paid
execution; the saved study plan explicitly remains unlaunched.

Prepare the full study and its interactive cost estimate, without API calls:

```bash
.venv/bin/python -m hayekmas.experiments.population_study \
  --out runs/my-population-study-plan
```

Open `runs/my-population-study-plan/index.html`. It includes eight cell
specifications, dataset/source hashes, token assumptions, one- and three-seed
scenarios, prior spending, and single-agent/pass@k references. These cell files
are **study specifications**, not inputs for the old campaign launcher. A new
budget-enforcing launcher is required before paid execution. The estimator
does not load an API key or make requests.

With lambda=0, compact uses **3k + I** calls before each winning action: k
membership proposals, I invitation replies (0 through k), k private messages
and k combined act/pledge ballots. At k=10 that is 30–40 rather than 80. The
central estimate assumes I=0.5k, not a measured switching rate. One seed of 40
training plus 19 test tasks at ten rounds projects about $30.39 with the new
solve-first token allocation (previous compact $28.72; full $51.26), before a
20% allowance. Original EoM remains about $4. No savings are assumed from abstention or caching. These estimates
use the stored Luna prices and unmeasured mean token assumptions, not spending
caps. The generated plan includes exact phase arithmetic and full comparisons.

### Spend computational effort on solving

Every compact team agent receives this system instruction:

> Spend most of your reasoning and response-token budget on useful contributions to solving the problem.
> Keep membership, act/abstain, and pledge decisions brief.

The prompt clarifies that this concerns computational effort, not the amount of
personal wealth pledged, and discourages filling the budget with unnecessary
text. It does not assign any agent a role or change the wealth/reward objective.

The current demo and study explicitly set `token_profile: "solve_first"` and:

| Phase | Output cap per call | OpenRouter reasoning effort |
|---|---:|---|
| Membership / invitation reply | 192 | none |
| Act vote and own pledge | 96 | none |
| Private problem-solving discussion | 2,048 | medium |
| Winning public work and complete final proposals | 16,384 | high |
| Candidate-selection vote | 64 | none |

The prepared study's solving cap doubles from 8,192 to 16,384; its overall
input context limit increases to 262,144 so long candidates still fit in team
discussion/voting. The 8,192-token grader budget is unchanged. Original EoM
settings are unchanged; this comparison is not matched in compute or effort.

Act/pledge prompts contain the latest current-round message from each teammate,
team balances, identity and decision rules. They do not resend the complete
problem or old public solution. The complete serialized input, including system
and learned strategy instructions, is capped at 3,072 reference tokens.
Membership prompts use bounded public-work and profile excerpts with a 6,144
input-token limit. All roster identities, memberships, balances and invitation
IDs are preserved. Truncation is labeled; the original audit log is retained.
If required fields alone exceed a cap, stop before a paid call. No extra LLM
summary call is used. These smaller views provide less detail for partner
selection; substantive work still sees the full problem and bounded shared state.

The `standard` profile preserves the previous allocation for ablations. It is
the general configuration default; use the updated demo or prepared cell specs
to select solve-first. Both profiles use the brief-budget system guidance.

OpenRouter requests now explicitly set the phase effort and record phase,
output cap, confirmed cost, and reasoning-token usage when provided. The model
must support the requested efforts; the client does not silently fall back.
Other providers retain their provider-specific reasoning behavior. Provider
`max_tokens` can cover reasoning and visible output together; a 16,384-token cap
is not a promise of 16,384 visible answer tokens. See the
[OpenRouter reasoning documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

`usage.json` includes provider `phase_costs`. Its solving share includes private
problem-solving discussion, winning public work and final proposals; selection
votes and grading remain separate. Unknown reasoning counts stay unknown, and
unresolved cost reservations are separate from confirmed charges. The central
k=10 plan allocates about **78%** to these solving opportunities, including
about **53%** to winning public work and final proposals alone. These are
planning assumptions, not proof of useful work or guaranteed actual spending.

We expect substantive reasoning and answer checking to influence task score
more directly than repeated membership/bidding administration, but this has
not been established by paid experiments. Equal R/N is a payment rule, not a
causal estimate of each round's value. Measure final environment score,
completion and dollars separately from amplified agent wealth. Proposed
ablations hold activation/reward/finalization fixed while changing formation
frequency, private chat length or repeated bidding individually. To estimate a
round's marginal value, remove its public contribution and rerun the remaining
episode on paired tasks/seeds. Those additional runs are not part of the
prepared budget and are not authorized to execute.

Compare original and team arms at matched population sizes, model, training
exposure, and per-task API budgets; preserve a separately labeled upstream
evolution baseline if fixing its population changes its original mechanism.
Report task scores, answer-submission rates and dollar cost separately from
wealth: amplified path payouts increase wealth mechanically with team size.
Larger populations alone do not guarantee more diverse reasoning or better
answers. The earlier paid team arm already had 12 agents; the original arm
evolved from five specialists to 19 agents, so that result was not a six-agent
comparison.

### Hypotheses to examine

This is a collaboration-incentive prototype, not yet evidence of better task
performance. Equal per-member path rewards make joining a productive team
valuable even without useful work or money. Expect pressure toward the team-size
cap, unequal personal pledges, and possible free riding. Next-bid payments can
reinforce incumbents: when the same members win consecutive auctions, the
payment is redistributed within that group. Switching may instead follow
changing subproblems if agents learn who contributes useful work.

Repeated path participation is rewarded, so agents may also publish unnecessary
intermediate actions to gain a larger fraction of the action path. Accepted
work is not guaranteed useful. Large populations can add redundancy and
coordination costs rather than expertise. These are mechanism-derived
hypotheses, not observations from the scripted demonstration. Track team sizes,
membership duration, unique contributions, pledge inequality, and task score
separately. Compact majority activation may reduce unilateral expensive actions
but can also create deadlock (a two-person team needs both votes). Track all
abstentions, canceled pledges and deadline failures. Full-schedule finalization
lotteries remain visible in historical replays; compact removes that recovery.

### Coordination fee (first study: lambda = 0)

`teams.coordination_fee_lambda` defaults to **0.0**. Both the demo and all
prepared population-study cells set it to zero explicitly. At zero there are
no fee charges or extra consent calls; the R/N reward rule is unchanged.

For a later ablation, a positive lambda charges each member
`lambda * (team_size - 1)` once per decision round before private discussion.
Fees are burned, including for teams that lose the auction. The value is in
simulated wealth units, not API dollars. Singleton agents pay zero.

Every member first receives the maximum fee and chooses whether to pay or
leave. Missing/invalid consent and unaffordable fees cause departure without
a charge. After all departures, the fee is recomputed from each remaining
team's size; it cannot exceed what anyone consented to. No fee is charged
when the call budget cannot fit consent plus a complete discussion turn.
Positive fees currently require dynamic round teaming so voluntary exit is
available. No new member can enter between consent and payment.

Replay events show consent and burned fees. `coordination_paid` records each
agent's episode total; `coordination_burn_total` is cumulative and survives
checkpoints. Accounting uses initial wealth + issued rewards − burned bids −
reflection costs − coordination fees. A joining fee alone would discourage
mobility rather than large stable teams; the recurring coordination fee targets
team size. Positive-fee experiments and their extra consent-call costs are
deferred, not silently included in this first study.

### What the earlier low-scoring team run actually measured

The sealed `runs/eom-vs-teams-12h` experiment used the legacy mechanism:

| Aspect | Earlier campaign | New round-team protocol |
|---|---|---|
| Membership | Changed every five training tasks; fixed within tasks and all tests | Voluntary switching windows between decision rounds |
| Before auction | Negotiate monetary pledges; only winners then discuss solutions | Every team discusses first, then explicit activation and pledge negotiation |
| Empty candidate window | End the episode without a final answer | Continue; reserve finalization and log any recovery |
| Terminal credit | Only final acting team; R divided by its size | Every accepted round; each saved member receives full R/N |
| Earlier work | Receives next-bid transfers only | Retains next-bid transfers and terminal path credit |

Primary held-out results: original submitted 17/19 answers and scored 0.256;
teams submitted 7/19 and scored 0.122. Of 12 team non-submissions, nine happened
before any public submission and three after intermediate work. In the six
tasks where both systems submitted, means were 0.305 versus 0.296. This selected
subset is descriptive, not a controlled estimate of answer quality. Missing
answers account arithmetically for most of the aggregate gap. They do not
prove that repairing finalization, changing rewards, or adding communication
will improve future scores.

The original baseline retained its upstream engine. The team arm was a separate
prototype, not a faithful implementation of the round-team mechanism described
above. Training also differed: original EoM evolved its population and prompts;
the team population stayed fixed, and all 480 optional reflections were declined.
Different populations, supervision, economics and stopping policies prevent a
causal conclusion that teamwork is worse. Exact evidence remains in
`runs/eom-vs-teams-12h/SUBMISSION_REVIEW.txt` and `RESEARCH_FINDINGS.txt`.

The remaining sections document the older configurations and additional modes.

## Run and watch

From this repository:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[teams]'
.venv/bin/python main.py global_configs/teams_demo.json
```

Open `runs/teams-demo/replay.html` in a browser. It is self-contained and works
offline. Select a task, then use Play, Previous/Next event, or the event slider
to follow each message, pledge, auction, vote and settlement. Phase buttons jump
to bidding, discussion, finalization, submission or rewards. The diagram and
inspector show state through the selected event, including intermediate balances.
The complete transcript remains readable while playback advances; scrolling
pauses playback. Text selections and expanded event details survive playback.
Select an agent, click a membership-timeline cell, or filter messages by team,
action step and activity. Follow playback is optional and off by default.
The activity stream includes invitations, negotiated pledges, discussion,
candidates, votes, payments, reflection inspection and prompt updates.

To watch a live run update **after every completed round**, serve its output
directory locally in another terminal:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory runs/teams-demo
```

Open `http://127.0.0.1:8765/replay.html`. It checks for newly completed rounds
every two seconds. New tasks appear in the selector without changing your
playback or reading position. “Go to end” explicitly selects the latest event.
Completed runs need no server. Files are replaced atomically between rounds.

To upgrade an archived replay without modifying the recorded run, render copies
into a separate directory:

```bash
.venv/bin/python scripts/render_team_replays.py runs/old-run/teams runs/old-run-viewers
```

Open the destination's `index.html`. Its manifest records source hashes, and the
original JSON and HTML files remain unchanged.

Output directories must be new. To repeat a run:

```bash
.venv/bin/python -m hayekmas.adapters.teams.runtime \
  --out runs/another-demo --rounds 12 --seed 17
```

The **scripted demo is a plumbing demonstration, not evidence of emergence**.
Its arithmetic, invitations, voluntary exits and new pairings are programmed.
The default live backend lets the model choose those behaviors with identical
initial prompts and no functional role assignments. The opt-in reviewed mode
below assigns temporary draft, check and revision duties.

## What changes in EoM

| Existing component | Extension |
|---|---|
| `main.py` | Registers `domain: teams` using the normal adapter launcher |
| `BaseAgent` | `TeamAgent` adds team tag, private trajectory, summary and mutable strategy |
| `Population` | The original population stores all team agents |
| `BaseEnv` | Exact-answer diagnostic environment, plus a bridge to the original `ResearchEnv` |
| Original LLM clients | Reused by `ModelPolicy`; metered `OpenRouterClient` implements the same EoM interface |
| Individual execution engine | Separate `TeamMAS` adapter for repeated team auctions and actions within each task |

The original adapters, individual auction, and evolution engine remain usable
through their original configurations. The team adapter intentionally does
not call the original role-based wakeup or mutation loop. The original
`ResearchEnv` grader is used unchanged through an action bridge; no specialized
`ResearchAgent` is instantiated. The native action's `answer` tag is only a
grader protocol field and is never shown as an agent role.

## Legacy episode-level team protocol

This section describes the default `negotiated` bidding and `discussion`
collaboration protocol. Existing configurations retain this behavior.

1. **Formation:** before revealing the task, agents can invite, accept, send a
   message, leave, or pass. Invites need the recipient's acceptance, expire at
   phase end, and cannot move an entire existing team. Membership persists.
2. **Task reveal and funding discussion:** teammates use a shared channel to
   discuss the task and contribution allocation. Each turn can revise the
   speaker's own pledge. Current pledges and member wealth are visible within
   the team. Requests to others are nonbinding; only an agent can commit its
   own money. Final valid pledges after the bounded conversation form the bid.
3. **Auction:** largest sum wins; exact ties use a seeded lottery. Only the
   winning members pay `bid_cost_rate × contribution`. The first payment of each
   task goes to the void. Later payments go equally to the previous winning
   membership. An all-zero auction ends the task without a new action or reward.
   Membership is locked for the entire task.
4. **Shared scratchpad:** task, current environment state, member public
   summaries, the funding discussion, subsequent team discussion, and the
   wealth objective are available. Every member has the same message and
   candidate actions. Prompts do not request role assignment or critique.
5. **Submission:** every member can propose an intermediate public step
   (`final: false`) or final answer (`final: true`). One vote per member
   chooses the candidate; ties use a seeded lottery. An empty candidate pool
   opens the final-answer recovery window described below. No eligible final
   candidate after recovery, or no valid votes, means no submission.
   This uses the voting alternative in the design
   notes and adds no external solver or leader. Intermediate steps update the
   environment visible to all teams, then bidding restarts. The team scratchpad
   persists throughout the task. The default limit is three action steps; at
   the last step only final candidates are eligible. Agents can finish earlier.
6. **Reward:** the environment's reward is divided equally among the frozen
   winning membership, including zero contributors. Only the current auction winner pays; previous
   winners receive bid income even when they lose the next auction.
7. **Summaries and reflection:** bounded running extracts are updated. Agents
   may pay a fixed fee for an inspection call followed by a strategy-update call.
   Only their mutable strategy changes. Reflection happens once after the task.
   Completed state and replay are saved.

The current configuration uses 12 agents, personal wealth 20, a maximum reward
of 12, two funding turns, two task-discussion turns and a reflection fee of 2.
Formation is available every five episodes with three bounded formation turns
so invitations can be accepted without depending on a single scheduling pass.
Speaking order is randomized using the recorded seed. There is an engineering
cap of four members; it is not an emergent team-size result.

### Final-answer recovery

New runs default to `teams.finalization_enabled: true`. If discussion produces
no eligible candidate, the winning team receives a dedicated final-answer
window before the episode can end. Every member can return a nonempty string
`candidate` with `final: true`, or explicitly abstain with `{"abstain": true}`.
Each malformed final response gets at most one correction attempt. Valid
abstentions are not retried. The existing equal vote selects among eligible
final answers. Discussion text and intermediate steps are never automatically
promoted into final answers; all abstentions or invalid votes can still leave
the task unanswered. Existing successful submissions need no recovery calls.

The engine reserves `3 * winning_member_count + 1` calls: a final proposal,
one possible correction and one vote per member, plus one possible rubric
judge call. Before bidding it checks the full auction cost and the largest
possible winning group's reserve. Discussion can shorten to protect these
calls, and its countdown reflects the available turns. If another complete
auction cannot fit, only final candidates are eligible at the current step.
Optional formation and reflection can be skipped when their worst-case costs
would exceed the available call budget. The shared comparison call limit is
also respected. Provider dollar limits, deadlines and transport failures remain
independent hard limits; this reservation is not a guarantee against them.

Recovery belongs to the same paid auction: it never charges the bid again or
changes transfers, reward sharing, membership or reflection prices. Replays
show when finalization starts, agents abstain, or budget limits shorten a phase.

Set `"finalization_enabled": false` inside `teams` to reproduce the legacy
execution protocol, including immediate termination on an empty candidate pool.
Old config files still load; specify this flag explicitly when reproducing old
runs. The completed `eom-vs-teams-12h` campaign used the legacy protocol and its
saved results have not been rerun or changed. Start new experiments in a new
output directory. The regression tests include the pre-change legacy trace,
recovery through the HTTP client and native research grader, payment accounting,
checkpoint continuation, abstention, malformed outputs and tight call budgets.

No team treasury, salaries, contracts, kicking, permanent functional role labels,
diversity reward, rent, or birth/death evolution was enabled in that legacy experiment.
The current round-team study enables the population evolution described above.

### Experimental sealed bids and reviewed answers

Set `bidding_mode: "sealed"` and `collaboration_mode: "reviewed"` inside `teams`
to enable the new variant. The two switches are independent so either change
can be evaluated alone. Both default to the existing protocol. Reviewed mode
requires `finalization_enabled: true`.

1. Each agent makes **one binding personal pledge** plus a brief assessment,
   limited by `bid_tokens` (default 256). No current peer pledges or assessments
   are exposed in this request. Membership, bid payment, auction tie-breaking,
   and equal `R / |T|` reward sharing remain unchanged.
   If every initial pledge is zero, one additional funding round exposes actual
   teammate pledges and explains that no team can act without a positive bid.
   Contributions remain voluntary; if bids stay zero, no winner is fabricated.
2. The winning members produce independent complete answers and checklists
   derived only from the public question. No member sees another draft until
   all independent proposals finish. A seeded order assigns a temporary drafter
   and independent checkers; these duties confer no extra reward or authority
   over bids. For pairs, both members contribute one independent proposal.
3. The other members review the primary draft against their independent work,
   checking each requirement and giving evidence for errors or uncertainty.
   A singleton self-reviews. Malformed reviews are explicitly marked unavailable.
   Shared evidence is bounded by `evidence_tokens`; truncation is flagged.
4. The primary author revises once, accounting for the review and unresolved
   disagreement in a complete final answer. There is no final-answer vote or
   lottery. This mode finishes within the same paid auction, rather than
   publishing intermediate actions and reauctioning the task.

Each draft and revision allows at most one format correction. Explicit
abstentions are not retried. An invalid revision can fall back to an earlier
explicitly complete final proposal; it never promotes discussion or an
intermediate step. Revision abstention withdraws that author's proposal, so a
fallback must come from another member. No valid proposal means no submission.
The fallback and all draft, review and revision events are visible in the replay.
Provider failure, deadlines and dollar limits remain hard stops, not fake answers.

The engine reserves `2*n + max(1,n-1) + 3` calls before bidding: independent
proposals and possible repairs, reviews, revision and possible repair, and
grading. For 12 agents in pairs, a clean one-auction task needs 12 bid calls,
2 independent proposals, 1 review, 1 revision and 1 grader: **17 calls**.
The prior recovery path commonly used 33. Actual token usage and score still
need measurement; fewer calls do not guarantee better answers or equal compute.
The auction also reserves one extra bid call per agent for the conditional
all-zero funding round; a paired 12-agent task using it needs 29 calls without
format repairs. Unused funding calls are not spent. The metered campaign client
normalizes the last complete JSON decision for each new phase, without changing
its values; later explicit abstentions take precedence over earlier proposals.

Run the offline plumbing demo (no credentials or network required):

```bash
.venv/bin/python main.py global_configs/teams_reviewed_demo.json
```

Prepare a new checkpoint-based 19-task evaluation **without starting model calls**:

```bash
.venv/bin/python -m hayekmas.experiments.team_finalization_evaluation \
  --root runs/teams-reviewed-19-new \
  --prepare-from runs/eom-vs-teams-12h --protocol reviewed
```

`--protocol sealed_only` and `--protocol review_only` support separate ablations;
`--protocol finalization` preserves the previous submission-recovery variant.
Preparation freezes code, checkpoint, task order and protocol in a fresh output
directory. Run that prepared directory using the existing evaluator, which
prompts for a key if none is supplied in the environment. These are development
tests on already inspected tasks. An equal dollar ceiling is not a matched
compute experiment; compare actual tokens/cost and repeat on fresh tasks before
claiming a team advantage. Old output directories and the original EoM engine
are not modified by selecting this mode.

## Bid transfers and personal wealth

Let `c_i,t` be one agent's negotiated pledge and `B_t = sum(c_i,t)` the current
winning team's total bid. With `bid_cost_rate = 1`, every current winning member
pays its own `c_i,t`. Every **previous** winning member receives
`B_t / size(previous_team)`. Environmental reward goes equally to the team
submitting the final answer, independent of its members' contribution sizes.

For example, team A's members pay 6 and 2 on step 1 (8 to the void). On step 2,
team B's members pay 3 and 3: A's members each receive 3. If B solves the task
for reward 12, B's members each receive 6. Starting at wealth 20 each, the final
wealths are **A: 17, 21; B: 23, 23**. Total wealth is 80 − 8 + 12 = 84.

A repeated win by the same team still debits individual pledges and credits
an equal share: its total wealth is unchanged by that transfer, but unequal
contributors redistribute wealth within the team. The previous membership is
captured at the prior action, not looked up after a future membership change.
The payment chain resets for each new task. A one-step solution has no later
bid to transfer. The final bid does not carry into the next task.

With cost rate `r`, both debits and transferred totals are multiplied by `r`.
The conservation check is `total wealth = initial wealth + environment rewards
− first-bid burns − reflection fees`; transfers neither create nor destroy money.
Simulation wealth and reflection fees are **not API dollars**.

The replay labels events by action step, offers a Step filter, and shows every
bid's recipients and credited amounts. Agent inspection shows total payments,
bid income, and reward income for the selected round; "Last pledge" refers to
the last auction. `metrics.json` also contains per-step winners and wealth.

## Observation and token boundaries

- Own reflection evidence: recent detailed local events.
- Teammates: bounded individual summaries.
- Competitors: bounded published team-history extracts, rather than complete
  conversations or private strategies.
- The audit log/replay can show everything to the researcher. Agents never
  receive that global log, ground-truth answers or grading rubrics.
- Summaries are deterministic rolling extracts, not a semantic LLM summarizer.
  They can omit context or truncate a record; the full recorded messages remain
  available in the replay and event files.
- `evidence_tokens` bounds serialized reflection evidence. `inspect_tokens`
  and `update_tokens` independently cap the two generation calls. Output is also
  checked locally; an over-budget response is invalid, never replaced by a
  scripted live-model response.
- Local text counts use **o200k_base** as an explicit reference encoding.
  Provider `max_tokens` bounds native completion generation. Provider-specific
  framing and hidden reasoning tokens are not included in reference counts.
  The tokenizer may download its encoding file on first use.
- `max_calls` bounds requested decision and judge calls. Upstream clients can
  retry internally; this is not a dollar cap or an exact count of billable
  requests for those backends. The OpenRouter client makes one HTTP request per
  generation, logs actual `usage.cost`, and additionally enforces a USD reservation
  budget. See the OpenRouter instructions below.

## Real models and tasks

Edit `model.name` in `global_configs/teams_localhost.json` to match your served
model, then run it with `main.py`. The existing EoM `localhost` client uses
`LOCALHOST_API_KEY` when needed. Model configuration also supports EoM's
`litellm` and `together` backends; install their upstream dependencies first.
Keep credentials in the environment. No paid provider was used for development.

For an exact-answer dataset, add `dataset` and `split` to the config:

```json
{"id":"task-1","split":"train","problem":"What is 3 + 4?","answer":7}
```

Use one JSON object per line, unique IDs within a split, and at least as many
selected tasks as requested rounds. Strings are matched after trimming and
case folding; numbers use absolute tolerance `1e-8`. There is no symbolic-math
equivalence or code execution grader. No dataset means generated arithmetic
diagnostics, including when using a live model.

`global_configs/teams_researchworld.json` instead uses the **original EoM
ResearchEnv rubric grader** with the same live model as the agents. Point
`dataset` at your research JSONL with `id`, `problem`, and rubric in `answer`.
The checkout includes **40 training tasks and 19 test tasks** under
`third_party/benchmarks/frontier-science-research/data/`. The loader now accepts
its native `task_group_id`, `problem`, `answer` (rubric), and `subject` fields.
A per-row `split`, if present, is filtered; otherwise the configured file defines
the split. The repaired config points to the bundled training file and uses
OpenRouter with a $10 total run allowance.
Intermediate public actions are ungraded; the final answer is graded once.
Reference rubrics go only
to the judge; malformed responses and transport errors stop the run rather
than count as task failures. The reward scale is configured to produce
`reward × score` without negative centering. Original CloudCast and accelerator
multi-step tool environments are not wired into this V1 team adapter.

The normal team runtime adapts continually on its selected tasks. Merely
setting `split: test` there does not freeze adaptation. Use the comparison
harness below for isolated test attempts and an optional separate training phase.

## OpenRouter: start with a bounded pilot

The ready config is `global_configs/teams_openrouter.json`: 12 agents, 12 tasks,
up to 3 steps each, 512 action tokens (256 for solutions unless overridden), 2,500 decision/judge calls,
and a **$10 local run allowance**. The initial model is
`qwen/qwen3-30b-a3b-instruct-2507`, a non-thinking model suitable for testing this
JSON protocol. This is a starting choice, not a measured quality recommendation.
Its model ID was verified against the public OpenRouter model catalogue during
development. Recheck availability and pricing before larger runs.

Create a dedicated key in your OpenRouter workspace with a $10 credit limit
and no automatic reset. Keep it out of JSON configs, logs, and git. In zsh:

```bash
read -s 'OPENROUTER_API_KEY?OpenRouter key: '
export OPENROUTER_API_KEY
.venv/bin/python -m hayekmas.adapters.teams.runtime \
  --config global_configs/teams_openrouter.json --out runs/openrouter-pilot-1
unset OPENROUTER_API_KEY
```

Alternatively, put a key in a private file outside the repo with mode `600`,
and set `model.extra_kwargs.api_key_file` to that path. Do not paste the key
into the conversation. The client reads it locally and sends it only as the
Authorization header to `https://openrouter.ai/api/v1/chat/completions`.
Redirects and automatic retries/fallbacks are disabled.

What generates paid calls:

- Every agent's formation turn, when formation is enabled.
- Each member's bid-negotiation turn; a singleton makes one contribution call.
- Each winning member's discussion turns and vote.
- One reflection decision per agent per task; choosing reflection adds an
  inspection call and a strategy-update call.
- ResearchWorld only: one additional rubric-judge call for a submitted final
  answer. Exact-answer grading and summation task generation are local.

There are no web-search, browser, code-execution, or external task-tool calls
in this team adapter. An agent message is model output delivered by Python;
it is not another network service. The original repo's other domain adapters
are separate and are not invoked by this config.

Each OpenRouter request specifies provider price ceilings (`$0.15/M` input,
`$0.60/M` output in the pilot), rejects per-request fees, and reserves estimated
input cost plus the maximum output cost before sending. The estimate uses text
bytes plus a framing margin; it is not a universal tokenizer guarantee. The
**OpenRouter key credit limit is the authoritative account-side cap**. The
local guard stops when the next reservation does not fit, or when usage is
missing, a request fails, or actual cost exceeds its reservation. Unconfirmed
reservations remain in the audit log; inspect them before restarting. An
interrupted run does not automatically resume. Other EoM clients do not have
this dollar guard.

`usage.json` → `provider` reports actual returned cost, native input/output
counts, request count, and any unconfirmed amount. `api_usage.jsonl` is flushed
and synced before every dispatch, so an abrupt stop still leaves a request
reservation. The manifest records the requested model; the usage log also
records the responding model/provider and generation ID.

For your $2,500 credit, a staged **allocation ceiling**, not a spending target:
$25 for protocol pilots; $175 to choose tasks/models using measured quality and
cost; $1,800 for matched comparisons across four conditions and multiple seeds;
$500 reserved for replication and follow-up. Begin with the $10 key/run cap.
Raise limits only after checking invalid actions, task difficulty, and measured
cost per completed round. Arithmetic is a smoke test; use the bundled FrontierScience-Research config
or supply meaningful exact/rubric tasks before the main study. Do not interpret spending the
whole credit as evidence of a better experiment.

Project total cost from the pilot's reported USD per completed round, then
multiply by planned rounds, condition count, and seeds. Formation and optional
reflection make cost nonconstant, so keep headroom. Inspect error runs separately.
Larger action/context budgets may be necessary for substantive research answers.
All agents and the rubric judge currently share one configured model.

Official references: [usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting),
[provider price ceilings](https://openrouter.ai/docs/guides/routing/provider-selection),
and [key credit limits](https://openrouter.ai/docs/api_reference/limits).

## Scientific tasks, single agent, and pass@k

The scientific comparison uses the **actual FrontierScience-Research files in
this EoM checkout**, rather than generated arithmetic. Subjects cover physics,
chemistry and biology. Examples include deriving a scheme for weak-value
amplification of particle lifetimes and reasoning about molecular-junction
breaking energies. The paper reports a 40/20 split, but the checked-in files
contain **40/19**; the harness records the actual IDs and dataset hashes and
rejects requests for a nonexistent twentieth test task.

`global_configs/compare_science_openrouter.json` uses all 19 held-out test tasks,
12 independent samples per task, k = 1, 2, 4, 8, 12, and one local run seed.
`global_configs/compare_science_pilot.json` instead uses three **training-file**
problems for development/calibration, keeping the test file untouched.
It uses the same configured model and original EoM rubric grader for every
method. Scientific solutions can use 4,096 output tokens per discussion/answer;
judges have 2,048; up to eight team action steps are available. Funding/vote
calls retain the smaller action budget. Context and public-environment budgets
are larger than the arithmetic demo. These are configurable limits, not a claim
that this allowance is sufficient for every scientific problem.

Validate the full dataset/config **without credentials or paid calls**:

```bash
.venv/bin/python -m hayekmas.adapters.teams.evaluation \
  --config global_configs/compare_science_openrouter.json \
  --tasks 19 --train-tasks 40 --validate-only
```

After setting `OPENROUTER_API_KEY`, run the bounded scientific pilot:

```bash
.venv/bin/python -m hayekmas.adapters.teams.evaluation \
  --config global_configs/compare_science_pilot.json \
  --out runs/science-comparison-development-1
```

Open its `comparison.html`: a pass-rate/compute table, selectable task and k,
independent candidate answers with scores, and a link to each team's full
formation/conversation replay. `comparison.json` contains mean rubric scores,
per-task outcomes, per-replicate rates and their standard deviation, native
and reference token counts, and costs. `outcomes.jsonl` and `requests.jsonl`
preserve partial work on interruption. Private rubrics appear only in judge
requests in the researcher audit log; they are never included in agent or
single-agent observations. Summary aggregates are withheld until the complete
planned comparison finishes, so a budget-truncated subset is not presented as
the final result.

The three comparisons mean:

- **Team:** the full team mechanism submits one selected final answer per task.
- **Single agent:** one direct solve, with no auction, teammates, or feedback.
  It is the first predesignated sample in the independent pool, never a selected
  successful answer. It differs from the old `individual` condition, which
  still runs a population of auctioning singleton agents.
- **pass@k:** generate n fresh attempts per task with identical public input,
  no shared conversation/history, and no reference or judge feedback. If c
  pass, estimate `1 - choose(n-c, k) / choose(n, k)` and average across tasks
  within each replicate. All configured k must be <= n. Single-agent results
  and every k reuse this pool; no extra calls are made just to report a new k.
  `pass@1` uses c/n and can differ from first-attempt single-agent accuracy.

Pass@k is **oracle success coverage**: it does not provide a deployable answer
selector. Team accuracy and pass@k have different inference costs. The report
shows measured team/first-sample usage, the expected cost of k randomly chosen
samples from the pool, and the actual total cost of generating/grading the
entire pool. Do not add the pass@k rows together. Grader calls are included and
also counted separately. Unmetered backends report dollar/native-token usage
as unavailable; the scripted demo reports $0. The comparison's dollar cap and
`max_total_calls` are shared across every method, seed, training, and grading
call in this invocation, rather than multiplied per condition.

The science pass threshold is explicitly **0.5**, matching the pinned upstream
adapter config; the main README's prose gives another default. Change it only
as a predefined experiment setting. Mean rubric scores are reported alongside
pass rates. The same model currently generates and judges solutions, so inspect
rubric judgments before making claims about model or team superiority.

By default, each test task starts with a fresh untrained team population,
allowing formation before the task is revealed. There is no test reflection or
cross-test-task feedback. To study learned teams, set `--train-tasks 40`: the
team population first adapts on the separate training file, then each test
receives an independent copy of its trained membership, strategies, memories,
and wealth. Test membership/strategies are fixed; negotiation and payments
still happen within each test task, and all resulting changes are discarded
before the next one. Training expenses are reported separately. This preserves
the requested team economy; it is **not an exact reproduction** of the paper's
test protocol, which freezes economic state. The single-agent baseline is
zero-shot even when teams are trained, so training is additional compute and
data exposure, not a matched-training comparison.

For a larger predefined comparison, use the test config, adjust budget/model
ceilings and `max_total_calls` first, then
use `--tasks 19 --train-tasks 40 --samples 24 --ks 1,2,4,8,12,24 --seeds 7,17,29`.
The seed controls Python scheduling/ties, not guaranteed provider decoding.
Choose model, k, threshold, prompts, and budgets using training/development
problems before the final test. The three-task pilot uses the training file and is exploratory. Keep
the separate test config for the final predefined evaluation. Do not adapt on
the pilot tasks and also treat those same tasks as held-out evaluation. High
single-agent accuracy can saturate the benchmark even for hard-looking
questions; establish that empirically rather than assuming a task is difficult
for Astra or another strong model. Use mean scores and cost as additional
signals, and do not manufacture collaboration gains by withholding information
or tools from the single-agent baseline.

Offline integration demo (scripted arithmetic; no research-performance claim):

```bash
.venv/bin/python -m hayekmas.adapters.teams.evaluation \
  --config global_configs/compare_demo.json --out runs/comparison-demo-1
```

References: [EoM scientific experiments](https://arxiv.org/html/2606.02859v1#A4.SS4)
and the [original pass@k implementation](https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py).

## Compare conditions

```bash
.venv/bin/python scripts/compare_teams.py \
  --config global_configs/teams_demo.json \
  --seeds 7,17,29 --out runs/team-comparison --no-plots
```

The four conditions are `individual`, `random_fixed`, `self_selected_fixed`,
and `dynamic`. The individual condition is an **EoM-like singleton ablation of
this adapter**, not the full original engine or a paper-score reproduction.
The original engine remains the separate upstream baseline. Conditions use
matched task seeds but have different inference costs, reported per run.
The call budget applies per run; a comparison multiplies it by cells and seeds.
For OpenRouter, `budget.max_usd` is instead divided across all condition/seed
cells, so it is the total comparison allowance. Unused cell allowances are not
reassigned. Separate invocations have separate local budgets; use the OpenRouter
key cap to bound spending across invocations.

## Recorded outputs

| File | Contents |
|---|---|
| `replay.html`, `replay.json` | Event-by-event interaction replay with task selection, updated after completed tasks |
| `events.jsonl` | Full requests, responses, conversations, invitations, pledges, money and strategy updates |
| `agents/*.jsonl` | Agent-attributed event streams for behavioral annotation |
| `population.json` | Final wealth, membership, summaries, strategies and accounting |
| `metrics.json`, `metrics.csv` | Scores, memberships, wealth, pledges, inequality and membership turnover |
| `agent_features.csv` | Message/proposal/invitation/leave/reflection counts and contribution fractions |
| `communication_graph.json` | Directed message counts, including bid negotiation |
| `overview.png`, `overview.svg` | Static wealth, membership, contribution and communication figures |
| `manifest.json`, `usage.json` | Source hash, upstream commit, config, task IDs, status, call counters and OpenRouter token/cost totals |
| `api_usage.jsonl` (OpenRouter) | Durable before-request reservations and response costs, generation IDs, provider/model; no credentials |

`interrupted_state.json` is an audit snapshot, **not** a resumable checkpoint.
The replay retains all completed rounds after an interrupted run. It does not
present a partial round as completed. Role clustering and semantic annotations
are left for post-hoc analysis; the current features do not prove specialization.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_teams*.py' -v
```

Tests cover the economic ledger, voluntary membership, bounded negotiation,
independent ownership of commitments, observation isolation, reflection,
reproducibility, artifacts, interruption, and safe rendering of agent text.

### Evaluation and repair safeguards

The simple launcher with `split: "test"` restores its initial untrained population
before every test. Use the dedicated evaluator for a trained checkpoint. Only
audit events and usage counters span tests; each reset is visible in the replay.
The final population/accounting file describes the last disposable test episode,
while metrics and API usage cover the complete run.

Research and exact-task environments expose references only through the internal
training-repair accessor. Public tasks, solving observations, and test adaptation
do not receive the reference. Financial budget stops and blocked-provider errors
propagate from mutation calls and mark the run interrupted; they are not silently
counted as ordinary failed births.
