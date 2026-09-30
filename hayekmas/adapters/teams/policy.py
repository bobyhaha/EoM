"""All agents receive identical interfaces and choose behavior for themselves."""

import ast
import json
import random

from hayekmas.utils.llm import LLMConfig, get_llm_client


INSTRUCTIONS = {
    "assess_bid": (
        "Make one sealed personal pledge. Assess relevant expertise and the main uncertainty in at most "
        "two short sentences; do not solve the entire problem or negotiate. Other current pledges are hidden. "
        'Return {"message":"brief assessment", "contribution":number}. Only your money may be pledged, '
        "between zero and your available wealth. Winning members pay their own pledges times bid_cost_rate; "
        "the eventual reward is split equally among winning members."
    ),
    "draft": (
        "Independently produce your best complete answer. Derive a checklist of requested deliverables "
        "from the public question, check units and assumptions, and distinguish facts from uncertainty. "
        "No teammate draft is visible yet. Your answer must stand alone; it may be submitted if revision fails. "
        'Return {"candidate":"complete answer", "final":true, "checklist":["requested deliverable"]}, '
        'or {"abstain":true}. Do not invent missing experimental details or sources.'
    ),
    "independent_check": (
        "Independently solve and check the public question before seeing any teammate draft. "
        "Prioritize calculations, units, assumptions and easily missed requested deliverables. "
        "Write a complete alternative answer that can stand alone, stating uncertainty explicitly. "
        'Return {"candidate":"complete answer", "final":true, "checklist":["requested deliverable"]}, '
        'or {"abstain":true}. Do not invent missing experimental details or sources.'
    ),
    "review": (
        "Compare the primary draft with the independently generated alternatives and the public question. "
        "For each requested deliverable, check coverage and correctness. Recompute disputed quantities. "
        "Identify concrete errors, disagreements and missing evidence; agreement is not verification. "
        "If an input is truncated, state what you could not verify. Use only supplied information; "
        "do not invent sources or claim to have run tools. "
        'Return {"checks":[{"requirement":"text", "status":"covered|missing|uncertain", '
        '"evidence":"calculation or reason"}], "issues":["specific correction or disagreement"]}.'
    ),
    "revise": (
        "Produce one complete final answer using the public question, independent drafts and reviews. "
        "Address each checklist item and review issue; resolve disagreements by calculation or evidence, "
        "not majority vote. Reject unsupported corrections and state unresolved uncertainty in the answer. "
        "Check truncated-input warnings. Do not just summarize the discussion or invent missing facts. "
        'Return {"candidate":"complete revised answer", "final":true, '
        '"checklist":["requested deliverable"], "resolutions":["how an issue was resolved or remains uncertain"]}, '
        'or {"abstain":true}. The candidate field is the entire submitted answer.'
    ),
    "reflect": 'Choose whether to pay for reflection now. Return {"reflect": true} or {"reflect": false}.',
    "inspect": 'Inspect only the supplied experience. Return {"analysis": "your observations"}.',
    "improve": 'Use your inspection to update your future strategy. Return {"strategy": "new strategy"}.',
    "formation": (
        'Choose one action: {"action":"invite", "target":"agent name", "text":"message"}, '
        '{"action":"accept", "invitation":"invitation id"}, {"action":"leave"}, '
        '{"action":"message", "target":"agent name", "text":"message"}, or {"action":"pass"}. '
        "Only unteamed agents may accept. An invitation from a teammate admits a new member "
        "to that team. Invites expire after this formation phase. No kicking or unilateral merging."
    ),
    "contribute": 'As a singleton choose your personal contribution. Return {"contribution": number}.',
    "negotiate": (
        "Discuss how much each teammate should contribute using the visible pledges and conversation. "
        "You may ask others to change their amounts, explain a proposed allocation, and revise your own pledge. "
        'Return {"message":"text", "contribution":number}. The contribution sets ONLY your pledge. '
        "Your last valid pledge at the end of this fixed discussion window is binding if your team wins. "
        "You may contribute zero. No agent can pledge money on behalf of another."
    ),
    "discuss": (
        'Use the team channel freely. Return {"message":"text", "candidate":"text or null", "final":true}. '
        "Set final=false for an intermediate public step; final=true submits an answer and ends the task. "
        "You can discuss without a candidate. On must_finalize=true only final answers are eligible. "
        "Everyone has the same actions. "
        "After the fixed discussion window, every member votes for a candidate."
    ),
    "finalize": (
        "Discussion has ended. Propose your best complete final answer using the task and shared notes. "
        'Return {"candidate":"your complete answer", "final":true}, or explicitly abstain with '
        '{"abstain":true}. State unresolved assumptions or uncertainty in the answer. '
        "Text in message alone is not submitted. Intermediate steps and empty candidates are not eligible. "
        "Every member has the same opportunity; the team votes on eligible final candidates. "
        "A malformed response receives at most one correction attempt. Abstention is not retried."
    ),
    "vote": 'Choose a candidate ID from the supplied list. Return {"candidate_id":"id"}. Ties are broken by seeded lottery.',
}


class ModelPolicy:
    label = "llm"

    def __init__(self, model, budget=None):
        if model.get("api") == "openrouter":
            from .openrouter import OpenRouterClient

            if not model.get("name") or model.get("api_base"):
                raise ValueError("OpenRouter requires an explicit model name and uses its fixed official endpoint")
            self.client = OpenRouterClient(
                model["name"], budget=budget or {}, api_key=model.get("api_key"), **model.get("extra_kwargs", {})
            )
            return
        cfg = LLMConfig(**model)
        if cfg.api not in {"localhost", "litellm", "together"}:
            raise ValueError("Team prototype supports localhost, litellm or together model APIs")
        if not cfg.name:
            raise ValueError("Set an explicit model name for reproducible runs")
        kwargs = dict(cfg.extra_kwargs)
        kwargs.update(api_key=cfg.api_key, api_base=cfg.api_base)
        if cfg.api == "localhost":
            kwargs.update(max_retries=1, timeout=60)
        self.client = get_llm_client(cfg.api, cfg.name, **kwargs)
        self.client.MAX_EMPTY_RETRIES = 1

    def respond(self, phase, agent, observation, max_tokens):
        return self.client.generate(
            json.dumps(
                {"instruction": INSTRUCTIONS[phase], "observation": observation},
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ),
            system_prompt=agent.get_system_prompt(),
            max_tokens=max_tokens,
        )


class DemoPolicy:
    """Scripted mechanism smoke test; never evidence of emergent behavior."""

    label = "scripted_demo"

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def respond(self, phase, agent, observation, max_tokens):
        if phase == "reflect":
            result = {"reflect": observation["round"] > 0 and observation["round"] % 4 == 0}
        elif phase == "inspect":
            result = {"analysis": "Compare money spent with equal reward share in the supplied history."}
        elif phase == "improve":
            result = {
                "strategy": "Preserve capital by contributing a small fraction of wealth. Check arithmetic before voting."
            }
        elif phase == "formation":
            invitations = observation["invitations"]
            if observation["round"] > 0 and observation["turn"] == 0 and agent.team_tag is not None:
                result = {"action": "leave"}
            elif agent.team_tag is None and invitations:
                result = {"action": "accept", "invitation": invitations[0]["id"]}
            elif agent.team_tag is None:
                own = int(agent.name.split("-")[-1])
                peer = f"agent-{own ^ (2 if (observation['round'] // 5) % 2 else 1)}"
                names = {p["name"] for p in observation["roster"] if p["team"] is None}
                result = (
                    {"action": "invite", "target": peer, "text": "Would you like to work together?"}
                    if peer in names
                    else {"action": "pass"}
                )
            else:
                result = {"action": "pass"}
        elif phase in {"contribute", "negotiate", "assess_bid"}:
            result = {
                "contribution": round(agent.wealth * self.rng.uniform(0.03, 0.12), 6),
                "message": ("I can attempt this arithmetic. This is my one binding pledge."
                            if phase == "assess_bid" else
                            "Here is my current pledge. Please consider contributing an affordable amount too."),
            }
        elif phase in {"discuss", "finalize", "draft", "independent_check", "revise"}:
            # Uses only the public problem, never the environment's hidden answer.
            problem = observation["task"]["problem"]
            try:
                values = ast.literal_eval(problem.split(":", 1)[1].strip())
                answer = str(sum(values))
            except (SyntaxError, ValueError, TypeError, IndexError):
                answer = None
            result = {
                "message": "I independently checked the public arithmetic.",
                "candidate": answer,
                "final": observation["must_finalize"],
                **({"checklist": ["Compute the sum"], "resolutions": ["Checked the arithmetic"]}
                   if phase in {"draft", "independent_check", "revise"} else {}),
            }
            if phase in {"finalize", "draft", "independent_check", "revise"} and answer is None:
                result = {"abstain": True}
        elif phase == "review":
            result = {"checks": [{"requirement": "Compute the sum", "status": "covered",
                                  "evidence": "Recomputed the public arithmetic"}], "issues": []}
        else:
            candidates = observation["candidates"]
            result = {"candidate_id": candidates[0]["id"] if candidates else None}
        return json.dumps(result)
