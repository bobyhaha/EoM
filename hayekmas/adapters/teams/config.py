from dataclasses import dataclass, fields
import math


@dataclass
class TeamConfig:
    interaction_protocol: str = "legacy"
    round_schedule: str = "compact"
    coordination_fee_lambda: float = 0.0
    condition: str = "dynamic"
    seed: int = 7
    num_agents: int = 12
    rounds: int = 12
    max_steps: int = 3
    initial_wealth: float = 20.0
    reward: float = 12.0
    reflection_cost: float = 2.0
    bid_cost_rate: float = 1.0
    negotiation_interval: int = 5
    max_team_size: int = 4
    bidding_turns: int = 2
    formation_turns: int = 3
    discussion_turns: int = 2
    finalization_enabled: bool = True
    bidding_mode: str = "negotiated"
    collaboration_mode: str = "discussion"
    bid_tokens: int = 256
    fixed_team_size: int = 2
    action_tokens: int = 256
    solution_tokens: int = 256
    judge_tokens: int = 1024
    environment_tokens: int = 2048
    inspect_tokens: int = 256
    update_tokens: int = 256
    evidence_tokens: int = 2048
    summary_tokens: int = 192
    context_tokens: int = 8192
    max_calls: int = 10000

    def __post_init__(self):
        if self.interaction_protocol not in {"legacy", "rounds"}:
            raise ValueError("interaction_protocol must be legacy or rounds")
        if self.round_schedule not in {"compact", "full"}:
            raise ValueError("round_schedule must be compact or full")
        if self.interaction_protocol == "rounds" and (
            self.bidding_mode != "negotiated" or self.collaboration_mode != "discussion"
            or not self.finalization_enabled
        ):
            raise ValueError("rounds protocol requires negotiated bids, free discussion and finalization")
        if type(self.finalization_enabled) is not bool:
            raise ValueError("finalization_enabled must be a boolean")
        if self.bidding_mode not in {"negotiated", "sealed"}:
            raise ValueError("bidding_mode must be negotiated or sealed")
        if self.collaboration_mode not in {"discussion", "reviewed"}:
            raise ValueError("collaboration_mode must be discussion or reviewed")
        if self.collaboration_mode == "reviewed" and not self.finalization_enabled:
            raise ValueError("reviewed collaboration requires finalization_enabled")
        if self.condition not in {"individual", "random_fixed", "self_selected_fixed", "dynamic"}:
            raise ValueError("Unknown team condition")
        for name in ("initial_wealth", "reward", "reflection_cost", "bid_cost_rate", "coordination_fee_lambda"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.coordination_fee_lambda > 0 and (self.interaction_protocol != "rounds" or self.condition != "dynamic"):
            raise ValueError("Positive coordination fees require dynamic round teaming so members can decline and leave")
        if self.bid_cost_rate > 1:
            raise ValueError("bid_cost_rate must be in [0, 1]")
        for f in fields(self):
            if f.type is int:
                value = getattr(self, f.name)
                if type(value) is not int or value < (0 if f.name == "seed" else 1):
                    raise ValueError(f"{f.name} must be a positive integer (seed may be zero)")
        if (
            min(
                self.action_tokens,
                self.bid_tokens,
                self.solution_tokens,
                self.judge_tokens,
                self.inspect_tokens,
                self.update_tokens,
                self.summary_tokens,
            )
            < 64
        ):
            raise ValueError("Generation and summary budgets must be at least 64 tokens")
        if self.context_tokens < 2048 or self.evidence_tokens < 128:
            raise ValueError("context_tokens >= 2048 and evidence_tokens >= 128 are required")
        if self.evidence_tokens + self.update_tokens + 1024 > self.context_tokens:
            raise ValueError("context_tokens must leave room for evidence, strategy and instructions")
        if self.fixed_team_size > self.max_team_size:
            raise ValueError("fixed_team_size exceeds the engineering team-size cap")


class TokenBudget:
    """Explicit reference tokenizer; provider max_tokens limits generation separately.

    This is an o200k_base text-token budget, not a claim about every provider's
    native tokenizer, chat framing, or hidden reasoning tokens.
    """

    def __init__(self):
        import tiktoken

        self.encoding = tiktoken.get_encoding("o200k_base")

    def count(self, text):
        return len(self.encoding.encode(text, disallowed_special=()))

    def clip(self, text, limit, *, tail=False):
        tokens = self.encoding.encode(str(text), disallowed_special=())
        if limit <= 0:
            return ""
        selected = tokens[-limit:] if tail else tokens[:limit]
        # Decode only complete UTF-8 characters at a truncation boundary.
        return self.encoding.decode_bytes(selected).decode("utf-8", errors="ignore")
