"""Describe pledge revisions and independently reconcile recorded team auctions."""

from collections import Counter, defaultdict
import math


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8)


def negotiation_trace(events):
    counts, errors, groups = Counter(), [], []
    histories = defaultdict(lambda: defaultdict(list))
    previous_members = []
    auction = None
    initial_names = next(
        ({a["name"] for a in e.get("agents", [])} for e in events if e["event"] == "initialized"), set()
    )
    rate = next((e.get("config", {}).get("bid_cost_rate", 1.0) for e in events if e["event"] == "initialized"), 1.0)

    def check(condition, requirement, step, agent=None):
        if not condition:
            errors.append({"requirement": requirement, "step": step, "agent": agent})

    for event in events:
        kind, step = event["event"], event.get("step")
        if kind == "pledge":
            histories[event["group"]][event["agent"]].append(event["amount"])
        elif kind == "auction":
            auction = event
            counts["auctions"] += 1
            check(set(histories) == set(event["bids"]), "All bidding groups have a recorded pledge history", step)
            check(
                {name for people in histories.values() for name in people} == set(event["contributions"]),
                "All personal pledges are recorded",
                step,
            )
            check(
                bool(initial_names) and set(event["opening_wealth"]) == initial_names,
                "Opening wealth covers the full fixed population",
                step,
            )
            for group, people in histories.items():
                first = {name: amounts[0] for name, amounts in people.items()}
                final = {name: amounts[-1] for name, amounts in people.items()}
                revised = [name for name in people if len(people[name]) > 1 and not close(first[name], final[name])]
                eligible = sum(len(values) > 1 for values in people.values())
                counts["revisable_personal_pledges"] += eligible
                counts["revised_personal_pledges"] += len(revised)
                total = math.fsum(final.values())
                check(close(total, event["bids"][group]), "Team bid equals sum of final personal pledges", step)
                for name, amount in final.items():
                    counts["binding_values_checked"] += 1
                    check(close(amount, event["contributions"][name]), "Last personal pledge is binding", step, name)
                won = group == event["winner"]
                unequal = len(people) > 1 and max(final.values()) - min(final.values()) > 1e-8
                zeros = [name for name, amount in final.items() if amount == 0]
                if won and len(people) > 1:
                    counts["winning_teams"] += 1
                    counts["unequal_winning_teams"] += unequal
                    counts["winning_teams_with_zero_money_contributor"] += bool(zeros)
                groups.append(
                    {
                        "step": step,
                        "group": group,
                        "first": first,
                        "final": final,
                        "total_bid": total,
                        "winner": won,
                        "revised_agents": revised,
                        "revisable_pledges": eligible,
                        "unequal": unequal,
                        "zero_money_contributors": zeros,
                    }
                )
            maximum = max(event["bids"].values(), default=0)
            winner = event["winner"]
            check(
                (winner is None and maximum == 0)
                or (winner in event["bids"] and maximum > 0 and close(event["bids"][winner], maximum)),
                "Positive highest total bid wins",
                step,
            )
            paid = math.fsum(event["contributions"][name] for name in event["members"]) * rate
            check(close(paid, event["paid"]), "Winning members pay their own pledges at the bid cost rate", step)
            credits = (
                {name: paid / len(previous_members) for name in previous_members} if previous_members and winner else {}
            )
            check(
                set(credits) == set(event["credits"])
                and all(close(value, event["credits"][name]) for name, value in credits.items()),
                "Current bid is split equally among previous winning members",
                step,
            )
            check(
                close(event["burned"], paid if not credits else 0),
                "Only bids without a previous recipient are burned",
                step,
            )
            previous_members = event["members"]
            histories = defaultdict(lambda: defaultdict(list))
        elif kind == "settlement" and auction is not None:
            counts["settlements"] += 1
            check(step == auction.get("step"), "Settlement belongs to the latest auction", step)
            check(set(event["members"]) == set(auction["members"]), "Rewards go to current winning members", step)
            check(set(event["wealth"]) == set(auction["opening_wealth"]), "Settled wealth covers every agent", step)
            share = event["reward"] / len(event["members"]) if event["members"] else 0
            check(close(share, event["share"]), "Reward is shared equally among winning members", step)
            for name, opening in auction["opening_wealth"].items():
                expected = opening - (auction["contributions"][name] * rate if name in auction["members"] else 0)
                expected += auction["credits"].get(name, 0) + (share if name in event["members"] else 0)
                counts["wealth_values_checked"] += 1
                check(close(expected, event["wealth"][name]), "Per-auction personal wealth reconciles", step, name)
    check(counts["auctions"] > 0, "Completed episode has recorded auctions", None)
    check(counts["auctions"] == counts["settlements"], "Every recorded auction has a settlement", None)
    return {
        "groups": groups,
        "counts": dict(counts),
        "errors": errors,
        "interpretation": "Revision means a person's final pledge differs from their first pledge. Initial pledges may already respond to others. Zero money contribution does not imply zero intellectual contribution.",
    }
