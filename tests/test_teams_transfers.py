import json
import unittest

from hayekmas.adapters.teams.agent import TeamAction
from hayekmas.adapters.teams.config import TeamConfig
from hayekmas.adapters.teams.env import ExactTaskEnv, ResearchTaskEnv, Task
from hayekmas.adapters.teams.mas import TeamMAS
from hayekmas.adapters.teams.replay import replay_data


class ChainPolicy:
    label = "test"

    def __init__(self, same_team=False):
        self.same_team = same_team
        self.observations = []

    def respond(self, phase, agent, observation, max_tokens):
        self.observations.append((phase, observation))
        step = observation.get("step") or 0
        if phase in {"negotiate", "contribute"}:
            target = "team-1" if self.same_team or step == 0 else "team-2"
            # First team pledges 6 and 2; second team pledges 3 and 3.
            amount = (6 if agent.name == "agent-0" else 2) if target == "team-1" else 3
            result = {"contribution": amount if agent.team_tag == target else 0, "message": "proposal"}
        elif phase == "discuss":
            result = {
                "candidate": "7" if observation["must_finalize"] else "PUBLIC STEP",
                "final": observation["must_finalize"],
            }
        elif phase == "vote":
            result = {"candidate_id": observation["candidates"][0]["id"]}
        else:
            result = {}
        return json.dumps(result)


def setup(same_team=False):
    mas = TeamMAS(TeamConfig(num_agents=4, condition="self_selected_fixed", max_steps=2), ChainPolicy(same_team))
    mas.team_manager.create_team(mas.agents[:2], 0)
    mas.team_manager.create_team(mas.agents[2:], 0)
    mas.round = 1
    return mas


class TransferTests(unittest.TestCase):
    def test_chain_conserves_money_and_pays_previous_members_equally(self):
        mas = setup()
        metric = mas.run_one_episode(ExactTaskEnv(Task("x", "3+4", 7), 12))
        self.assertEqual([a.wealth for a in mas.agents], [17, 21, 23, 23])
        self.assertEqual((mas.bid_paid_total, mas.bid_transfer_total, mas.bid_burn_total), (14, 6, 8))
        self.assertEqual(metric["bid_income"], {"agent-0": 3, "agent-1": 3, "agent-2": 0, "agent-3": 0})
        self.assertEqual(metric["paid"], {"agent-0": 6, "agent-1": 2, "agent-2": 3, "agent-3": 3})
        self.assertEqual(metric["step_count"], 2)
        mas.assert_accounting()
        for phase, observation in mas.policy.observations:
            self.assertIn(observation["you"]["name"], {a.name for a in mas.agents})
            self.assertIn("wealth", observation["you"])
        events = replay_data(mas, "complete")["rounds"][0]["events"]
        self.assertEqual(len([e for e in events if e["event"] == "bid_transfer"]), 2)
        later = [o for p, o in mas.policy.observations if p == "negotiate" and o["step"] == 1]
        self.assertTrue(all("PUBLIC STEP" in o["environment_state"]["state"] for o in later))
        self.assertTrue(any("proposal" in o["discussion"] for o in later))

    def test_no_transfer_between_tasks(self):
        mas = setup()
        for i in range(2):
            mas.run_one_episode(ExactTaskEnv(Task(str(i), "3+4", 7), 12))
        firsts = [e for e in mas.events if e["event"] == "bid_transfer" and e["step"] == 0]
        self.assertEqual(len(firsts), 2)
        self.assertTrue(all(e["receiver"] is None and e["credits"] == {} and e["burned"] == 8 for e in firsts))
        mas.assert_accounting()

    def test_same_team_win_redistributes_equal_income(self):
        mas = setup(True)
        mas.run_one_episode(ExactTaskEnv(Task("x", "3+4", 7), 12))
        # First 8 burned; second 8 returned equally (4 each); reward 6 each.
        self.assertEqual([a.wealth for a in mas.agents], [18, 26, 20, 20])
        self.assertEqual(mas.bid_transfer_total, 8)
        mas.assert_accounting()

    def test_intermediate_actions_do_not_grade_or_end_tasks(self):
        env = ExactTaskEnv(Task("x", "question", "PRIVATE"), 12)
        env.initialize()
        self.assertEqual(env.apply(TeamAction("PUBLIC", final=False)), 0)
        self.assertFalse(env.reach_termination())
        self.assertNotIn("PRIVATE", env.get_state_description())
        calls = []
        research = ResearchTaskEnv(
            Task("r", "question", "private rubric"), 12, lambda prompt: calls.append(prompt) or "SCORE: 1\nREASON: ok"
        )
        research.initialize()
        self.assertEqual(research.apply(TeamAction("intermediate", final=False)), 0)
        self.assertFalse(research.reach_termination())
        self.assertEqual(calls, [])
        self.assertEqual(research.apply(TeamAction("answer", final=True)), 12)
        self.assertEqual(len(calls), 1)

    def test_invalid_intermediate_candidate_at_limit_does_not_fake_final_answer(self):
        mas = setup()
        old = mas.policy.respond
        mas.policy.respond = lambda phase, a, o, cap: (
            json.dumps({"candidate": "7", "final": False}) if phase == "discuss" else old(phase, a, o, cap)
        )
        metric = mas.run_one_episode(ExactTaskEnv(Task("x", "3+4", 7), 12))
        self.assertEqual(metric["score"], 0)
        self.assertTrue(mas.invalid_actions > 0)
        mas.assert_accounting()
