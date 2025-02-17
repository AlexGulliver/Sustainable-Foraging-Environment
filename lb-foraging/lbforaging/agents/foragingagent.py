import random
from lbforaging.agents import BaseAgent

class ForagingAgent(BaseAgent):
    name = "Foraging Agent"

    def __init__(self, agent_params):
        eta, self.carry_capacity, self.survival_cost, self.tau, self.k = tuple(agent_params.values())
        self.energy = 0
        self.tau = 0 # Moderate agent collection threshold

    def step(self, obs):
        # Observe current state s
        state = self._make_state(obs)

        # Choose to be greedy or moderate
        action = random.choice(obs.actions)

        return action

