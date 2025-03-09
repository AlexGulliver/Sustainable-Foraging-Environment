import random
from lbforaging.agents import BaseAgent
from lbforaging.foraging.environment import Action

class ForagingAgent(BaseAgent):
    name = "Foraging Agent"


    def __init__(self, agent_params):
        eta, self.carry_capacity, self.survival_cost, self.tau, self.k = tuple(agent_params.values())
        self.energy = 0
        self.tau = 0  # Moderate agent collection threshold


    def step(self, obs):
        action = random.choice(obs.actions)

        if action in [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]:
            self.energy -= self.survival_cost

        if action == Action.LOAD:
            if obs["food_collected"].get(self.id, False):  # Check if agent successfully loaded food
                self.energy += self.carry_capacity

        return action


    def _successful_load(self, obs):
        """
        Check if the agent successfully loads food based on the observation.
        This assumes the environment provides info about food collection.
        """
        return obs.get("food_collected", False)  # Adjust this based on how env reports success
