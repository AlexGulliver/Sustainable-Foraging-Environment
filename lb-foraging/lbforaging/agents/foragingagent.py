import random
from lbforaging.agents import BaseAgent
from lbforaging.foraging.environment import Action
import numpy as np

class ForagingAgent(BaseAgent):
    name = "Foraging Agent"


    def __init__(self, agent_params):
        eta, self.carry_capacity, self.survival_cost, self.tau, self.k = tuple(agent_params.values())
        self.energy = 5  # Initial energy level 
        self.tau = 0  # Moderate agent collection threshold


    def step(self, obs):
        print(f"{self.energy} ENERGY LEVEL")
        action = np.random.choice(list(Action))  # ✅ Correct: Returns a single Action object


        if action in [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]:
            self.energy -= self.survival_cost

        if action == Action.LOAD:
            # if obs["food_collected"].get(self.id, False):  # Check if agent successfully loaded food
            #     self.energy += self.carry_capacity
            pass

        return action
