import random
from lbforaging.agents import BaseAgent
from lbforaging.foraging.environment import Action
import numpy as np

class ForagingAgent(BaseAgent):
    name = "Foraging Agent"

    def __init__(self, agent_params):
        eta, self.carry_capacity, self.survival_cost, self.tau, self.k = tuple(agent_params.values())
        self.movement_cost = 1
        self.food_value = 1
        self.energy = 5  # Initial energy level
        self.tau = 0  # Moderate agent collection threshold
        self.invalid_action = False  # Flag to track invalid action

    def notify_food_loaded(self, food_loaded):
        if food_loaded:
            self.energy += self.food_value
            print(f"Energy level increased by {self.food_value}")
        else:
            pass


    def step(self, obs):
        print(f"Current energy level: {self.energy}")

        action = np.random.choice(list(Action))  # Choose an action randomly

        return action

