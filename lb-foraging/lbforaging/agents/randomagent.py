import random
from lbforaging.agents.foragingagent import BaseForagingAgent
from lbforaging.foraging.environment import Action
import numpy as np


class RandomForagingAgent(BaseForagingAgent):
    """Foraging Agent that chooses actions randomly"""

    def __init__(self, agent_params):
        super().__init__(agent_params)

    def get_state(self, obs):
        return tuple(obs)

    def choose_action(self, state):
        # Randomly choose an action from the available actions
        return np.random.choice(list(Action))

    def receive_reward(self, reward):
        pass

    def step(self, obs):

        state = self.get_state(obs)

        action = self.choose_action(state)
        # print(f"Chosen action: {action}")

        self.energy -= self.survival_cost  # Deduct survival cost

        return action
