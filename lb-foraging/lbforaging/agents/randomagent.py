import random
from lbforaging.agents.foragingagent import BaseForagingAgent
from lbforaging.foraging.environment import Action
import numpy as np

class RandomForagingAgent(BaseForagingAgent):
    """Foraging Agent that chooses actions randomly"""

    def __init__(self, agent_params):
        super().__init__(agent_params)
        self.q_table = {}  # Q-table is no longer needed for random agent

    def get_state(self, obs):
        # Simple state representation based on energy
        return int(self.energy)

    def choose_action(self, state):
        # Randomly choose an action from the available actions
        return np.random.choice(list(Action))

    def step(self, obs):
        # Get the current state
        state = self.get_state(obs)

        # Choose a random action
        action = self.choose_action(state)
        print(f"Chosen action: {action}")

        # Perform action and update the energy level (simplified for now)
        self.energy -= self.survival_cost  # Deduct survival cost

        # Assuming food loading logic is simplified for now
        food_loaded = random.choice([True, False])  # Randomly decide if food is loaded
        if food_loaded:
            self.energy += self.food_value
            reward = self.food_value
        else:
            reward = -self.survival_cost  # Penalize for survival cost without food

        # Show agent's current status
        print(f"Energy level: {self.energy}, Reward: {reward}, State: {state}")

        return action
