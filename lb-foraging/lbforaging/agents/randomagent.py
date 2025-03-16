import random
from lbforaging.agents import BaseAgent
from lbforaging.foraging.environment import Action
from lbforaging.agents.foragingagent import ForagingAgent
import numpy as np

class RandomForagingAgent(ForagingAgent):
    name = "Random Foraging Agent"

    def __init__(self, agent_params):
        super().__init__(agent_params)  # Inherit from ForagingAgent
        # You can customize further parameters or logic specific to the random agent here.

    def choose_action(self, state):
        # Override the choose_action method to always choose a random action
        return np.random.choice(list(Action))

    def step(self, obs):
        # Inherited step method, but will choose random actions instead of using Q-learning
        state = self.get_state(obs)

        # Choose action randomly
        action = self.choose_action(state)
        print(f"Chosen action: {action}")

        # Perform action and update energy level (simplified for now)
        self.energy -= self.survival_cost  # Deduct survival cost

        # Assuming reward is simply the food value when food is loaded (you can modify this logic)
        food_loaded = True  # This should be determined by the environment, simplified for now
        if food_loaded:
            self.energy += self.food_value
            reward = self.food_value
        else:
            reward = -self.survival_cost  # Penalize for survival cost without food

        # Get the next state
        next_state = self.get_state(obs)

        # Show progress (no Q-learning update in this version)
        print(f"Energy level: {self.energy}, Reward: {reward}, Next state: {next_state}")

        return action
