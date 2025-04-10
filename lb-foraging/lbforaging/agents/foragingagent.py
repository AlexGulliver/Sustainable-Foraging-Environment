from abc import ABC, abstractmethod
from lbforaging.agents import BaseAgent
from lbforaging.foraging.environment import Action


class BaseForagingAgent(BaseAgent, ABC):
    """Abstract Base Class for Foraging Agents"""

    def __init__(self, agent_params):
        self.name = "Base Foraging Agent"
        self.energy = 10  # Initial energy level
        self.survival_cost = agent_params.get("survival_cost", 1)  # Default value
        self.food_value = 2
        self.movement_cost = 1
        self.carry_capacity = agent_params.get("carry_capacity", 1)
        self.tau = agent_params.get("tau", 0)
        self.k = agent_params.get("k", 0)
        self.invalid_action = False
        self.reward = 0

    @abstractmethod
    def step(self, obs):
        """Abstract method to define a step of the agent's decision-making."""
        pass

    def notify_food_loaded(self, food_loaded):
        """Updates the energy when food is loaded."""
        if food_loaded:
            self.energy += self.food_value
            print(f"Energy level increased by {self.food_value}")
        else:
            pass
