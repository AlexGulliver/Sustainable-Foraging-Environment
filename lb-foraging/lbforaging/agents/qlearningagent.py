import random
from lbforaging.agents.foragingagent import BaseForagingAgent
from lbforaging.foraging.environment import Action
import numpy as np

class QLearningForagingAgent(BaseForagingAgent):
    """Foraging Agent with Q-Learning Algorithm"""

    def __init__(self, agent_params):
        super().__init__(agent_params)
        # Q-learning parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.q_table = {}  # Q-table (state-action value)

    def get_state(self, obs):
        return tuple(obs)

    def update_q_value(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in Action}

        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in Action}

        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        self.q_table[state][action] += self.alpha * (reward + self.gamma * self.q_table[next_state][best_next_action] - self.q_table[state][action])

    def choose_action(self, state):
        # Epsilon-greedy strategy for exploration vs exploitation
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: choose a random action
            return np.random.choice(list(Action))
        else:
            # Exploitation: choose the best action based on Q-values
            if state not in self.q_table:
                return np.random.choice(list(Action))
            return max(self.q_table[state], key=self.q_table[state].get)

    def step(self, obs):
        state = self.get_state(obs)
        print(f"AGENT POSITION {self.position}")
        action = self.choose_action(state)
        # print(f"Chosen action: {action}")

        self.energy -= self.survival_cost  # Deduct survival cost

        reward = self.energy  # Reward is the agent's current energy level
        next_state = self.get_state(obs)

        self.update_q_value(state, action, reward, next_state)

        print(f"Energy level: {self.energy}, Reward: {reward}, Next state: {next_state}")
        # print(f"Updated Q-table: {self.q_table}")

        return action

