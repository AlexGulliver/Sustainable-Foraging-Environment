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
        self.energy = 10  # Initial energy level
        self.tau = 0  # Moderate agent collection threshold
        self.invalid_action = False  # Flag to track invalid action

        # Q-learning parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.q_table = {}  # Q-table (state-action value)

    def notify_food_loaded(self, food_loaded):
        if food_loaded:
            self.energy += self.food_value
            print(f"Energy level increased by {self.food_value}")
        else:
            pass

    def get_state(self, obs):
        # A simple state representation based on the agent's energy level
        return int(self.energy)

    def update_q_value(self, state, action, reward, next_state):
        # If the state-action pair doesn't exist in Q-table, initialize it
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in Action}

        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in Action}

        # Q-learning update rule
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
        # Current state based on energy
        state = self.get_state(obs)

        # Choose action based on the current state
        action = self.choose_action(state)
        print(f"Chosen action: {action}")

        # Perform action and update the energy level (simplified for now)
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

        # Update Q-value based on the action and reward
        self.update_q_value(state, action, reward, next_state)

        # Show training progress
        print(f"Energy level: {self.energy}, Reward: {reward}, Next state: {next_state}")
        print(f"Updated Q-table: {self.q_table}")

        return action
