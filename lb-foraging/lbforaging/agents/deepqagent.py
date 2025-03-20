import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from lbforaging.agents.foragingagent import BaseForagingAgent
from lbforaging.foraging.environment import Action

# Define the Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DeepQLearningForagingAgent(BaseForagingAgent):
    def __init__(self, agent_params):
        super().__init__(agent_params)
        self.alpha = 0.001  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.memory = deque(maxlen=10000)  # Experience replay buffer
        self.batch_size = 64
        
        self.input_dim = 3  # Example: energy, position_x, position_y
        self.output_dim = len(Action)
        
        self.model = DQN(self.input_dim, self.output_dim)
        self.target_model = DQN(self.input_dim, self.output_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()
        self.update_target_counter = 0
        self.target_update_freq = 100  # Update target network periodically

    def get_state(self, obs):
        return np.array([self.energy, self.position[0], self.position[1]], dtype=np.float32)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.choice(list(Action))
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return Action(torch.argmax(q_values).item())

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor([a.value for a in actions], dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target_counter += 1
        if self.update_target_counter % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def step(self, obs):
        state = self.get_state(obs)
        action = self.choose_action(state)
        self.energy -= self.survival_cost
        reward = self.energy
        next_state = self.get_state(obs)
        done = self.energy <= 0  # Example condition

        self.store_experience(state, action, reward, next_state, done)
        self.train()

        return action