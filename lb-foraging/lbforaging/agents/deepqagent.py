import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from lbforaging.agents.foragingagent import BaseForagingAgent
from lbforaging.foraging.environment import Action

# Define DQN model architecture
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Experience replay memory
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, experience):
        self.memory.append(experience)
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DeepQLearningForagingAgent(BaseForagingAgent):
    """Foraging Agent with Deep Q-Learning Algorithm"""
    def __init__(self, agent_params):
        super().__init__(agent_params)
        
        # DQN parameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.epsilon_min = 0.01  # Minimum epsilon value
        self.batch_size = 64  # Batch size for training
        self.target_update = 10  # How often to update target network
        self.learning_rate = 0.001  # Learning rate
        self.memory_size = 10000  # Replay memory size
        
        # Environment parameters
        self.input_dim = 10  # State dimension (needs to be adjusted based on observation space)
        self.output_dim = len(Action)  # Number of possible actions
        
        # Initialize replay memory
        self.memory = ReplayMemory(self.memory_size)
        
        # Initialize networks
        self.policy_net = DQN(self.input_dim, self.output_dim)
        self.target_net = DQN(self.input_dim, self.output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Training variables
        self.steps_done = 0
        self.episode_rewards = []
        self.current_state = None
        self.current_action = None
        
    def preprocess_state(self, obs):
        """Convert observation to tensor for DQN input"""
        # Flatten and normalize the observation
        # This function should be adjusted based on your observation space
        state = np.array(obs, dtype=np.float32)
        
        # If state has less dimensions than input_dim, pad with zeros
        if state.shape[0] < self.input_dim:
            padding = np.zeros(self.input_dim - state.shape[0], dtype=np.float32)
            state = np.concatenate([state, padding])
        # If state has more dimensions, truncate
        elif state.shape[0] > self.input_dim:
            state = state[:self.input_dim]
            
        return torch.tensor([state], dtype=torch.float32)
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Random action
            return random.choice(list(Action))
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.policy_net(state)
                action_idx = q_values.max(1)[1].item()
                return list(Action)[action_idx]
    
    def optimize_model(self):
        """Train the model with a batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # Compute a mask of non-final states
        non_final_mask = torch.tensor([not done for done in batch.done], dtype=torch.bool)
        non_final_next_states = torch.cat([s for s, d in zip(batch.next_state, batch.done) if not d])
        
        # Prepare batch data
        state_batch = torch.cat(batch.state)
        action_indices = [list(Action).index(a) for a in batch.action]
        action_batch = torch.tensor(action_indices, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size, dtype=torch.float32)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        # Compute the expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    
    def step(self, obs):
        """Take a step in the environment"""
        # Preprocess state
        print(f"AGENT POSITION {self.position}")
        state = self.preprocess_state(obs)
        
        # Select action
        action = self.select_action(state)
        
        # Deduct survival cost
        self.energy -= self.survival_cost
        
        # Save current state and action
        if self.current_state is not None:
            # Calculate reward
            reward = self.energy  # Use energy level as reward
            
            # Store experience in replay memory
            self.memory.push(Experience(
                self.current_state, 
                self.current_action,
                reward,
                state,
                self.energy <= 0  # Done if agent has no energy
            ))
            
            # Train the model
            self.optimize_model()
            
            # Update target network
            if self.steps_done % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Save current state and action for next step
        self.current_state = state
        self.current_action = action
        self.steps_done += 1
        
        print(f"Energy level: {self.energy}, Steps done: {self.steps_done}")
        
        return action