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
Experience = namedtuple(
    "Experience", ("state", "action", "reward", "next_state", "done")
)


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
        self.target_update = 10  # How often to update target network (steps)
        self.learning_rate = 0.001  # Learning rate
        self.memory_size = 10000  # Replay memory size

        # Output dimension is the number of possible actions
        self.output_dim = len(Action)
        
        # Initialize input_dim to None, will be set in the first step
        self.input_dim = None
        
        # Networks will be initialised after we know the input dimensions
        self.policy_net = None
        self.target_net = None
        self.optimiser = None
        
        # Initialise replay memory
        self.memory = ReplayMemory(self.memory_size)

        # Training variables
        self.steps_done = 0
        self.episode_rewards = []
        self.current_state = None
        self.current_action = None
        self.last_state = None
        self.last_action = None
        self.reward = 0

    def _initialise_networks(self, input_dim):
        """Initialise networks once we know the input dimension"""
        self.input_dim = input_dim
        
        # Initialise networks
        self.policy_net = DQN(self.input_dim, self.output_dim)
        self.target_net = DQN(self.input_dim, self.output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode

        # Initialise optimiser
        self.optimiser = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def get_state(self, obs):
        """Convert observation to a state representation"""
        return tuple(obs)

    def preprocess_state(self, obs):
        """Convert observation to tensor for DQN input"""
        # Flatten and normalize the observation
        state = np.array(obs, dtype=np.float32)
        
        # Initialize networks if this is the first time we're seeing data
        if self.input_dim is None:
            self._initialise_networks(len(state))
        
        # Handle case where observation dimension changes
        if len(state) != self.input_dim:
            print(f"Warning: Observation dimension changed from {self.input_dim} to {len(state)}. Reinitializing networks.")
            self._initialise_networks(len(state))

        return torch.tensor([state], dtype=torch.float32)

    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Random action
            return random.choice(list(Action))
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                action_idx = q_values.max(1)[1].item()
                return list(Action)[action_idx]

    def optimise_model(self):
        """Train the model with a batch from replay memory"""
        if len(self.memory) < self.batch_size or self.policy_net is None:
            return

        # Sample batch
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        # Compute a mask of non-final states
        non_final_mask = torch.tensor(
            [not done for done in batch.done], dtype=torch.bool
        )
        
        # Filter out None values that might occur before networks are initialised
        valid_next_states = [s for s, d in zip(batch.next_state, batch.done) if not d and s is not None]
        if valid_next_states:
            non_final_next_states = torch.cat(valid_next_states)
        else:
            # If there are no valid next states, we can't optimize yet
            return

        # Prepare batch data
        state_batch = torch.cat(batch.state)
        action_indices = [list(Action).index(a) for a in batch.action]
        action_batch = torch.tensor(action_indices, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)

        # Compute Q(s_t, a) - the model computes Q(s_t), then select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size, dtype=torch.float32)
        if non_final_next_states.size(0) > 0:  # Check if we have any non-final states
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(
                    non_final_next_states
                ).max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)

        # Compute Huber loss
        loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Optimise the model
        self.optimiser.zero_grad()
        loss.backward()
        # Clip gradients to stabilise training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimiser.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def step(self, obs):
        """Take a step in the environment"""

        # Convert observation to state
        current_state = self.get_state(obs)

        # Preprocess state for neural network
        state_tensor = self.preprocess_state(obs)

        # Select action once networks are initialised
        if self.policy_net is not None:
            action = self.select_action(state_tensor)
        else:
            # Default to random action if networks aren't initialised yet
            action = random.choice(list(Action))

        # Deduct survival cost
        self.energy = max(0, self.energy - self.survival_cost)

        # Store current state and action for potential reward update
        self.last_state = self.current_state
        self.last_action = action
        self.current_state = current_state

        print(f"Energy level: {self.energy}, Steps done: {self.steps_done}")
        print(f"AGENT POSITION {self.position}")
        # Increment steps
        self.steps_done += 1

        return action

    def receive_reward(self, reward):
        """Process reward from the environment"""
        self.reward = reward

        # If we have a previous state and action, store experience in replay memory
        if self.last_state is not None and self.last_action is not None and self.policy_net is not None:
            # Preprocess states for storage
            last_state_tensor = self.preprocess_state(self.last_state)
            current_state_tensor = self.preprocess_state(self.current_state)

            # Store experience in replay memory
            self.memory.push(
                Experience(
                    last_state_tensor,
                    self.last_action,
                    reward,
                    current_state_tensor,
                    self.energy <= 0,  # Done if agent has no energy
                )
            )

            # Train the model if networks are initialised
            self.optimise_model()

            # Update target network
            if self.steps_done % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        print(f"Received reward: {reward}, action {self.last_action}")
