import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from lbforaging.agents.foragingagent import BaseForagingAgent
from lbforaging.foraging.environment import Action

# Experience replay memory
Experience = namedtuple(
    "Experience",
    ("state", "action", "reward", "next_state", "intrinsic_reward", "done"),
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


# Forward Model for predicting next state
class ForwardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ForwardModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # State and action encoders
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # Add extra layer
            nn.ReLU()
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), nn.ReLU()
        )

        # Combined network to predict next state
        self.combined = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state, action):
        # Encode state and action
        state_encoding = self.state_encoder(state)

        # One-hot encode action
        action_one_hot = torch.zeros(
            action.size(0), self.action_dim, device=action.device
        )
        action_one_hot.scatter_(1, action.unsqueeze(1), 1)

        action_encoding = self.action_encoder(action_one_hot)

        # Combine encodings and predict next state
        combined = torch.cat([state_encoding, action_encoding], dim=1)
        next_state_pred = self.combined(combined)

        return next_state_pred

# Inverse Model for predicting actions from state transitions
class InverseModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(InverseModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state, next_state):
        # Concatenate state and next_state
        x = torch.cat([state, next_state], dim=1)
        return self.model(x)  # Returns logits for action classification


class CuriosityDrivenDQNAgent(BaseForagingAgent):
    """Foraging Agent with Deep Q-Learning and Curiosity-Driven Exploration"""

    def __init__(self, agent_params):
        super().__init__(agent_params)

        # DQN parameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.1  # Initial exploration rate
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.epsilon_min = 0.01  # Minimum epsilon value
        self.batch_size = 64  # Batch size for training
        self.target_update = 10  # How often to update target network (steps)
        self.learning_rate = 0.001  # Learning rate
        self.memory_size = 10000  # Replay memory size

        # Curiosity parameters
        self.curiosity_weight = 0.25  # Weight for intrinsic reward
        self.curiosity_lr = 0.0001  # Learning rate for curiosity model
        self.curiosity_decay = 0.9999

        # Output dimension is the number of possible actions
        self.output_dim = len(Action)
        
        # Initialize input_dim to None, will be set in the first step
        self.input_dim = None

        # Initialise replay memory
        self.memory = ReplayMemory(self.memory_size)

        # Networks will be initialised after we know the input dimensions
        self.policy_net = None
        self.target_net = None
        self.q_optimiser = None
        self.forward_model = None
        self.curiosity_optimiser = None

        # Training variables
        self.steps_done = 0
        self.episode_rewards = []
        self.current_state = None
        self.current_action = None
        self.last_state = None
        self.last_action = None
        self.reward = 0
        self.intrinsic_rewards = []

    def _initialise_networks(self, input_dim):
        """Initialise networks once we know the input dimension"""
        self.input_dim = input_dim
        
        # Initialise Q-networks
        self.policy_net = DQN(self.input_dim, self.output_dim)
        self.target_net = DQN(self.input_dim, self.output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode

        # Initialise curiosity model
        self.forward_model = ForwardModel(self.input_dim, self.output_dim)
        self.inverse_model = InverseModel(self.input_dim, self.output_dim)
        self.inverse_optimiser = optim.Adam(self.inverse_model.parameters(), lr=self.curiosity_lr)


        # Initialise optimisers
        self.q_optimiser = optim.Adam(
            self.policy_net.parameters(), lr=self.learning_rate
        )
        self.curiosity_optimiser = optim.Adam(
            self.forward_model.parameters(), lr=self.curiosity_lr
        )

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
            # Greedy action
            with torch.no_grad():
                q_values = self.policy_net(state)
                action_idx = q_values.max(1)[1].item()
                return list(Action)[action_idx]

    def compute_intrinsic_reward(self, state, action, next_state):
        """Compute intrinsic reward based on prediction error"""
        if self.forward_model is None:
            return 0.0  # Return zero intrinsic reward if model isn't initialised yet
        
        # Convert action to tensor index
        action_idx = torch.tensor([list(Action).index(action)], dtype=torch.long)

        # Predict next state
        with torch.no_grad():
            next_state_pred = self.forward_model(state, action_idx)

        # Compute prediction error (curiosity)
        prediction_error = F.mse_loss(next_state_pred, next_state)

        return prediction_error.item()

    def update_curiosity_model(self, state_batch, action_batch, next_state_batch):
        """Update the forward model to better predict state transitions"""
        # Convert actions to tensor indices
        action_indices = torch.tensor(
            [list(Action).index(a) for a in action_batch], dtype=torch.long
        )

        # Predict next states
        next_state_preds = self.forward_model(state_batch, action_indices)

        # Compute prediction loss
        forward_loss = F.mse_loss(next_state_preds, next_state_batch)

        # Update forward model
        self.curiosity_optimiser.zero_grad()
        forward_loss.backward()
        self.curiosity_optimiser.step()

        return forward_loss.item()

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
        intrinsic_reward_batch = torch.tensor(
            batch.intrinsic_reward, dtype=torch.float32
        )

        # Combine extrinsic and intrinsic rewards
        combined_reward_batch = (
            reward_batch + self.curiosity_weight * intrinsic_reward_batch
        )

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size, dtype=torch.float32)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states
            ).max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = combined_reward_batch + (
            self.gamma * next_state_values
        )

        # Compute Huber loss
        q_loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Optimise the model
        self.q_optimiser.zero_grad()
        q_loss.backward()
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q_optimiser.step()

        # Update curiosity model
        curiosity_loss = self.update_curiosity_model(
            state_batch, batch.action, torch.cat(batch.next_state)
        )

        inverse_loss = self.update_inverse_model(
        state_batch, torch.cat(batch.next_state), batch.action
        )
        
        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return q_loss.item(), curiosity_loss
    
    def update_inverse_model(self, state_batch, next_state_batch, action_batch):
        action_indices = torch.tensor(
            [list(Action).index(a) for a in action_batch], dtype=torch.long
        )
        logits = self.inverse_model(state_batch, next_state_batch)
        loss = F.cross_entropy(logits, action_indices)

        self.inverse_optimiser.zero_grad()
        loss.backward()
        self.inverse_optimiser.step()

        return loss.item()


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

            # Calculate intrinsic reward
            intrinsic_reward = self.compute_intrinsic_reward(
                last_state_tensor, self.last_action, current_state_tensor
            )

            self.intrinsic_rewards.append(intrinsic_reward)

            # Store experience in replay memory
            self.memory.push(
                Experience(
                    last_state_tensor,
                    self.last_action,
                    reward,
                    current_state_tensor,
                    intrinsic_reward,
                    self.energy <= 0,  # Done if agent has no energy
                )
            )

            # Train the model
            if len(self.memory) >= self.batch_size:
                loss_info = self.optimise_model()
                if loss_info and self.steps_done % 10 == 0:
                    q_loss, curiosity_loss = loss_info
                    print(
                        f"Step {self.steps_done}: Q-Loss: {q_loss:.4f}, Curiosity Loss: {curiosity_loss:.4f}"
                    )

            # Update target network
            if self.steps_done % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        print(f"Received reward: {reward}, action {self.last_action}")
