import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from lbforaging.foraging.environment import Action
import random

# Q-learning parameters
ALPHA = 0.05  # Learning rate (lowered)
GAMMA = 0.99  # Discount factor (kept high)
EPSILON = 0.1  # Exploration rate
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--env", type=str, default="Foraging-8x8-2p-4f-v3", help="Environment to use")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--display_info", action="store_true", help="Display agent info per step")
    return parser.parse_args()

class QLearningAgent:
    """ Q-learning agent for foraging """

    def __init__(self, action_space_size):
        self.q_table = {}  # State-action value table
        self.action_space = list(Action)
        self.action_size = action_space_size

    def get_q_values(self, state):
        """ Retrieve Q-values for a given state, initializing with small random values """
        if state not in self.q_table:
            self.q_table[state] = np.random.uniform(low=-0.1, high=0.1, size=self.action_size)
        return self.q_table[state]

    def select_action(self, state):
        """ Selects an action using epsilon-greedy with epsilon decay """
        global EPSILON
        if random.uniform(0, 1) < EPSILON:
            return random.choice(range(self.action_size))  # Explore
        else:
            return np.argmax(self.get_q_values(state))  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        """ Update Q-values using the Bellman equation """
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)
        best_next_q = np.max(next_q_values)
        q_values[action] += ALPHA * (reward + GAMMA * best_next_q - q_values[action])

    def update_epsilon(self, episode):
        """ Gradually decay epsilon over episodes """
        global EPSILON
        new_epsilon = EPSILON * (EPSILON_DECAY ** episode)
        EPSILON = max(new_epsilon, MIN_EPSILON)

class VisualisedEnv:
    """ Run a visualised environment with Q-learning agents. """

    def __init__(self, env: str, max_steps: int, display_info: bool = True):
        self.env = gym.make(env, render_mode=None, max_episode_steps=max_steps)  # Turn off rendering
        self.n_agents = self.env.unwrapped.n_agents
        self.display_info = display_info
        self.agents = [QLearningAgent(len(Action)) for _ in range(self.n_agents)]
        self.episode_rewards = []
        self._run_episodes(num_episodes=1000000)

    def _run_episodes(self, num_episodes=100):
        """ Runs Q-learning for a specified number of episodes """
        total_rewards = np.zeros(self.n_agents)

        for episode in range(num_episodes):
            ep_returns = np.zeros(self.n_agents)
            obss, _ = self.env.reset()
            state = tuple(map(tuple, obss))  # Convert each observation to a tuple for Q-table

            if self.display_info:
                print(f"Episode {episode + 1} begins.")

            for step in range(100):
                actions = [agent.select_action(state) for agent in self.agents]
                next_obss, rews, done, trunc, _ = self.env.step([act for act in actions])
                next_state = tuple(map(tuple, next_obss))

                for i, agent in enumerate(self.agents):
                    agent.update_q_table(state, actions[i], rews[i], next_state)
                
                ep_returns += np.array(rews)
                state = next_state  # Move to the next state

                if done or trunc:
                    break

            total_rewards += ep_returns
            self.episode_rewards.append(sum(ep_returns))
            for agent in self.agents:
                agent.update_epsilon(episode)  # Update epsilon for each agent

            if self.display_info:
                print(f"Episode {episode + 1} reward: {ep_returns}\n")

        print(f"Total rewards after {num_episodes} episodes: {total_rewards}")
        self._plot_results()

    def _plot_results(self):
        """ Plot average reward over time """
        window_size = 100
        moving_avg_rewards = np.convolve(self.episode_rewards, np.ones(window_size)/window_size, mode='valid')

        # Plot the moving average of rewards
        plt.plot(moving_avg_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.title("Q-Learning Performance - Moving Average of Rewards")
        plt.show()

if __name__ == "__main__":
    args = parse_args()
    VisualisedEnv(env=args.env, display_info=True, max_steps=args.max_steps)
