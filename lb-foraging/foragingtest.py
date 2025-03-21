from argparse import ArgumentParser
import warnings
import numpy as np
import gymnasium as gym
from lbforaging.foraging.environment import Action
from lbforaging.agents.qlearningagent import QLearningForagingAgent
from lbforaging.agents.randomagent import RandomForagingAgent
from lbforaging.agents.deepqagent import DeepQLearningForagingAgent
from lbforaging.agents.deepqcuriousagent import CuriosityDrivenDQNAgent
import pyglet
import time
import matplotlib.pyplot as plt

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="SustainableForagingEnv-v0",
        help="Environment to use",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Maximum number of steps per episode",
    )
    parser.add_argument(
        "--display_info",
        action="store_true",
        help="Display agent info per step",
    )
    return parser.parse_args()


class VisualisedEnv:
    def __init__(self, env: str, max_steps: int, display_info: bool = True):
        self.env = gym.make(env, render_mode="human", max_episode_steps=max_steps)
        self.n_agents = self.env.unwrapped.n_agents
        self.display_info = display_info

        self.episode_lengths = []

        # # Initialize agents
        # self.agents = [
        #     RandomForagingAgent(agent_params={
        #         "eta": 0.5, "carry_capacity": 10, "survival_cost": 0.5, "tau": 0, "k": 1
        #     }) for _ in range(self.n_agents)
        # ]

        obss, _ = self.env.reset()
        self.env.render()

        if hasattr(self.env.unwrapped, "viewer") and self.env.unwrapped.viewer:
            self.env.unwrapped.viewer.window.on_key_press = self._key_press
        else:
            print("Warning: Viewer not initialized. Keyboard input may not work.")

        self.episode_rewards = [[] for _ in range(self.n_agents)]
        self.avg_episode_rewards = []
        self.episode_lengths = []
        self.cumulative_rewards = np.zeros(self.n_agents) 
        self.max_timesteps = max_steps  # Set max_timesteps to max_steps (given as argument)

        self._run_episodes(num_episodes=100)


    def _key_press(self, k, mod):
        from pyglet.window import key
        if k == key.ESCAPE:
            self.running = False
            self.env.close()


    def _run_episodes(self, num_episodes=1):
        total_rewards = np.zeros(self.n_agents)

        for episode in range(num_episodes):
            ep_returns = np.zeros(self.n_agents)  # Reset episode rewards
            ep_length = 0
            # Reset the environment
            obss, _ = self.env.reset()

            # Respawn all agents with full energy
            self.agents = [
                CuriosityDrivenDQNAgent(agent_params={
                    "eta": 0.5, "carry_capacity": 10, "survival_cost": 1, "tau": 0, "k": 1
                }) for _ in range(self.n_agents)
            ]

            for player, agent in zip(self.env.unwrapped.players, self.agents):
                print(f"Assigning {agent} to {player}")
                player.set_controller(agent)

            self.env.render()

            if self.display_info:
                print(f"Episode {episode + 1} begins.")


            for step in range(100): 
                # Remove dead agents before taking actions
                self.agents = [agent for agent in self.agents if agent.energy > 0]
                if not self.agents:  # If all agents are dead, end the episode early
                    print("All agents have died. Ending episode early.")
                    break

                # Get actions from remaining agents
                actions = [agent.step(obs) for agent, obs in zip(self.agents, obss)]
                actions = [act for act in actions if act is not None]  # Remove None actions
                print(actions)

                if not actions:  # If no valid actions, stop the episode
                    print("No valid actions remaining. Ending episode.")
                    break

                obss, rews, done, trunc, _ = self.env.step([act.value for act in actions])

                ep_returns += np.array(rews)

                if self.display_info:
                    print(f"Step {step + 1}: Rewards {rews}")

                ep_length += 1

                self.env.render()
                # time.sleep(0.5)

                if done or trunc:
                    break  # Stop episode if terminated early

            self.episode_lengths.append(ep_length)
                    
            total_rewards += ep_returns  # Accumulate rewards
            
            # Store episode rewards for each agent
            for i in range(self.n_agents):
                self.episode_rewards[i].append(ep_returns[i])
            
            # Calculate and store average reward for this episode
            self.avg_episode_rewards.append(np.mean(ep_returns))
            
            # Update cumulative rewards
            self.cumulative_rewards += ep_returns

            if self.display_info:
                print(f"Episode {episode + 1} reward: {ep_returns}\n")

        print(f"Total rewards after {num_episodes} episodes: {total_rewards}")
        
        # Plot rewards over time
        self._plot_rewards()

        self._plot_episode_lengths(self.episode_lengths)

        time.sleep(15)
        self.env.close()

    def _plot_rewards(self):
        """Plots the rewards over episodes and average rewards trend."""
        plt.figure(figsize=(12, 8))
        
        # Create subplot layout: 2x1 grid
        plt.subplot(2, 1, 1)
        
        # Plot individual agent rewards per episode
        for i, rewards in enumerate(self.episode_rewards):
            plt.plot(range(1, len(rewards) + 1), rewards, marker='o', label=f'Agent {i + 1}')
            
        plt.plot(range(1, len(self.avg_episode_rewards) + 1), self.avg_episode_rewards, 
                 color='black', linestyle='--', linewidth=2, marker='s', label='Average Reward')
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Agent Rewards per Episode')
        plt.legend()
        plt.grid(True)
        
        # Create subplot for cumulative rewards
        plt.subplot(2, 1, 2)
        
        # Calculate cumulative average reward
        cum_avg_reward = np.cumsum(self.avg_episode_rewards) / np.arange(1, len(self.avg_episode_rewards) + 1)
        
        plt.plot(range(1, len(self.avg_episode_rewards) + 1), cum_avg_reward, 
                 color='green', linestyle='-', linewidth=2, marker='d', label='Cumulative Average Reward')
        
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Average Reward')
        plt.title('Cumulative Average Reward Over Episodes')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('rewards_over_time.png')  # Save the figure
        plt.show()

    def _plot_episode_lengths(self, episode_lengths):
        """Plots the episode lengths over training time."""
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(episode_lengths) + 1), episode_lengths, marker='o', linestyle='-', label="Episode Length")
        
        # Optional: Rolling average for smoothing
        window = 10
        if len(episode_lengths) > window:
            rolling_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
            plt.plot(range(window, len(episode_lengths) + 1), rolling_avg, linestyle='--', color='red', label="Rolling Avg (10 episodes)")

        plt.xlabel('Episode')
        plt.ylabel('Steps Until Termination')
        plt.title('Episode Length Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig('episode_length_over_time.png')  # Save the figure
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    VisualisedEnv(env=args.env, display_info=True, max_steps=args.max_steps)