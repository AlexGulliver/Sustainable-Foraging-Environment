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
    parser.add_argument(
        "--agent_type",
        type=str,
        default="curious_dqn",
        choices=["random", "qlearning", "dqn", "curious_dqn"],
        help="Type of agent to use",
    )
    return parser.parse_args()


import random
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import time
from typing import List


class VisualisedEnv:
    def __init__(
        self,
        env: str,
        max_steps: int,
        num_episodes: int,
        agent_type: str,
        display_info: bool = True,
    ):
        self.env = gym.make(env, render_mode="human", max_episode_steps=max_steps)
        self.n_agents = self.env.unwrapped.n_agents
        self.display_info = display_info
        self.agent_type = agent_type
        self.max_timesteps = max_steps

        # Parameters
        self.agent_params = {
            "eta": 0.5,
            "carry_capacity": 5,
            "survival_cost": 1,
            "tau": 0,
            "k": 1,
        }

        self.agents = self._create_agents()

        obss, _ = self.env.reset()
        self.env.render()

        # Data collection
        self.episode_rewards = [[] for _ in range(self.n_agents)]
        self.episode_intrinsic_rewards = [[] for _ in range(self.n_agents)]
        self.avg_episode_rewards = []
        self.avg_episode_intrinsic_rewards = []
        self.episode_lengths = []
        self.cumulative_rewards = np.zeros(self.n_agents)
        self.cumulative_intrinsic_rewards = np.zeros(self.n_agents)
        self.initial_energy = 10  # Store initial energy value for respawning

        self._run_episodes(num_episodes=num_episodes)

    def _create_agents(self):
        """Create agents based on the selected agent type."""
        if self.agent_type == "random":
            return [
                RandomForagingAgent(agent_params=self.agent_params)
                for _ in range(self.n_agents)
            ]
        elif self.agent_type == "qlearning":
            return [
                QLearningForagingAgent(agent_params=self.agent_params)
                for _ in range(self.n_agents)
            ]
        elif self.agent_type == "dqn":
            return [
                DeepQLearningForagingAgent(agent_params=self.agent_params)
                for _ in range(self.n_agents)
            ]
        elif self.agent_type == "curious_dqn":
            return [
                CuriosityDrivenDQNAgent(agent_params=self.agent_params)
                for _ in range(self.n_agents)
            ]
        else:
            print(f"Unknown agent type: {self.agent_type}, defaulting to Curious DQN")
            return [
                CuriosityDrivenDQNAgent(agent_params=self.agent_params)
                for _ in range(self.n_agents)
            ]

    def _run_episodes(self, num_episodes=1):
        total_rewards = np.zeros(self.n_agents)
        total_intrinsic_rewards = np.zeros(self.n_agents)

        for episode in range(num_episodes):
            ep_returns = np.zeros(self.n_agents)
            ep_intrinsic_returns = np.zeros(self.n_agents)
            ep_length = 0

            # Reset environment
            obss, _ = self.env.reset()

            # Respawn all agents for new episode
            for i, (player, agent) in enumerate(
                zip(self.env.unwrapped.players, self.agents)
            ):
                player.set_controller(agent)
                agent.position = player.position
                agent.energy = self.initial_energy  # Explicitly reset to initial energy

            self.env.render()

            for step in range(self.max_timesteps):
                # Agents with 0 energy should not act
                actions = [
                    agent.step(obs) if agent.energy > 0 else None
                    for agent, obs in zip(self.agents, obss)
                ]
                actions = [act for act in actions if act is not None]

                # If no agents can act, end the episode early
                if not actions:
                    print(f"Episode {episode + 1} ended early: All agents are dead.")
                    break

                obss, rews, done, trunc, _ = self.env.step(
                    [act.value for act in actions]
                )

                # Collect rewards for agents, including intrinsic rewards for curious DQN agents
                for i, (agent, rew) in enumerate(zip(self.agents, rews)):
                    intrinsic_rew = 0
                    try:
                        # Only access intrinsic_rewards if it exists directly on the agent object
                        if hasattr(type(agent), 'intrinsic_rewards') or '__intrinsic_rewards' in agent.__dict__:
                            intrinsic_rew = agent.intrinsic_rewards[-1] if agent.intrinsic_rewards else 0
                    except (AttributeError, RecursionError):
                        # If any error occurs, default to 0 for intrinsic reward
                        intrinsic_rew = 0

                    # Zero out rewards for dead agents
                    if agent.energy == 0:
                        rews[i] = 0  # Dead agents receive no reward
                        intrinsic_rew = 0

                    ep_returns[i] += rews[i]
                    ep_intrinsic_returns[i] += intrinsic_rew

                if self.display_info:
                    print(f"Step {step + 1}: Rewards {rews}")

                ep_length += 1
                self.env.render()

                # If the environment signals an end, stop
                if done or trunc:
                    break

            self.episode_lengths.append(ep_length)
            total_rewards += ep_returns  # Accumulate rewards
            total_intrinsic_rewards += ep_intrinsic_returns

            # Store episode rewards for each agent
            for i in range(self.n_agents):
                self.episode_rewards[i].append(ep_returns[i])

                try:
                    if hasattr(type(self.agents[i]), 'intrinsic_rewards') or '__intrinsic_rewards' in self.agents[i].__dict__:
                        self.episode_intrinsic_rewards[i].append(ep_intrinsic_returns[i])
                    else:
                        self.episode_intrinsic_rewards[i].append(0)
                except (AttributeError, RecursionError):
                    self.episode_intrinsic_rewards[i].append(0)

            self.avg_episode_rewards.append(np.mean(ep_returns))
            self.avg_episode_intrinsic_rewards.append(np.mean(ep_intrinsic_returns))

            self.cumulative_rewards += ep_returns
            self.cumulative_intrinsic_rewards += ep_intrinsic_returns

            if self.display_info:
                print(f"Episode {episode + 1} reward: {ep_returns}")
                print(
                    f"Episode {episode + 1} intrinsic reward: {ep_intrinsic_returns}\n"
                )

        # Plot rewards and intrinsic rewards
        self._plot_rewards()
        self._plot_intrinsic_rewards()
        self._plot_episode_lengths(self.episode_lengths)

        self.env.close()

    def _plot_rewards(self):
        """Plots the rewards over episodes and average episodic reward trend."""
        plt.figure(figsize=(12, 9))

        # Rewards subplot
        plt.subplot(2, 1, 1)

        # Plot individual agent rewards per episode
        for i, rewards in enumerate(self.episode_rewards):
            plt.plot(
                range(1, len(rewards) + 1), rewards, marker="o", label=f"Agent {i + 1}"
            )

        plt.plot(
            range(1, len(self.avg_episode_rewards) + 1),
            self.avg_episode_rewards,
            color="black",
            linestyle="--",
            linewidth=2,
            marker="s",
            label="Average Reward Per Episode",
        )

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Agent Rewards per Episode")
        plt.legend()
        plt.grid(True)

        # Cumulative average subplot
        plt.subplot(2, 1, 2)

        # Average episodic rewards
        avg_episodic_reward = np.cumsum(self.avg_episode_rewards) / np.arange(
            1, len(self.avg_episode_rewards) + 1
        )

        plt.plot(
            range(1, len(self.avg_episode_rewards) + 1),
            avg_episodic_reward,
            color="green",
            linestyle="-",
            linewidth=2,
            marker="d",
            label="Average Episodic Reward",
        )

        plt.xlabel("Episode")
        plt.ylabel("Average Episodic Reward")
        plt.title("Average Episodic Reward")
        plt.legend()
        plt.grid(True)

        plt.tight_layout(h_pad=1)
        plt.savefig("rewards_plot.png")

    def _plot_intrinsic_rewards(self):
        """Plots the intrinsic rewards over episodes and average episodic intrinsic reward trend."""
        plt.figure(figsize=(12, 9))

        # Intrinsic rewards subplot
        plt.subplot(2, 1, 1)

        # Plot individual agent intrinsic rewards per episode
        for i, intrinsic_rewards in enumerate(self.episode_intrinsic_rewards):
            plt.plot(
                range(1, len(intrinsic_rewards) + 1),
                intrinsic_rewards,
                marker="o",
                label=f"Agent {i + 1}",
            )

        plt.plot(
            range(1, len(self.avg_episode_intrinsic_rewards) + 1),
            self.avg_episode_intrinsic_rewards,
            color="black",
            linestyle="--",
            linewidth=2,
            marker="s",
            label="Average Intrinsic Reward Per Episode",
        )

        plt.xlabel("Episode")
        plt.ylabel("Intrinsic Reward")
        plt.title("Agent Intrinsic Rewards per Episode")
        plt.legend()
        plt.grid(True)

        # Cumulative average subplot
        plt.subplot(2, 1, 2)

        # Average episodic intrinsic rewards
        avg_episodic_intrinsic_reward = np.cumsum(
            self.avg_episode_intrinsic_rewards
        ) / np.arange(1, len(self.avg_episode_intrinsic_rewards) + 1)

        plt.plot(
            range(1, len(self.avg_episode_intrinsic_rewards) + 1),
            avg_episodic_intrinsic_reward,
            color="blue",
            linestyle="-",
            linewidth=2,
            marker="d",
            label="Average Episodic Intrinsic Reward",
        )

        plt.xlabel("Episode")
        plt.ylabel("Average Episodic Intrinsic Reward")
        plt.title("Average Episodic Intrinsic Reward")
        plt.legend()
        plt.grid(True)

        plt.tight_layout(h_pad=1)
        plt.savefig("intrinsic_rewards_plot.png")

    def _plot_episode_lengths(self, episode_lengths):
        """Plots the episode lengths over training time."""
        plt.figure(figsize=(10, 5))
        plt.plot(
            range(1, len(episode_lengths) + 1),
            episode_lengths,
            marker="o",
            linestyle="-",
            label="Episode Length",
        )

        # Optional: Rolling average for smoothing
        window = 10
        if len(episode_lengths) > window:
            rolling_avg = np.convolve(
                episode_lengths, np.ones(window) / window, mode="valid"
            )
            plt.plot(
                range(window, len(episode_lengths) + 1),
                rolling_avg,
                linestyle="--",
                color="red",
                label="Rolling Avg (10 episodes)",
            )

        plt.xlabel("Episode")
        plt.ylabel("Steps Until Termination")
        plt.title("Episode Length Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig("episode_length_over_time.png")
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    VisualisedEnv(
        env=args.env,
        display_info=True,
        max_steps=100,
        num_episodes=2500,
        agent_type="curious_dqn",  # "random", "qlearning", "dqn", "curious_dqn"
    )
