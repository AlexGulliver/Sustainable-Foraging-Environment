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
import os
import datetime


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
        starting_energy: int,
        display_info: bool = True,
    ):
        # Setup logging directory and file
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs_{agent_type}_{self.timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = open(f"{self.log_dir}/simulation_log.txt", "w")
        
        # Initialise environment and parameters
        self.env = gym.make(env, render_mode="human", max_episode_steps=max_steps)
        self.n_agents = self.env.unwrapped.n_agents
        self.display_info = display_info
        self.agent_type = agent_type
        self.max_timesteps = max_steps
        self.starting_energy = starting_energy

        # Log initial setup
        self.log(f"Simulation started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Agent type: {agent_type}")
        self.log(f"Number of agents: {self.n_agents}")
        self.log(f"Max steps per episode: {max_steps}")
        self.log(f"Starting energy: {starting_energy}")
        self.log(f"Number of episodes: {num_episodes}")
        self.log("="*50)

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
        self.initial_energy = self.starting_energy # Store initial energy value for respawning
        
        self.max_episodes_to_track = 3  # Number of top episodes to track
        self.episodes_completed = 0
        self.episodes_reached_max_steps = 0
        self.episode_survival_rates = []
        self.top_episodes = []  # Will store (episode_num, total_reward) tuples
        self.agent_deaths_per_episode = []
        self.food_collected_per_episode = []
        self.max_food_collected = 0
        self.peak_reward = -float('inf')
        self.peak_reward_episode = 0
        self.agent_lifetime_stats = [{"deaths": 0, "food_collected": 0} for _ in range(self.n_agents)]

        self._run_episodes(num_episodes=num_episodes)

    def log(self, message):
        """Write message to log file and print if display_info is True"""
        self.log_file.write(f"{message}\n")
        self.log_file.flush()
        if self.display_info:
            print(message)
    
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
            self.log(f"Unknown agent type: {self.agent_type}, defaulting to Curious DQN")
            return [
                CuriosityDrivenDQNAgent(agent_params=self.agent_params)
                for _ in range(self.n_agents)
            ]

    def _run_episodes(self, num_episodes=1):
        total_rewards = np.zeros(self.n_agents)
        total_intrinsic_rewards = np.zeros(self.n_agents)
        total_steps = 0
        
        start_time = time.time()

        for episode in range(num_episodes):
            ep_returns = np.zeros(self.n_agents)
            ep_intrinsic_returns = np.zeros(self.n_agents)
            ep_length = 0
            ep_food_collected = 0
            ep_deaths = 0
            agent_initial_food = [0] * self.n_agents  # Track food at start of episode

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
            
            # Log episode start
            self.log(f"\nEpisode {episode + 1} started")
            self.log(f"Initial positions: {[agent.position for agent in self.agents]}")
            self.log(f"Initial energy: {[agent.energy for agent in self.agents]}")
            
            # Track which agents die during this episode
            agents_died_this_episode = [False] * self.n_agents

            for step in range(self.max_timesteps):
                # Agents with 0 energy should not act
                actions = []
                for i, (agent, obs) in enumerate(zip(self.agents, obss)):
                    if agent.energy > 0:
                        action = agent.step(obs)
                        actions.append(action)
                    else:
                        actions.append(None)
                        # If agent just died this step
                        if not agents_died_this_episode[i] and agent.energy <= 0:
                            agents_died_this_episode[i] = True
                            ep_deaths += 1
                            self.agent_lifetime_stats[i]["deaths"] += 1
                            self.log(f"  Step {step + 1}: Agent {i+1} died (energy depleted)")
                
                # Filter out None actions
                valid_actions = [act for act in actions if act is not None]

                # If no agents can act, end the episode early
                if not valid_actions:
                    self.log(f"Episode {episode + 1} ended early at step {step + 1}: All agents are dead.")
                    break

                obss, rews, done, trunc, info = self.env.step(
                    [act.value for act in valid_actions]
                )

                # Collect rewards for agents, including intrinsic rewards for curious DQN agents
                for i, (agent, rew) in enumerate(zip(self.agents, rews)):
                    intrinsic_rew = 0
                    
                    # Safe way to check if this agent tracks intrinsic rewards
                    if self.agent_type == "curious_dqn":
                        # Only try to access intrinsic_rewards if agent is a curious DQN
                        try:
                            # Use direct dict access to avoid recursion issues
                            if hasattr(agent, "__dict__") and "intrinsic_rewards" in agent.__dict__:
                                rewards_list = agent.__dict__["intrinsic_rewards"]
                                if rewards_list and len(rewards_list) > 0:
                                    intrinsic_rew = rewards_list[-1]
                        except Exception as e:
                            self.log(f"Error accessing intrinsic rewards: {e}")
                            intrinsic_rew = 0

                    # Zero out rewards for dead agents
                    if agent.energy == 0:
                        rews[i] = 0  # Dead agents receive no reward
                        intrinsic_rew = 0

                    ep_returns[i] += rews[i]
                    ep_intrinsic_returns[i] += intrinsic_rew

                # Detailed step logging
                if self.display_info:
                    self.log(f"  Step {step + 1}: Rewards {rews}, Energy {[agent.energy for agent in self.agents]}")

                ep_length += 1
                self.env.render()

                # If the environment signals an end, stop
                if done or trunc:
                    self.log(f"Episode {episode + 1} terminated at step {step + 1} with done={done}, truncated={trunc}")
                    break

            # Track if episode reached max steps
            reached_max_steps = (ep_length == self.max_timesteps)
            if reached_max_steps:
                self.episodes_reached_max_steps += 1
                self.log(f"Episode {episode + 1} reached maximum {self.max_timesteps} steps")
            
            # Calculate survival rate for this episode
            survival_rate = (self.n_agents - ep_deaths) / self.n_agents * 100
            self.episode_survival_rates.append(survival_rate)
            
            # Store episode stats
            self.episode_lengths.append(ep_length)
            self.agent_deaths_per_episode.append(ep_deaths)
            
            total_rewards += ep_returns  # Accumulate rewards
            total_intrinsic_rewards += ep_intrinsic_returns
            total_steps += ep_length
            
            # Calculate total episode reward
            total_ep_reward = np.sum(ep_returns)
            
            # Track peak reward
            if total_ep_reward > self.peak_reward:
                self.peak_reward = total_ep_reward
                self.peak_reward_episode = episode + 1
            
            # Keep track of top episodes
            self.top_episodes.append((episode + 1, total_ep_reward))
            self.top_episodes.sort(key=lambda x: x[1], reverse=True)
            self.top_episodes = self.top_episodes[:self.max_episodes_to_track]

            # Store episode rewards for each agent
            for i in range(self.n_agents):
                self.episode_rewards[i].append(ep_returns[i])
                
                # Store intrinsic rewards only if using curious_dqn agents
                if self.agent_type == "curious_dqn":
                    try:
                        if hasattr(self.agents[i], "__dict__") and "intrinsic_rewards" in self.agents[i].__dict__:
                            self.episode_intrinsic_rewards[i].append(ep_intrinsic_returns[i])
                        else:
                            self.episode_intrinsic_rewards[i].append(0)
                    except Exception as e:
                        self.log(f"Error storing intrinsic rewards: {e}")
                        self.episode_intrinsic_rewards[i].append(0)
                else:
                    # For non-curious agents, always store 0
                    self.episode_intrinsic_rewards[i].append(0)

            self.avg_episode_rewards.append(np.mean(ep_returns))
            self.avg_episode_intrinsic_rewards.append(np.mean(ep_intrinsic_returns))

            self.cumulative_rewards += ep_returns
            self.cumulative_intrinsic_rewards += ep_intrinsic_returns
            
            # Log episode summary
            self.log(f"Episode {episode + 1} summary:")
            self.log(f"  Episode length: {ep_length} steps")
            self.log(f"  Rewards: {ep_returns}")
            self.log(f"  Total episode reward: {total_ep_reward:.2f}")
            self.log(f"  Intrinsic rewards: {ep_intrinsic_returns}")
            self.log(f"  Agent deaths: {ep_deaths}")
            self.log(f"  Survival rate: {survival_rate:.1f}%")
            self.log(f"  Reached max steps: {reached_max_steps}")
            self.log("-" * 40)
            
            self.episodes_completed += 1

        elapsed_time = time.time() - start_time
        
        self._generate_summary_statistics(num_episodes, total_rewards, total_steps, elapsed_time)

        self._plot_rewards()
        if self.agent_type == "curious_dqn":
            self._plot_intrinsic_rewards()
            
        self._plot_episode_lengths(self.episode_lengths)
        self._plot_additional_metrics()

        self.env.close()
        self.log_file.close()

    def _generate_summary_statistics(self, num_episodes, total_rewards, total_steps, elapsed_time):
        """Generate and log summary statistics for the simulation"""
        # Calculate overall statistics
        avg_reward_per_agent = total_rewards / num_episodes
        avg_reward_per_episode = np.mean(self.avg_episode_rewards)
        avg_episode_length = np.mean(self.episode_lengths)
        median_episode_length = np.median(self.episode_lengths)
        max_episode_length = max(self.episode_lengths)
        min_episode_length = min(self.episode_lengths)
        avg_survival_rate = np.mean(self.episode_survival_rates)
        avg_deaths_per_episode = np.mean(self.agent_deaths_per_episode)
        
        # Calculate standard deviations
        std_reward = np.std(self.avg_episode_rewards)
        std_episode_length = np.std(self.episode_lengths)
        std_survival_rate = np.std(self.episode_survival_rates)
        
        # Write summary to log file
        self.log("\n" + "=" * 50)
        self.log("SIMULATION SUMMARY")
        self.log("=" * 50)
        self.log(f"Completed {self.episodes_completed} episodes with {self.n_agents} agents")
        self.log(f"Agent type: {self.agent_type}")
        self.log(f"Total simulation time: {elapsed_time:.2f} seconds")
        self.log(f"Average time per episode: {elapsed_time/num_episodes:.2f} seconds")
        self.log("\nREWARD STATISTICS:")
        self.log(f"  Peak total episode reward: {self.peak_reward:.2f} (Episode {self.peak_reward_episode})")
        self.log(f"  Average reward per episode: {avg_reward_per_episode:.2f} (±{std_reward:.2f})")
        self.log(f"  Average reward per agent: {avg_reward_per_agent}")
        self.log(f"  Top {len(self.top_episodes)} episodes by reward:")
        for rank, (ep_num, reward) in enumerate(self.top_episodes, 1):
            self.log(f"    #{rank}: Episode {ep_num} with reward {reward:.2f}")
            
        self.log("\nEPISODE STATISTICS:")
        self.log(f"  Episodes reaching maximum steps ({self.max_timesteps}): {self.episodes_reached_max_steps} ({self.episodes_reached_max_steps/num_episodes*100:.1f}%)")
        self.log(f"  Average episode length: {avg_episode_length:.2f} steps (±{std_episode_length:.2f})")
        self.log(f"  Median episode length: {median_episode_length:.1f} steps")
        self.log(f"  Min/Max episode length: {min_episode_length}/{max_episode_length} steps")
        
        self.log("\nSURVIVAL STATISTICS:")
        self.log(f"  Average survival rate: {avg_survival_rate:.1f}% (±{std_survival_rate:.1f}%)")
        self.log(f"  Average deaths per episode: {avg_deaths_per_episode:.2f}")
        for i in range(self.n_agents):
            death_rate = self.agent_lifetime_stats[i]["deaths"] / num_episodes * 100
            self.log(f"  Agent {i+1}: {self.agent_lifetime_stats[i]['deaths']} deaths ({death_rate:.1f}% of episodes)")
        
        self.log("\nSUMMARY FILE LOCATIONS:")
        self.log(f"  Log file: {self.log_dir}/simulation_log.txt")
        self.log(f"  Agent rewards plot: {self.log_dir}/agent_rewards_per_episode.png")
        self.log(f"  Average episodic reward plot: {self.log_dir}/average_episodic_reward.png")
        self.log(f"  Combined rewards plot: {self.log_dir}/rewards_plot.png")
        self.log(f"  Episode length plot: {self.log_dir}/episode_length_over_time.png")
        if self.agent_type == "curious_dqn":
            self.log(f"  Intrinsic rewards plot: {self.log_dir}/intrinsic_rewards_plot.png")
        self.log(f"  Additional metrics plot: {self.log_dir}/additional_metrics.png")
        self.log("=" * 50)

    def _plot_rewards(self):
        """Plots the rewards over episodes and average episodic reward trend as separate files."""
        # Plot individual agent rewards per episode
        plt.figure(figsize=(10, 6))
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
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/agent_rewards_per_episode.png")
        plt.close()

        plt.figure(figsize=(10, 6))
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
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/average_episodic_reward.png")
        plt.close()
        
        plt.figure(figsize=(12, 9))
        
        # Rewards subplot
        plt.subplot(2, 1, 1)
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
        plt.savefig(f"{self.log_dir}/rewards_plot.png")
        plt.close()

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
        plt.savefig(f"{self.log_dir}/intrinsic_rewards_plot.png")

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
        plt.savefig(f"{self.log_dir}/episode_length_over_time.png")
        
    def _plot_additional_metrics(self):
        """Plot additional performance metrics."""
        plt.figure(figsize=(12, 10))
        
        # Survival rates subplot
        plt.subplot(2, 2, 1)
        plt.plot(range(1, len(self.episode_survival_rates) + 1), 
                self.episode_survival_rates, 'g-', label='Agent Survival Rate')
        plt.xlabel('Episode')
        plt.ylabel('Survival Rate (%)')
        plt.title('Agent Survival Rate per Episode')
        plt.grid(True)
        
        # Deaths per episode subplot
        plt.subplot(2, 2, 3)
        plt.plot(range(1, len(self.agent_deaths_per_episode) + 1), 
                self.agent_deaths_per_episode, 'r-', label='Agent Deaths')
        plt.xlabel('Episode')
        plt.ylabel('Number of Deaths')
        plt.title('Agent Deaths per Episode')
        plt.grid(True)
        
        # Combined metrics subplot - normalise and plot together
        plt.subplot(2, 2, 4)
        
        # Normalise metrics
        if self.episode_survival_rates:
            norm_survival = [x / 100 for x in self.episode_survival_rates]
            plt.plot(range(1, len(norm_survival) + 1), norm_survival, 'g-', label='Normalised Survival')
        
        if self.avg_episode_rewards:
            max_reward = max(max(self.avg_episode_rewards), abs(min(self.avg_episode_rewards))) if self.avg_episode_rewards else 1
            norm_rewards = [x / max_reward for x in self.avg_episode_rewards]
            plt.plot(range(1, len(norm_rewards) + 1), norm_rewards, 'orange', label='Normalised Reward')
        
        plt.xlabel('Episode')
        plt.ylabel('Normalised Value')
        plt.title('Combined Performance Metrics')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/additional_metrics.png")


if __name__ == "__main__":
    args = parse_args()
    VisualisedEnv(
        env=args.env,
        display_info=True,
        max_steps=30,
        num_episodes=1000,
        agent_type="dqn",  # "random", "qlearning", "dqn", "curious_dqn"
        starting_energy=10,  # Initial energy for agents
    )