"""For logging, data collection, and visualisation for foraging simulations"""

import numpy as np
import matplotlib.pyplot as plt
import os
import datetime


class DataCollector:
    """Handles logging, data collection, and visualisation for foraging simulations"""
    
    def __init__(
        self, 
        log_dir, 
        n_agents, 
        agent_type, 
        max_timesteps, 
        starting_energy, 
        num_episodes,
        display_info=True
    ):
        # Setup logging directory and file
        self.log_dir = log_dir
        self.log_file = open(f"{log_dir}/simulation_log.txt", "w")
        
        # Store simulation parameters
        self.n_agents = n_agents
        self.agent_type = agent_type
        self.max_timesteps = max_timesteps
        self.starting_energy = starting_energy
        self.display_info = display_info
        
        # Data collection
        self.episode_rewards = [[] for _ in range(self.n_agents)]
        self.episode_intrinsic_rewards = [[] for _ in range(self.n_agents)]
        self.avg_episode_rewards = []
        self.avg_episode_intrinsic_rewards = []
        self.episode_lengths = []
        self.cumulative_rewards = np.zeros(self.n_agents)
        self.cumulative_intrinsic_rewards = np.zeros(self.n_agents)
    
        self.max_episodes_to_track = 3  # Number of top episodes to track
        self.episodes_completed = 0
        self.episodes_reached_max_steps = 0
        self.episode_survival_rates = []
        self.top_episodes = []  # Stores (episode_num, total_reward) tuples
        self.agent_deaths_per_episode = []
        self.food_collected_per_episode = []
        self.max_food_collected = 0
        self.peak_reward = -float('inf')
        self.peak_reward_episode = 0
        self.agent_lifetime_stats = [{"deaths": 0, "food_collected": 0} for _ in range(self.n_agents)]
        
        # Log initial setup
        self.log(f"Simulation started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Agent type: {agent_type}")
        self.log(f"Number of agents: {n_agents}")
        self.log(f"Max steps per episode: {max_timesteps}")
        self.log(f"Starting energy: {starting_energy}")
        self.log(f"Number of episodes: {num_episodes}")
        self.log("="*50)
    
    def log(self, message):
        """Write message to log file and print if display_info is True"""
        self.log_file.write(f"{message}\n")
        self.log_file.flush()
        if self.display_info:
            print(message)
    
    def update_episode_data(self, episode_num, ep_returns, ep_intrinsic_returns, ep_length, ep_deaths, reached_max_steps):
        """Update data storage with results from a completed episode"""
        # Calculate survival rate for this episode
        survival_rate = (self.n_agents - ep_deaths) / self.n_agents * 100
        self.episode_survival_rates.append(survival_rate)
        
        # Store episode stats
        self.episode_lengths.append(ep_length)
        self.agent_deaths_per_episode.append(ep_deaths)
        
        # Calculate total episode reward
        total_ep_reward = np.sum(ep_returns)
        
        # Track peak reward
        if total_ep_reward > self.peak_reward:
            self.peak_reward = total_ep_reward
            self.peak_reward_episode = episode_num
        
        # Keep track of top episodes
        self.top_episodes.append((episode_num, total_ep_reward))
        self.top_episodes.sort(key=lambda x: x[1], reverse=True)
        self.top_episodes = self.top_episodes[:self.max_episodes_to_track]
        
        # Store episode rewards for each agent
        for i in range(self.n_agents):
            self.episode_rewards[i].append(ep_returns[i])
            
            # Store intrinsic rewards only if using curious_dqn agents
            if self.agent_type == "curious_dqn":
                try:
                    self.episode_intrinsic_rewards[i].append(ep_intrinsic_returns[i])
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
        self.log(f"Episode {episode_num} summary:")
        self.log(f"  Episode length: {ep_length} steps")
        self.log(f"  Rewards: {ep_returns}")
        self.log(f"  Total episode reward: {total_ep_reward:.2f}")
        self.log(f"  Intrinsic rewards: {ep_intrinsic_returns}")
        self.log(f"  Agent deaths: {ep_deaths}")
        self.log(f"  Survival rate: {survival_rate:.1f}%")
        self.log(f"  Reached max steps: {reached_max_steps}")
        self.log("-" * 40)
        
        self.episodes_completed += 1
    
    def generate_summary(self, elapsed_time):
        """Generate and log summary statistics and create visualisation plots"""
        self._generate_summary_statistics(elapsed_time)
        self._plot_rewards()
        
        if self.agent_type == "curious_dqn":
            self._plot_intrinsic_rewards()
            
        self._plot_episode_lengths()
        self._plot_additional_metrics()
        
        self.log_file.close()
    
    def _generate_summary_statistics(self, elapsed_time):
        """Generate and log summary statistics for the simulation"""
        # Calculate overall statistics
        avg_reward_per_agent = self.cumulative_rewards / self.episodes_completed if self.episodes_completed > 0 else 0
        avg_reward_per_episode = np.mean(self.avg_episode_rewards) if self.avg_episode_rewards else 0
        avg_episode_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
        median_episode_length = np.median(self.episode_lengths) if self.episode_lengths else 0
        max_episode_length = max(self.episode_lengths) if self.episode_lengths else 0
        min_episode_length = min(self.episode_lengths) if self.episode_lengths else 0
        avg_survival_rate = np.mean(self.episode_survival_rates) if self.episode_survival_rates else 0
        avg_deaths_per_episode = np.mean(self.agent_deaths_per_episode) if self.agent_deaths_per_episode else 0
        
        # Calculate standard deviations
        std_reward = np.std(self.avg_episode_rewards) if self.avg_episode_rewards else 0
        std_episode_length = np.std(self.episode_lengths) if self.episode_lengths else 0
        std_survival_rate = np.std(self.episode_survival_rates) if self.episode_survival_rates else 0
        
        # Write summary to log file
        self.log("\n" + "=" * 50)
        self.log("SIMULATION SUMMARY")
        self.log("=" * 50)
        self.log(f"Completed {self.episodes_completed} episodes with {self.n_agents} agents")
        self.log(f"Agent type: {self.agent_type}")
        self.log(f"Total simulation time: {elapsed_time:.2f} seconds")
        self.log(f"Average time per episode: {elapsed_time/max(1, self.episodes_completed):.2f} seconds")
        
        self.log("\nREWARD STATISTICS:")
        self.log(f"  Peak total episode reward: {self.peak_reward:.2f} (Episode {self.peak_reward_episode})")
        self.log(f"  Average reward per episode: {avg_reward_per_episode:.2f} (±{std_reward:.2f})")
        self.log(f"  Average reward per agent: {avg_reward_per_agent}")
        self.log(f"  Top {len(self.top_episodes)} episodes by reward:")
        for rank, (ep_num, reward) in enumerate(self.top_episodes, 1):
            self.log(f"    #{rank}: Episode {ep_num} with reward {reward:.2f}")
            
        self.log("\nEPISODE STATISTICS:")
        self.log(f"  Episodes reaching maximum steps ({self.max_timesteps}): {self.episodes_reached_max_steps} ({self.episodes_reached_max_steps/max(1, self.episodes_completed)*100:.1f}%)")
        self.log(f"  Average episode length: {avg_episode_length:.2f} steps (±{std_episode_length:.2f})")
        self.log(f"  Median episode length: {median_episode_length:.1f} steps")
        self.log(f"  Min/Max episode length: {min_episode_length}/{max_episode_length} steps")
        
        self.log("\nSURVIVAL STATISTICS:")
        self.log(f"  Average survival rate: {avg_survival_rate:.1f}% (±{std_survival_rate:.1f}%)")
        self.log(f"  Average deaths per episode: {avg_deaths_per_episode:.2f}")
        for i in range(self.n_agents):
            death_rate = self.agent_lifetime_stats[i]["deaths"] / max(1, self.episodes_completed) * 100
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
        if len(self.episode_rewards[0]) == 0:
            self.log("No episodes completed, skipping reward plots")
            return
            
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

        # Plot average episodic reward
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
        
        # Combined rewards plot
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
        if len(self.episode_intrinsic_rewards[0]) == 0:
            self.log("No episodes completed, skipping intrinsic reward plots")
            return
            
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
        plt.close()

    def _plot_episode_lengths(self):
        """Plots the episode lengths over training time."""
        if len(self.episode_lengths) == 0:
            self.log("No episodes completed, skipping episode length plots")
            return
            
        plt.figure(figsize=(10, 5))
        plt.plot(
            range(1, len(self.episode_lengths) + 1),
            self.episode_lengths,
            marker="o",
            linestyle="-",
            label="Episode Length",
        )

        # Optional: Rolling average for smoothing
        window = min(10, len(self.episode_lengths))
        if len(self.episode_lengths) > window:
            rolling_avg = np.convolve(
                self.episode_lengths, np.ones(window) / window, mode="valid"
            )
            plt.plot(
                range(window, len(self.episode_lengths) + 1),
                rolling_avg,
                linestyle="--",
                color="red",
                label=f"Rolling Avg ({window} episodes)",
            )

        plt.xlabel("Episode")
        plt.ylabel("Steps Until Termination")
        plt.title("Episode Length Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.log_dir}/episode_length_over_time.png")
        plt.close()
        
    def _plot_additional_metrics(self):
        """Plot additional performance metrics."""
        if len(self.episode_survival_rates) == 0:
            self.log("No episodes completed, skipping additional metrics plots")
            return
            
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


