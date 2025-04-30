"""Handles logging, data collection, and data visualisation for foraging simulations"""

import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import seaborn as sns

# Global Plotting Configuration
PLOT_CONFIG = {
    'font_scale': 1.5,
    'title_size': 20,
    'label_size': 18,
    'tick_size': 12,
    'legend_size': 12,
    'dpi': 300,
    'figure_size': (12, 8)
}

class DataCollector:
    
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
        # Plot styling
        plt.rcParams.update({
            'font.size': PLOT_CONFIG['font_scale'] * 10,
            'axes.titlesize': PLOT_CONFIG['title_size'],
            'axes.labelsize': PLOT_CONFIG['label_size'],
            'xtick.labelsize': PLOT_CONFIG['tick_size'],
            'ytick.labelsize': PLOT_CONFIG['tick_size'],
            'legend.fontsize': PLOT_CONFIG['legend_size']
        })

        # Setup logging directory and file
        self.log_dir = log_dir
        self.log_file = open(f"{log_dir}/simulation_log.txt", "w")
        
        # Store simulation parameters
        self.n_agents = n_agents
        self.agent_type = agent_type
        self.max_timesteps = max_timesteps
        self.starting_energy = starting_energy
        self.display_info = display_info
        self.num_episodes = num_episodes
        
        # Data collection
        self.episode_rewards = [[] for _ in range(self.n_agents)]
        self.episode_intrinsic_rewards = [[] for _ in range(self.n_agents)]
        self.avg_episode_rewards = []
        self.avg_episode_intrinsic_rewards = []
        self.episode_lengths = []
        self.cumulative_rewards = np.zeros(self.n_agents)
        self.cumulative_intrinsic_rewards = np.zeros(self.n_agents)
        self.episode_actions = [[] for _ in range(n_agents)]
    
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
    
    def update_episode_data(self, episode_num, ep_returns, ep_intrinsic_returns, ep_length, ep_deaths, reached_max_steps, ep_actions=None):
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

        if ep_actions is not None:
            for i in range(len(ep_actions)):
                self.episode_actions[i].append(ep_actions[i])
        
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
        # if ep_actions:
        #     self.log(f"  Actions: {[str(actions) for actions in ep_actions]}")
        
        self.episodes_completed += 1

    def save_episode_data(self, save_path):
        """Save episode-by-episode data to CSV files for external analysis"""
        import csv
        
        # Save average rewards per episode
        with open(f"{save_path}/episode_rewards.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Average_Reward"])
            for i, reward in enumerate(self.avg_episode_rewards):
                writer.writerow([i+1, reward])
        
        # Save episode lengths
        with open(f"{save_path}/episode_lengths.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Length"])
            for i, length in enumerate(self.episode_lengths):
                writer.writerow([i+1, length])
        
        # Save survival rates
        with open(f"{save_path}/episode_survival.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Survival_Rate"])
            for i, rate in enumerate(self.episode_survival_rates):
                writer.writerow([i+1, rate])
    
    def generate_summary(self, elapsed_time):
        """Generate and log summary statistics and create visualisation plots"""
        self._generate_summary_statistics(elapsed_time)
        self._plot_rewards()
        
        if self.agent_type == "curious_dqn":
            self._plot_intrinsic_rewards()
            self._plot_intrinsic_vs_extrinsic_rewards()  # New call to plot comparison
            
        self._plot_episode_lengths()
        
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
        self.log(f"  Episode length plot: {self.log_dir}/episode_length_over_time.png")
        if self.agent_type == "curious_dqn":
            self.log(f"  Intrinsic rewards plot: {self.log_dir}/intrinsic_rewards_plot.png")
        # Removed reference to additional_metrics.png
        self.log("=" * 50)
    
    def _plot_rewards(self):
        """Plots the rewards over episodes and average episodic reward trend as separate files."""
        if len(self.episode_rewards[0]) == 0:
            self.log("No episodes completed, skipping reward plots")
            return

        # Always use the large dataset approach for consistency
        self._plot_rewards_large_dataset()
            
    def _plot_rewards_large_dataset(self):
        """
        Optimised plotting for better visualisation.
        """
        total_episodes = len(self.episode_rewards[0])
        
        # Use a rolling window to smooth out fluctuations
        window_size = max(10, int(total_episodes / 100))
        
        # Create a plot for smoothed agent rewards
        plt.figure(figsize=PLOT_CONFIG['figure_size'])
        
        # Create rolling window averages for each agent
        for i, rewards in enumerate(self.episode_rewards):
            # Calculate rolling average
            rolling_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            # Plot with fewer markers for clarity
            plt.plot(
                range(window_size, len(rewards) + 1), 
                rolling_avg,
                label=f"Agent {i + 1} (Smoothed)",
                linewidth=1.5
            )
        
        # Add the average reward line
        rolling_avg_all = np.convolve(self.avg_episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(
            range(window_size, len(self.avg_episode_rewards) + 1),
            rolling_avg_all,
            color="black",
            linestyle="-",
            linewidth=3,
            label=f"Average Reward ({window_size}-episode Moving Average)"
        )

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Agent Rewards per Episode (Smoothed over {window_size} episodes)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/agent_rewards_per_episode.png", dpi=PLOT_CONFIG['dpi'])
        plt.close()

        # Plot average episodic reward
        plt.figure(figsize=PLOT_CONFIG["figure_size"])
        
        # Compute cumulative average reward
        avg_episodic_reward = np.cumsum(self.avg_episode_rewards) / np.arange(
            1, len(self.avg_episode_rewards) + 1
        )
        
        # Create a smoothed version for easier readability
        if total_episodes > 500:
            # Subsample for very large datasets
            sample_rate = int(total_episodes / 500)
            plt.plot(
                range(1, len(avg_episodic_reward) + 1, sample_rate),
                avg_episodic_reward[::sample_rate],
                color="green",
                linestyle="-",
                linewidth=2,
                label="Cumulative Average Reward"
            )
        else:
            plt.plot(
                range(1, len(avg_episodic_reward) + 1),
                avg_episodic_reward,
                color="green",
                linestyle="-",
                linewidth=2,
                label="Cumulative Average Reward"
            )

        plt.xlabel("Episode")
        plt.ylabel("Average Episodic Reward")
        plt.title("Average Episodic Reward Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/average_episodic_reward.png", dpi=PLOT_CONFIG['dpi'])
        plt.close()

    def _plot_intrinsic_rewards(self):
        """Plots the intrinsic rewards over episodes and average episodic intrinsic reward trend."""
        if len(self.episode_intrinsic_rewards[0]) == 0:
            self.log("No episodes completed, skipping intrinsic reward plots")
            return
            
        # Always use the large dataset approach for consistency
        self._plot_intrinsic_rewards_large_dataset()
    
    def _plot_intrinsic_rewards_large_dataset(self):
        """
        Optimised plotting for intrinsic rewards with better visualisation.
        """
        total_episodes = len(self.episode_intrinsic_rewards[0])
        window_size = max(10, int(total_episodes / 100))

        plt.figure(figsize=PLOT_CONFIG["figure_size"])

        # Intrinsic rewards subplot - smoothed version
        plt.subplot(2, 1, 1)

        # Plot smoothed individual agent intrinsic rewards
        for i, intrinsic_rewards in enumerate(self.episode_intrinsic_rewards):
            rolling_avg = np.convolve(intrinsic_rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(
                range(window_size, len(intrinsic_rewards) + 1),
                rolling_avg,
                label=f"Agent {i + 1} (Smoothed)",
                linewidth=1.5
            )

        # Smoothed average intrinsic rewards
        avg_smoothed = np.convolve(
            self.avg_episode_intrinsic_rewards, 
            np.ones(window_size)/window_size, 
            mode='valid'
        )
        plt.plot(
            range(window_size, len(self.avg_episode_intrinsic_rewards) + 1),
            avg_smoothed,
            color="black",
            linestyle="-",
            linewidth=3,
            label=f"Average Intrinsic Reward ({window_size}-episode Moving Average)",
        )

        plt.xlabel("Episode")
        plt.ylabel("Intrinsic Reward")
        plt.title(f"Agent Intrinsic Rewards (Smoothed over {window_size} episodes)")
        plt.legend()
        plt.grid(True)

        # Cumulative average subplot
        plt.subplot(2, 1, 2)

        # Average episodic intrinsic rewards
        avg_episodic_intrinsic_reward = np.cumsum(
            self.avg_episode_intrinsic_rewards
        ) / np.arange(1, len(self.avg_episode_intrinsic_rewards) + 1)

        # Subsample for very large datasets
        if total_episodes > 500:
            sample_rate = int(total_episodes / 500)
            plt.plot(
                range(1, len(avg_episodic_intrinsic_reward) + 1, sample_rate),
                avg_episodic_intrinsic_reward[::sample_rate],
                color="blue",
                linestyle="-",
                linewidth=2,
                label="Cumulative Average Intrinsic Reward",
            )
        else:
            plt.plot(
                range(1, len(avg_episodic_intrinsic_reward) + 1),
                avg_episodic_intrinsic_reward,
                color="blue",
                linestyle="-",
                linewidth=2,
                label="Cumulative Average Intrinsic Reward",
            )

        plt.xlabel("Episode")
        plt.ylabel("Average Episodic Intrinsic Reward")
        plt.title("Cumulative Average Intrinsic Reward")
        plt.legend()
        plt.grid(True)

        plt.tight_layout(h_pad=1)
        plt.savefig(f"{self.log_dir}/intrinsic_rewards_plot.png", dpi=PLOT_CONFIG['dpi'])
        plt.close()

    def _plot_episode_lengths(self):
        """Plots the episode lengths over training time."""
        if len(self.episode_lengths) == 0:
            self.log("No episodes completed, skipping episode length plots")
            return
            
        # Determine the window size based on the number of episodes
        total_episodes = len(self.episode_lengths)
        window_size = max(10, int(total_episodes / 50))
            
        plt.figure(figsize=PLOT_CONFIG["figure_size"])
        
        # Calculate rolling average
        rolling_avg = np.convolve(
            self.episode_lengths, np.ones(window_size) / window_size, mode="valid"
        )
        
        # Plot raw data with very light blue color instead of low opacity
        plt.plot(
            range(1, len(self.episode_lengths) + 1),
            self.episode_lengths,
            marker="", 
            linestyle="-",
            color="lightblue",
            label="Episode Length"
        )
        
        # Plot smoothed data with stronger color
        plt.plot(
            range(window_size, len(self.episode_lengths) + 1),
            rolling_avg,
            linestyle="-",
            linewidth=2,
            color="blue",
            label=f"Moving Average ({window_size} episodes)"
        )

        plt.xlabel("Episode")
        plt.ylabel("Steps Until Termination")
        plt.title("Episode Length Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/episode_length_over_time.png", dpi=PLOT_CONFIG['dpi'])
        plt.close()

    def _plot_intrinsic_vs_extrinsic_rewards(self):
        """
        Plots intrinsic vs extrinsic rewards for curious DQN agent to visualise
        the balance between exploration and exploitation during training.
        """
        if self.agent_type != "curious_dqn" or len(self.episode_rewards[0]) == 0:
            self.log("Not a curious DQN agent or no episodes completed, skipping intrinsic vs extrinsic reward plot")
            return
            
        total_episodes = len(self.episode_rewards[0])
        window_size = 50
        # window_size = max(10, int(total_episodes / 100))
        
        plt.figure(figsize=PLOT_CONFIG["figure_size"])
        
        # Subplot 1: Intrinsic vs Extrinsic rewards over time
        plt.subplot(2, 1, 1)
        
        # Calculate average rewards across all agents
        avg_extrinsic = self.avg_episode_rewards
        avg_intrinsic = self.avg_episode_intrinsic_rewards
        
        # Calculate smoothed versions for better visualisation
        if len(avg_extrinsic) > window_size:
            smoothed_extrinsic = np.convolve(avg_extrinsic, np.ones(window_size)/window_size, mode='valid')
            smoothed_intrinsic = np.convolve(avg_intrinsic, np.ones(window_size)/window_size, mode='valid')
            episodes_range = range(window_size, len(avg_extrinsic) + 1)
            
            plt.plot(
                episodes_range, 
                smoothed_extrinsic, 
                color="green", 
                linewidth=2, 
                label=f"Extrinsic Reward ({window_size}-episode Moving Average)"
            )
            plt.plot(
                episodes_range, 
                smoothed_intrinsic, 
                color="purple", 
                linewidth=2, 
                label=f"Intrinsic Reward ({window_size}-episode Moving Average)"
            )
        else:
            # When we don't have enough episodes for smoothing
            plt.plot(
                range(1, len(avg_extrinsic) + 1), 
                avg_extrinsic, 
                color="green", 
                linewidth=2, 
                label="Extrinsic Reward"
            )
            plt.plot(
                range(1, len(avg_intrinsic) + 1), 
                avg_intrinsic, 
                color="purple", 
                linewidth=2, 
                label="Intrinsic Reward"
            )
        
        plt.xlabel("Episode")
        plt.ylabel("Reward Value")
        plt.title("Intrinsic vs Extrinsic Rewards Over Time")
        plt.grid(True)
        plt.legend()
        
        # Subplot 2: Ratio of intrinsic to extrinsic rewards
        plt.subplot(2, 1, 2)
        
        # Calculate ratio (avoid division by zero)
        ratios = []
        for ex, intr in zip(avg_extrinsic, avg_intrinsic):
            if abs(ex) < 1e-10:  # Prevent division by zero
                ratios.append(0 if intr == 0 else 1e6 if intr > 0 else -1e6)
            else:
                ratios.append(intr / ex)
        
        # Normalise extremely large values for visualisation
        normalised_ratios = []
        for r in ratios:
            if r > 10:
                normalised_ratios.append(10)
            elif r < -10:
                normalised_ratios.append(-10)
            else:
                normalised_ratios.append(r)
        
        # Plot the trend of this ratio
        if len(normalised_ratios) > window_size:
            smoothed_ratio = np.convolve(normalised_ratios, np.ones(window_size)/window_size, mode='valid')
            plt.plot(
                range(window_size, len(normalised_ratios) + 1),
                smoothed_ratio,
                color="blue",
                linewidth=2
            )
        else:
            plt.plot(
                range(1, len(normalised_ratios) + 1),
                normalised_ratios,
                color="blue",
                linewidth=2
            )
        
        # Add a horizontal line at ratio = 1 (balanced rewards)
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label="Balanced rewards (ratio=1)")
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        plt.xlabel("Episode")
        plt.ylabel("Intrinsic/Extrinsic Ratio (capped at ±10)")
        plt.title("Ratio of Intrinsic to Extrinsic Rewards")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/intrinsic_vs_extrinsic_rewards.png", dpi=PLOT_CONFIG["dpi"])
        plt.close()
