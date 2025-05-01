"""Aggregated trainer for running multiple simulations with 
different agent types and generating summary statistics and plots."""

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
from foragingtrainer import ForagingTrainer

def run_multiple_simulations(
        agent_types, 
        num_runs, 
        max_steps, 
        num_episodes, 
        starting_energy, 
        energy_cost,
        replenishment_rate,
        food_energy_value,
        render_mode,
        display_info,
        env_name="SustainableForagingEnv-v0"
        ):
    """Run multiple simulations for each agent type and collect aggregate results"""
    # Create directory for aggregate results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"aggregate_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Dictionary to store aggregated results
    results = {agent_type: {
        "avg_rewards": [],
        "peak_rewards": [],
        "avg_episode_lengths": [],
        "survival_rates": [],
        "episodes_max_steps": [],
        "avg_deaths": [],
        "simulation_times": []
    } for agent_type in agent_types}
    
    # Run simulations for each agent type
    for agent_type in agent_types:
        print(f"\n{'='*50}")
        print(f"Starting {num_runs} simulations for {agent_type} agent")
        print(f"{'='*50}")
        
        for run in range(num_runs):
            print(f"\nRun {run+1}/{num_runs} for {agent_type} agent")
            
            # Create log directory for this run
            run_log_dir = f"{results_dir}/{agent_type}_run_{run+1}"
            
            # Initialise trainer with the current agent type
            trainer = ForagingTrainer(
                env_name=env_name,
                max_steps=max_steps,
                num_episodes=num_episodes,
                agent_type=agent_type,
                starting_energy=starting_energy,
                energy_cost=energy_cost,
                replenishment_rate=replenishment_rate,
                food_energy_value=food_energy_value,
                display_info=display_info,
                log_dir=run_log_dir,
                render_mode=render_mode,
            )
            
            # Train the agent
            trainer.train()
            
            # Extract and store results
            logger = trainer.logger
            
            # Extract metrics
            avg_reward = np.mean(logger.avg_episode_rewards) if logger.avg_episode_rewards else 0
            peak_reward = logger.peak_reward if hasattr(logger, 'peak_reward') else 0
            avg_episode_length = np.mean(logger.episode_lengths) if logger.episode_lengths else 0
            survival_rate = np.mean(logger.episode_survival_rates) if logger.episode_survival_rates else 0
            episodes_max_steps = logger.episodes_reached_max_steps
            avg_deaths = np.mean(logger.agent_deaths_per_episode) if logger.agent_deaths_per_episode else 0
            sim_time = logger.elapsed_time if hasattr(logger, 'elapsed_time') else 0
            
            # Store metrics for this run
            results[agent_type]["avg_rewards"].append(avg_reward)
            results[agent_type]["peak_rewards"].append(peak_reward)
            results[agent_type]["avg_episode_lengths"].append(avg_episode_length)
            results[agent_type]["survival_rates"].append(survival_rate)
            results[agent_type]["episodes_max_steps"].append(episodes_max_steps)
            results[agent_type]["avg_deaths"].append(avg_deaths)
            results[agent_type]["simulation_times"].append(sim_time)
            
    # Generate summary statistics and plots
    generate_summary(results, agent_types, num_runs, results_dir, num_episodes)
    
    return results, results_dir, num_runs

def plot_learning_curves(results_dir, agent_types, num_runs):
    """Plot learning curves from saved episode data"""
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    
    colors = {
        "random": "gray",
        "qlearning": "green",
        "dqn": "blue",
        "curious_dqn": "purple"
    }
    
    agent_labels = {
        "random": "Random Agent",
        "qlearning": "Q-Learning",
        "dqn": "DQN",
        "curious_dqn": "Curious DQN"
    }
    
    plt.figure(figsize=(18, 10))
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle("Learning Curves Analysis", fontsize=16)
    
    ax1 = axes[0, 0]
    
    window_size = 100
    
    all_agent_data = {}
    max_episodes = 0
    
    for agent_type in agent_types:
        all_runs_data = []
        
        for run in range(1, num_runs + 1):
            try:
                csv_path = f"{results_dir}/{agent_type}_run_{run}/episode_rewards.csv"
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    all_runs_data.append(df["Average_Reward"].values)
                    max_episodes = max(max_episodes, len(df["Average_Reward"]))
            except Exception as e:
                print(f"Error loading data for {agent_type} run {run}: {e}")
        
        if all_runs_data:
            min_length = min(len(data) for data in all_runs_data)
            all_runs_data = [data[:min_length] for data in all_runs_data]
            
            all_runs = np.array(all_runs_data)
            
            mean_rewards = np.mean(all_runs, axis=0)
            std_rewards = np.std(all_runs, axis=0)
            
            all_agent_data[agent_type] = {
                "mean": mean_rewards,
                "std": std_rewards,
                "length": min_length
            }
    
    for agent_type in agent_types:
        if agent_type in all_agent_data:
            data = all_agent_data[agent_type]
            episodes = np.arange(1, data["length"] + 1)
            
            mean_smooth = pd.Series(data["mean"]).rolling(window=window_size, min_periods=1).mean().values
            
            ax1.plot(episodes, mean_smooth, 
                    label=agent_labels.get(agent_type, agent_type),
                    color=colors.get(agent_type, "blue"),
                    linewidth=2)
    
    ax1.set_title("Smoothed Learning Curves (Moving Average)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Average Reward")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    for i, agent_type in enumerate(agent_types):
        if agent_type in all_agent_data:
            data = all_agent_data[agent_type]
            episodes = np.arange(1, data["length"] + 1)
            
            mean_smooth = pd.Series(data["mean"]).rolling(window=window_size, min_periods=1).mean().values
            std_smooth = pd.Series(data["std"]).rolling(window=window_size, min_periods=1).mean().values
            
            ax = fig.add_subplot(2, len(agent_types), len(agent_types) + i + 1)
            
            ax.plot(episodes, mean_smooth, 
                   color=colors.get(agent_type, "blue"),
                   linewidth=2)
            
            sample_rate = max(1, len(episodes) // 100)
            sampled_episodes = episodes[::sample_rate]
            sampled_mean = mean_smooth[::sample_rate]
            sampled_std = std_smooth[::sample_rate]
            
            ax.fill_between(
                sampled_episodes,
                sampled_mean - sampled_std,
                sampled_mean + sampled_std,
                alpha=0.2,
                color=colors.get(agent_type, "blue")
            )
            
            ax.set_title(f"{agent_labels.get(agent_type, agent_type)} Learning Curve")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Average Reward")
            ax.grid(True, linestyle='--', alpha=0.7)
    
    ax4 = axes[0, 1]
    ax4.set_visible(False)
    
    ax3 = axes[1, 0]
    ax3.set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    
    plt.savefig(f"{results_dir}/learning_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    
    for agent_type in agent_types:
        if agent_type in all_agent_data:
            data = all_agent_data[agent_type]
            episodes = np.arange(1, data["length"] + 1)
            
            mean_smooth = pd.Series(data["mean"]).rolling(window=window_size, min_periods=1).mean().values
            
            plt.plot(episodes, mean_smooth, 
                    label=agent_labels.get(agent_type, agent_type),
                    color=colors.get(agent_type, "blue"),
                    linewidth=2)
    
    plt.title("Smoothed Learning Curves Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(f"{results_dir}/smooth_learning_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary(results, agent_types, num_runs, results_dir, num_episodes):
    """Generate summary statistics and plots for the aggregate results"""
    
    # Create summary file
    with open(f"{results_dir}/aggregate_summary.txt", "w") as f:
        f.write(f"Aggregate Results Summary\n")
        f.write(f"="*30)
        f.write(f"Number of runs per agent type: {num_runs}\n")
        f.write(f"Number of episodes per run: {num_episodes}\n\n")
        
        # Table header
        f.write(f"{'Metric':<25} | ")
        for agent_type in agent_types:
            f.write(f"{agent_type:<15} | ")
        f.write("\n")
        f.write(f"{'-'*25} | ")
        for _ in agent_types:
            f.write(f"{'-'*15} | ")
        f.write("\n")
        
        # Metrics to report
        metrics = [
            ("Average Reward", "avg_rewards"),
            ("Peak Reward", "peak_rewards"),
            ("Avg Episode Length", "avg_episode_lengths"),
            ("Survival Rate %", "survival_rates"),
            ("Episodes Max Steps", "episodes_max_steps"),
            ("Avg Deaths per Episode", "avg_deaths"),
            ("Simulation Time (s)", "simulation_times")
        ]
        
        # Write metrics
        for metric_name, metric_key in metrics:
            f.write(f"{metric_name:<25} | ")
            for agent_type in agent_types:
                values = results[agent_type][metric_key]
                avg_value = np.mean(values)
                std_value = np.std(values)
                f.write(f"{avg_value:.2f} (±{std_value:.2f}) | ")
            f.write("\n")
    
    # Generate comparative plots
    generate_comparison_plots(results, agent_types, results_dir)

def generate_comparison_plots(results, agent_types, results_dir):
    """Generate plots comparing agent types across metrics"""
    
    # Colors for each agent type
    colors = {
        "random": "gray",
        "qlearning": "green",
        "dqn": "blue",
        "curious_dqn": "purple"
    }
    
    # Metrics to plot
    metrics = [
        ("avg_rewards", "Average Reward"),
        ("avg_episode_lengths", "Average Episode Length"),
        ("survival_rates", "Survival Rate (%)"),
    ]
    
    # Bar charts for each metric
    for metric_key, metric_name in metrics:
        # Create figure with additional space at the top
        plt.figure(figsize=(10, 6))
        
        # Data for bar chart
        means = [np.mean(results[agent_type][metric_key]) for agent_type in agent_types]
        stds = [np.std(results[agent_type][metric_key]) for agent_type in agent_types]
        
        # Create bar chart
        bars = plt.bar(agent_types, means, yerr=stds, alpha=0.7, 
                 color=[colors.get(agent_type, "blue") for agent_type in agent_types])
        
        # Add values on top of bars
        for i, bar in enumerate(bars):
            # Calculate vertical position
            height = bar.get_height()
            y_pos = height + stds[i] if stds[i] > 0 else height
            
            plt.text(
                bar.get_x() + bar.get_width()/2, 
                y_pos + 0.01 * max(means),  # Small offset above error bars
                f"{means[i]:.2f}", 
                ha='center', 
                va='bottom', 
                fontsize=10,
                fontweight='bold'
            )
        
        plt.title(f"Comparison of {metric_name} Across Agent Types")
        plt.ylabel(metric_name)
        plt.xlabel("Agent Type")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Explicitly set y-axis limits to add space for the labels
        y_max = max(means) + max(stds) + max(means) * 0.1
        plt.ylim(0, y_max)
        
        plt.tight_layout()
        
        # Save the figure with high DPI
        plt.savefig(f"{results_dir}/{metric_key}_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
