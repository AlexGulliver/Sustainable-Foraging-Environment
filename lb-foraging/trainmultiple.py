#!/usr/bin/env python3
"""Run multiple simulations of the Sustainable Foraging
environment with different agent types and plot learning curves."""

from aggregatetrainer import run_multiple_simulations, plot_learning_curves

if __name__ == "__main__":
    agent_types = ["random", "dqn", "curious_dqn"]

    # Run simulations
    results, results_dir, num_runs = run_multiple_simulations(
        agent_types=agent_types,
        num_runs=1,
        num_episodes=50000,
        max_steps=30,  # Maximum number of steps per episode
        starting_energy=10,  # Initial energy for agents
        energy_cost=1,  # Energy cost per step
        replenishment_rate=1,  # Rate at which food is replenished (1.0 = always, 0.5 = 50% chance each step)
        food_energy_value=3,  # Energy value of food when consumed
        display_info=True,  # Whether to display detailed info during training
        render_mode="human",  # Rendering mode ("human" for visualization, None for faster training)
    )

    # Generate learning curve plots for comparison
    plot_learning_curves(results_dir, agent_types, num_runs)

    print(f"\nSimulation completed! Aggregate results saved to {results_dir}")
