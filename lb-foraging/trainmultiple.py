"Run multiple simulations of the Sustainable Foraging"
"environment with different agent types and plot learning curves."""

from aggregatetrainer import run_multiple_simulations, plot_learning_curves

if __name__ == "__main__":
    agent_types = ["random", "dqn", "curious_dqn"]
    
    # Run simulations
    results, results_dir, num_runs = run_multiple_simulations(
        agent_types=agent_types,
        num_runs=1,
        num_episodes=50000,
        max_steps=30,
        starting_energy=10,
        energy_cost=1,
        replenishment_rate=1, # Rate at which food is replenished
        food_energy_value=3, # Energy value of food
        display_info=True,
        render_mode="human",
    )

    plot_learning_curves(results_dir, agent_types, num_runs)
    
    print(f"\nSimulation completed! Aggregate results saved to {results_dir}")
