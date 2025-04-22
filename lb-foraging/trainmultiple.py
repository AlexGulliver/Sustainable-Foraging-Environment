"Run multiple simulations of the Sustainable Foraging"
"environment with different agent types and plot learning curves."""

from aggregatetrainer import run_multiple_simulations, plot_learning_curves

if __name__ == "__main__":
    agent_types = ["random", "dqn", "curious_dqn"]
    
    # Run simulations
    results, results_dir, num_runs = run_multiple_simulations(
        agent_types=agent_types,
        num_runs=10,
        num_episodes=20000,
        max_steps=50,
        starting_energy=10,
        energy_cost=1,
    )

    plot_learning_curves(results_dir, agent_types, num_runs)
    
    print(f"\nSimulation completed! Aggregate results saved to {results_dir}")
