"""Run a single simulation of the Sustainable Foraging 
environment with a specified agent type."""

from foragingtrainer import ForagingTrainer

if __name__ == "__main__":
    trainer = ForagingTrainer(
        env_name="SustainableForagingEnv-v0", # Environment to use
        max_steps=30, # Maximum number of steps per episode
        num_episodes=50000, # Number of episodes to run
        agent_type="curious_dqn", # Choose from: "random", "qlearning", "dqn", "curious_dqn"
        starting_energy=10, # Initial energy for agents
        energy_cost=1, # Energy cost per step
        display_info=True, # Whether to display detailed info during training
        log_dir=None, # Directory to save logs (None for auto-generated)
        render_mode=None, # Rendering mode (None for no rendering and faster training, "human" for human rendering)
    )
    
    trainer.train()
