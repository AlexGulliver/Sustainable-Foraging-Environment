import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np
from stable_baselines3 import PPO
from argparse import ArgumentParser
import pyglet
from gymnasium.envs.registration import register

# Register the custom environment
register(
    id="Foraging-8x8-2p-4f-v3",
    entry_point="lbforaging.lbforaging.foraging.environment.ForagingEnv",  # Replace with the actual path to your environment class
)

# Flatten the action space from Tuple(Discrete(6), Discrete(6)) to a single Discrete space
class FlattenedActionSpace(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Flatten the Tuple(Discrete(6), Discrete(6)) to a single Discrete space
        self.action_space = Discrete(self.env.action_space[0].n * self.env.action_space[1].n)

    def action(self, action):
        # Convert the flattened action back to a Tuple(Discrete(6), Discrete(6))
        action_1 = action // self.env.action_space[1].n
        action_2 = action % self.env.action_space[1].n
        return (action_1, action_2)

    def reverse_action(self, action):
        # Convert the Tuple(Discrete(6), Discrete(6)) back to a flattened action
        return action[0] * self.env.action_space[1].n + action[1]

# Visualized Environment Wrapper for running the environment with an RL agent
class VisualisedEnv:
    """Run a visualised environment with an RL agent."""

    def __init__(self, env: str, max_steps: int, display_info: bool = True):
        # Create and wrap the environment
        self.env = gym.make(env, render_mode="human", max_episode_steps=max_steps)
        self.env = FlattenedActionSpace(self.env)  # Apply the custom action space wrapper
        self.n_agents = self.env.unwrapped.n_agents
        self.display_info = display_info

        # Initialize the PPO model
        self.model = PPO("MlpPolicy", self.env, verbose=1)

        # Reset the environment to initialize it
        obss, _ = self.env.reset()
        self.env.render()  # Ensures the viewer is created

        # Wait for the viewer to initialize
        if hasattr(self.env.unwrapped, "viewer") and self.env.unwrapped.viewer:
            self.env.unwrapped.viewer.window.on_key_press = self._key_press
        else:
            print("Warning: Viewer not initialized. Keyboard input may not work.")

        self._run_episodes(num_episodes=100)

    def _key_press(self, k, mod):
        from pyglet.window import key
        if k == key.ESCAPE:
            self.running = False
            self.env.close()

    def _run_episodes(self, num_episodes=10):
        """Runs the environment for a specified number of episodes (default 100)."""
        total_rewards = np.zeros(self.n_agents)

        for episode in range(num_episodes):
            ep_returns = np.zeros(self.n_agents)  # Reset episode rewards
            obss, _ = self.env.reset()
            self.env.render()

            if self.display_info:
                print(f"Episode {episode + 1} begins.")

            for step in range(100):  # Run up to 100 steps per episode
                # Use the RL model to select actions
                actions, _states = self.model.predict(obss, deterministic=True)

                # Step the environment with the selected actions
                obss, rews, done, trunc, _ = self.env.step(actions)
                ep_returns += np.array(rews)

                if self.display_info:
                    print(f"Step {step + 1}: Rewards {rews}")

                self.env.render()

                if done or trunc:
                    break  # Stop episode if terminated early

            total_rewards += ep_returns  # Accumulate rewards

            if self.display_info:
                print(f"Episode {episode + 1} reward: {ep_returns}\n")
        
        print(f"Total rewards after {num_episodes} episodes: {total_rewards}")
        self.env.close()

def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("--env", type=str, default="Foraging-8x8-2p-4f-v3", help="Environment to use")
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

if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()
    
    # Run the visualized environment
    VisualisedEnv(env=args.env, display_info=True, max_steps=args.max_steps)
