'''Script to verify the environment will run.'''

from argparse import ArgumentParser
import warnings
import numpy as np
import gymnasium as gym
from lbforaging.foraging.environment import Action
import pyglet
import time

def parse_args():
    parser = ArgumentParser()
    # parser.add_argument(
    #     "--env",
    #     type=str,
    #     default="Foraging-8x8-2p-6f-v3",
    #     help="Environment to use",
    # )
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
    return parser.parse_args()


class VisualisedEnv:
    """Run a visualised environment."""

    def __init__(self, env: str, max_steps: int, display_info: bool = True):
        self.env = gym.make(env, render_mode="human", max_episode_steps=max_steps)
        self.n_agents = self.env.unwrapped.n_agents
        self.display_info = display_info

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

    def _run_episodes(self, num_episodes=1):
        """Runs the environment for a specified number of episodes (default 100)."""
        total_rewards = np.zeros(self.n_agents)

        for episode in range(num_episodes):
            ep_returns = np.zeros(self.n_agents)  # Reset episode rewards
            obss, _ = self.env.reset()
            self.env.render()

            if self.display_info:
                print(f"Episode {episode + 1} begins.")

            for step in range(10):  # Run up to 100 steps per episode
                # TODO: add replenish food function here
                time.sleep(1)
                actions = [np.random.choice(list(Action)) for _ in range(self.n_agents)]
                obss, rews, done, trunc, _ = self.env.step([act.value for act in actions])
                ep_returns += np.array(rews)

                if self.display_info:
                    print(f"Step {step + 1}: Rewards {rews}")

                self.env.render()

                if done or trunc:
                    break  # Stop episode if terminated early
                    
            total_rewards += ep_returns  # Accumulate rewards

            if self.display_info:
                print(f"Episode {episode + 1} reward: {ep_returns}\n")
        print(f"Total rewards after 100 episodes: {total_rewards}")
        time.sleep(15)
        self.env.close()


if __name__ == "__main__":
    args = parse_args()
    VisualisedEnv(env=args.env, display_info=True, max_steps=args.max_steps)
