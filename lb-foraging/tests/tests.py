"""Unit tests for the Sustainable Foraging Environment."""

import unittest
import numpy as np
import gymnasium as gym
import lbforaging  # noqa
from lbforaging.foraging.environment import Action, ForagingEnv

class TestSustainableForagingEnvironment(unittest.TestCase):
    """Test cases for the Sustainable Foraging Environment."""

    def setUp(self):
        """Set up a test environment before each test."""
        # Create a simple 3x3 environment with a single agent and single food
        self.env = gym.make("SustainableForagingEnv-v0", render_mode=None)
        self.env.reset(seed=42)  # Use fixed seed for reproducibility
        
        # Force a specific setup for testing
        self.env.unwrapped.field[:] = 0
        self.env.unwrapped.field[2, 2] = 1  # Place food at bottom right
        self.env.unwrapped._food_spawned = self.env.unwrapped.field.sum()
        
        # Place agent at top left
        self.env.unwrapped.players[0].position = (0, 0)
        
        # Create a simple mock controller for the agent if none exists
        if not hasattr(self.env.unwrapped.players[0], 'controller') or self.env.unwrapped.players[0].controller is None:
            # Create a simple controller-like object with required attributes
            class MockController:
                def __init__(self):
                    self.energy = 10
                    self.position = (0, 0)
                    self.survival_cost = 1
                    
                def _step(self, obs):
                    return Action.NONE
                    
                def receive_reward(self, reward):
                    pass
                    
                def notify_food_loaded(self, food_loaded):
                    if food_loaded:
                        self.energy += 3
            
            self.env.unwrapped.players[0].controller = MockController()
            # Link the controller's position to the player's position
            self.env.unwrapped.players[0].controller.position = self.env.unwrapped.players[0].position
        else:
            # Ensure controller has energy set
            self.env.unwrapped.players[0].controller.energy = 10
            self.env.unwrapped.players[0].controller.survival_cost = 1
            
        # Ensure environment knows about valid actions
        self.env.unwrapped.test_gen_valid_moves()

    def tearDown(self):
        """Clean up after each test."""
        self.env.close()

    def test_agent_movement(self):
        """Test basic agent movement."""
        # Move agent SOUTH
        obs, rewards, done, _, _ = self.env.step([Action.SOUTH.value])
        self.assertEqual(self.env.unwrapped.players[0].position, (1, 0))
        
        # Move agent EAST
        obs, rewards, done, _, _ = self.env.step([Action.EAST.value])
        self.assertEqual(self.env.unwrapped.players[0].position, (1, 1))

    def test_boundary_conditions(self):
        """Test that agents cannot move beyond grid boundaries."""
        # Test upper boundary (row 0)
        self.env.unwrapped.players[0].position = (0, 1)
        self.env.unwrapped.test_gen_valid_moves()
        obs, _, _, _, _ = self.env.step([Action.NORTH.value])
        self.assertEqual(self.env.unwrapped.players[0].position, (0, 1), 
                        "Agent should not be able to move beyond top boundary")
        
        # Test left boundary (column 0)
        self.env.unwrapped.players[0].position = (1, 0)
        self.env.unwrapped.test_gen_valid_moves()
        obs, _, _, _, _ = self.env.step([Action.WEST.value])
        self.assertEqual(self.env.unwrapped.players[0].position, (1, 0), 
                        "Agent should not be able to move beyond left boundary")
        
        # Test lower boundary (row 2 in a 3x3 grid)
        self.env.unwrapped.players[0].position = (2, 1)
        self.env.unwrapped.test_gen_valid_moves()
        obs, _, _, _, _ = self.env.step([Action.SOUTH.value])
        self.assertEqual(self.env.unwrapped.players[0].position, (2, 1), 
                        "Agent should not be able to move beyond bottom boundary")
        
        # Test right boundary (column 2 in a 3x3 grid)
        self.env.unwrapped.players[0].position = (1, 2)
        self.env.unwrapped.test_gen_valid_moves()
        obs, _, _, _, _ = self.env.step([Action.EAST.value])
        self.assertEqual(self.env.unwrapped.players[0].position, (1, 2), 
                        "Agent should not be able to move beyond right boundary")

    def test_reset_functionality(self):
        """Test that the environment resets correctly."""
        # Make changes to the environment
        self.env.unwrapped.players[0].position = (1, 1)
        self.env.unwrapped.field[:] = 0
        
        # Reset the environment
        obs, _ = self.env.reset()
        
        # Check that the observation is properly formatted
        self.assertIsNotNone(obs)
        
        # Check that player positions have been reset
        for player in self.env.unwrapped.players:
            self.assertIsNotNone(player.position)

    def test_food_loading(self):
        """Test that agent can load food when adjacent."""
        # Position agent adjacent to food
        self.env.unwrapped.players[0].position = (1, 2)
        self.env.unwrapped.players[0].controller.position = (1, 2)
        self.env.unwrapped.test_gen_valid_moves()
        
        # Get initial energy
        initial_energy = self.env.unwrapped.players[0].controller.energy
        
        # Attempt to load food
        self.env.step([Action.LOAD.value])
         
        # Check if energy increased
        current_energy = self.env.unwrapped.players[0].controller.energy
        self.assertGreater(current_energy, initial_energy, 
                          f"Energy should increase after loading food. Initial: {initial_energy}, Current: {current_energy}")
            

if __name__ == "__main__":
    unittest.main()
