"""Handles the training loop, creating agents, running episodes, and collecting data."""

import warnings
import numpy as np
import gymnasium as gym
from lbforaging.foraging.environment import Action
from lbforaging.agents.qlearningagent import QLearningForagingAgent
from lbforaging.agents.randomagent import RandomForagingAgent
from lbforaging.agents.deepqagent import DeepQLearningForagingAgent
from lbforaging.agents.deepqcuriousagent import CuriosityDrivenDQNAgent
import time
import datetime
import os
from plotting import DataCollector


class ForagingTrainer:
    def __init__(
        self,
        env_name,
        max_steps,
        num_episodes,
        agent_type,
        starting_energy,
        energy_cost,
        display_info=True,
        log_dir=None,
        render_mode=None,
    ):
        # Setup parameters
        self.env_name = env_name
        self.max_steps = max_steps
        self.num_episodes = num_episodes
        self.agent_type = agent_type
        self.starting_energy = starting_energy
        self.display_info = display_info
        self.render_mode = render_mode
        
        # Create timestamp for logs if not provided
        if log_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = f"logs_{agent_type}_{timestamp}"
        else:
            self.log_dir = log_dir
            
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialise environment
        self.env = gym.make(env_name, render_mode=render_mode, max_episode_steps=max_steps)
        self.n_agents = self.env.unwrapped.n_agents
        
        # Initialise logger
        self.logger = DataCollector(
            self.log_dir, 
            self.n_agents, 
            self.agent_type, 
            self.max_steps, 
            self.starting_energy, 
            self.num_episodes,
            display_info
        )
        
        # Setup agent parameters
        self.agent_params = {
            "survival_cost": energy_cost,
        }
        
        # Create agents
        self.agents = self._create_agents()

    def _create_agents(self):
        """Create agents based on the selected agent type."""
        if self.agent_type == "random":
            return [
                RandomForagingAgent(agent_params=self.agent_params)
                for _ in range(self.n_agents)
            ]
        elif self.agent_type == "qlearning":
            return [
                QLearningForagingAgent(agent_params=self.agent_params)
                for _ in range(self.n_agents)
            ]
        elif self.agent_type == "dqn":
            return [
                DeepQLearningForagingAgent(agent_params=self.agent_params)
                for _ in range(self.n_agents)
            ]
        elif self.agent_type == "curious_dqn":
            return [
                CuriosityDrivenDQNAgent(agent_params=self.agent_params)
                for _ in range(self.n_agents)
            ]
        else:
            self.logger.log(f"Unknown agent type: {self.agent_type}, defaulting to Curious DQN")
            return [
                CuriosityDrivenDQNAgent(agent_params=self.agent_params)
                for _ in range(self.n_agents)
            ]
    
    def train(self):
        """Run training episodes and collect data"""
        # Reset environment to start
        obss, _ = self.env.reset()
        # Only render if render_mode is specified
        if self.render_mode is not None:
            self.env.render()
        
        # Setup agents with the environment
        for i, (player, agent) in enumerate(zip(self.env.unwrapped.players, self.agents)):
            player.set_controller(agent)
            agent.energy = self.starting_energy
        
        # Record start time
        start_time = time.time()
        
        # Run episodes
        for episode in range(self.num_episodes):
            # Initialise episode variables
            ep_returns = np.zeros(self.n_agents)
            ep_intrinsic_returns = np.zeros(self.n_agents)
            ep_length = 0
            ep_deaths = 0
            ep_actions = [[] for _ in range(self.n_agents)]
            
            # Reset environment for new episode
            obss, _ = self.env.reset()
            
            # Respawn all agents for new episode
            for i, (player, agent) in enumerate(zip(self.env.unwrapped.players, self.agents)):
                player.set_controller(agent)
                agent.energy = self.starting_energy  # Reset to initial energy

            if self.render_mode is not None:
                self.env.render()
            
            # Log episode start
            self.logger.log(f"\nEpisode {episode + 1} started")
            self.logger.log(f"Initial energy: {[agent.energy for agent in self.agents]}")
            
            # Track which agents die during this episode
            agents_died_this_episode = [False] * self.n_agents
            
            # Episode loop
            for step in range(self.max_steps):
                # Agents with 0 energy should not act
                # time.sleep(4)
                actions = []
                for i, (agent, obs) in enumerate(zip(self.agents, obss)):
                    if agent.energy > 0:
                        action = agent.step(obs)
                        actions.append(action)
                        ep_actions[i].append(action)
                    else:
                        actions.append(None)
                        # If agent just died this step
                        if not agents_died_this_episode[i] and agent.energy <= 0:
                            agents_died_this_episode[i] = True
                            ep_deaths += 1
                            self.logger.agent_lifetime_stats[i]["deaths"] += 1
                            self.logger.log(f"  Step {step + 1}: Agent {i+1} died (energy depleted)")
                
                # Filter out None actions
                valid_actions = [act for act in actions if act is not None]
                
                # If no agents can act, end the episode early
                if not valid_actions:
                    self.logger.log(f"Episode {episode + 1} ended early at step {step + 1}: All agents are dead.")
                    break
                
                # Take a step in the environment
                obss, rews, done, trunc, info = self.env.step(
                    [act.value for act in valid_actions]
                )
                
                # Collect rewards for agents, including intrinsic rewards for curious DQN agents
                for i, (agent, rew) in enumerate(zip(self.agents, rews)):
                    intrinsic_rew = 0
                    
                    # Safe way to check if this agent tracks intrinsic rewards
                    if self.agent_type == "curious_dqn":
                        # Only try to access intrinsic_rewards if agent is a curious DQN
                        try:
                            # Use direct dict access to avoid recursion issues
                            if hasattr(agent, "__dict__") and "intrinsic_rewards" in agent.__dict__:
                                rewards_list = agent.__dict__["intrinsic_rewards"]
                                if rewards_list and len(rewards_list) > 0:
                                    intrinsic_rew = rewards_list[-1]
                        except Exception as e:
                            self.logger.log(f"Error accessing intrinsic rewards: {e}")
                            intrinsic_rew = 0
                    
                    # Zero out rewards for dead agents
                    if agent.energy == 0:
                        rews[i] = 0  # Dead agents receive no reward
                        intrinsic_rew = 0
                    
                    ep_returns[i] += rews[i]
                    ep_intrinsic_returns[i] += intrinsic_rew
                
                # Detailed step logging
                if self.display_info:
                    self.logger.log(f"  Step {step + 1}: Rewards {rews}, Energy {[agent.energy for agent in self.agents]}, Actions {actions}, Positions {[player.position for player in self.env.unwrapped.players]}")
                
                ep_length += 1
                if self.render_mode is not None:
                    self.env.render()
                        
                # If the environment signals an end, stop
                if done or trunc:
                    self.logger.log(f"Episode {episode + 1} terminated at step {step + 1} with done={done}, truncated={trunc}")
                    break
            
            # Track if episode reached max steps
            reached_max_steps = (ep_length == self.max_steps)
            if reached_max_steps:
                self.logger.episodes_reached_max_steps += 1
                self.logger.log(f"Episode {episode + 1} reached maximum {self.max_steps} steps")
            
            # Update logger with episode data
            self.logger.update_episode_data(
                episode + 1,
                ep_returns,
                ep_intrinsic_returns,
                ep_length,
                ep_deaths,
                reached_max_steps,
                ep_actions
            )
        
        # Calculate and record elapsed time
        elapsed_time = time.time() - start_time
        
        # Generate summary statistics and plots
        self.logger.generate_summary(elapsed_time)

        # Save episode data for aggregates
        self.logger.save_episode_data(self.log_dir)
        
        # Close environment
        self.env.close()
