import numpy as np
import matplotlib.pyplot as plt
import os

class ExplorationTracker:
    """A class for tracking and visualizing agent exploration patterns"""
    
    def __init__(self, grid_size=(3, 3)):
        """
        Initialise the exploration tracker
        
        Parameters:
        -----------
        grid_size : tuple
            Size of the environment grid (rows, cols)
        """
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        
        # Initialise visitation maps for different agent types
        self.visitation_maps = {}
        
        # Track positions by episode
        self.episode_positions = []
        self.current_episode = 0
        
    def reset_episode(self):
        """Reset tracking for a new episode"""
        self.episode_positions = []
        self.current_episode += 1
        
    def track_position(self, position, agent_type):
        """
        Track a position visited by an agent
        
        Parameters:
        -----------
        position : tuple
            (row, col) position of the agent
        agent_type : str
            Type of agent (e.g., 'curious_dqn', 'dqn', etc.)
        """
        # Initialise visitation map for this agent type if it doesn't exist
        if agent_type not in self.visitation_maps:
            self.visitation_maps[agent_type] = np.zeros(self.grid_size)
        
        # Record the position
        row, col = position
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.visitation_maps[agent_type][row, col] += 1
            self.episode_positions.append((position, agent_type))
    
    def get_heatmap(self, agent_type):
        """
        Get the visitation heatmap for a specific agent type
        
        Parameters:
        -----------
        agent_type : str
            Type of agent to get heatmap for
            
        Returns:
        --------
        numpy.ndarray
            2D array representing visit counts
        """
        if agent_type not in self.visitation_maps:
            return np.zeros(self.grid_size)
        return self.visitation_maps[agent_type]
    
    def get_total_unique_states_visited(self, agent_type):
        """
        Calculate how many unique states have been visited
        
        Parameters:
        -----------
        agent_type : str
            Type of agent to calculate for
            
        Returns:
        --------
        int
            Number of unique states visited
        """
        if agent_type not in self.visitation_maps:
            return 0
        return np.sum(self.visitation_maps[agent_type] > 0)
    
    def get_exploration_percentage(self, agent_type):
        """
        Calculate the percentage of the grid that has been explored
        
        Parameters:
        -----------
        agent_type : str
            Type of agent to calculate for
            
        Returns:
        --------
        float
            Percentage of states explored (0-100)
        """
        total_states = self.rows * self.cols
        states_visited = self.get_total_unique_states_visited(agent_type)
        return (states_visited / total_states) * 100
    
    def get_exploration_efficiency(self, agent_type, step_count):
        """
        Calculate exploration efficiency (states/step)
        
        Parameters:
        -----------
        agent_type : str
            Type of agent to calculate for
        step_count : int
            Total number of steps taken
            
        Returns:
        --------
        float
            States explored per step
        """
        if step_count == 0:
            return 0
        states_visited = self.get_total_unique_states_visited(agent_type)
        return states_visited / step_count
    
    def plot_heatmap(self, agent_type, output_dir, episode_num=None):
        """
        Generate and save an exploration heatmap visualization
        
        Parameters:
        -----------
        agent_type : str
            Type of agent to visualize
        output_dir : str
            Directory to save the visualization
        episode_num : int, optional
            Current episode number for the filename
        """
        if agent_type not in self.visitation_maps:
            return
        
        heatmap = self.visitation_maps[agent_type]
        
        plt.figure(figsize=(8, 6))
        
        # Create a masked array to hide unvisited states
        masked_data = np.ma.masked_where(heatmap == 0, heatmap)
        
        # Set up the colormap
        cmap = plt.cm.viridis
        cmap.set_bad('white', 1.0)
        
        # Plot the heatmap
        img = plt.imshow(heatmap, interpolation='nearest', cmap=cmap)
        plt.colorbar(img, label='Visit Count')
        
        # Add grid lines
        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, self.cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.rows, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        
        # Add counts as text
        for i in range(self.rows):
            for j in range(self.cols):
                if heatmap[i, j] > 0:
                    plt.text(j, i, int(heatmap[i, j]), 
                             ha="center", va="center", 
                             color="white" if heatmap[i, j] > np.max(heatmap) / 2 else "black",
                             fontweight='bold')
        
        # Add title and labels
        episode_text = f" (Episode {episode_num})" if episode_num is not None else ""
        plt.title(f'{agent_type} Exploration Heatmap{episode_text}')
        plt.xlabel('Column')
        plt.ylabel('Row')
        
        # Save the figure
        filename = f"{agent_type}_exploration"
        if episode_num is not None:
            filename += f"_ep{episode_num}"
        
        plt.savefig(f"{output_dir}/{filename}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_comparison_heatmaps(self, agent_types, output_dir, episode_num=None):
        """
        Generate side-by-side heatmaps for multiple agent types
        
        Parameters:
        -----------
        agent_types : list
            List of agent types to compare
        output_dir : str
            Directory to save the visualization
        episode_num : int, optional
            Current episode number for the filename
        """
        if not agent_types:
            return
        
        # Only include agent types with data
        agent_types = [at for at in agent_types if at in self.visitation_maps]
        if not agent_types:
            return
        
        # Set up the figure
        n_agents = len(agent_types)
        fig, axes = plt.subplots(1, n_agents, figsize=(5 * n_agents, 5))
        if n_agents == 1:
            axes = [axes]  # Ensure axes is always a list
        
        # Determine the max value for consistent color scaling
        max_val = max(np.max(self.visitation_maps[at]) for at in agent_types)
        
        # Plot each heatmap
        for i, agent_type in enumerate(agent_types):
            heatmap = self.visitation_maps[agent_type]
            
            # Set up the colormap
            cmap = plt.cm.viridis
            cmap.set_bad('white', 1.0)
            
            # Plot the heatmap
            img = axes[i].imshow(heatmap, interpolation='nearest', cmap=cmap, vmin=0, vmax=max_val)
            
            # Add grid lines
            axes[i].set_xticks(np.arange(-0.5, self.cols, 1), minor=True)
            axes[i].set_yticks(np.arange(-0.5, self.rows, 1), minor=True)
            axes[i].grid(which='minor', color='black', linestyle='-', linewidth=1)
            
            # Add counts as text
            for r in range(self.rows):
                for c in range(self.cols):
                    if heatmap[r, c] > 0:
                        axes[i].text(c, r, int(heatmap[r, c]), 
                                   ha="center", va="center", 
                                   color="white" if heatmap[r, c] > max_val / 2 else "black",
                                   fontweight='bold')
            
            # Add title and labels
            axes[i].set_title(f'{agent_type}')
            axes[i].set_xlabel('Column')
            axes[i].set_ylabel('Row')
            
            # Add exploration percentage
            exp_pct = self.get_exploration_percentage(agent_type)
            axes[i].text(0.5, -0.1, f"Explored: {exp_pct:.1f}%", 
                       ha="center", transform=axes[i].transAxes)
        
        # Add a colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(img, cax=cbar_ax)
        cbar.set_label('Visit Count')
        
        # Add overall title
        episode_text = f" (Episode {episode_num})" if episode_num is not None else ""
        fig.suptitle(f'Exploration Pattern Comparison{episode_text}', fontsize=16)
        
        # Adjust spacing
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        # Save the figure
        filename = "exploration_comparison"
        if episode_num is not None:
            filename += f"_ep{episode_num}"
        
        plt.savefig(f"{output_dir}/{filename}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_exploration_metrics(self, agent_types, steps_by_agent, output_dir):
        """
        Plot exploration metrics comparing different agent types
        
        Parameters:
        -----------
        agent_types : list
            List of agent types to compare
        steps_by_agent : dict
            Dictionary mapping agent types to total steps taken
        output_dir : str
            Directory to save the visualization
        """
        # Only include agent types with data
        agent_types = [at for at in agent_types if at in self.visitation_maps]
        if not agent_types:
            return
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Exploration Coverage
        coverage_data = [self.get_exploration_percentage(at) for at in agent_types]
        ax1.bar(agent_types, coverage_data, color=['#8884d8', '#82ca9d', '#ffc658'][:len(agent_types)])
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('Grid Coverage (%)')
        ax1.set_title('Exploration Coverage')
        
        # Add value labels
        for i, v in enumerate(coverage_data):
            ax1.text(i, v + 2, f'{v:.1f}%', ha='center')
        
        # Plot 2: Exploration Efficiency
        efficiency_data = [self.get_exploration_efficiency(at, steps_by_agent.get(at, 1)) 
                          for at in agent_types]
        
        ax2.bar(agent_types, efficiency_data, color=['#8884d8', '#82ca9d', '#ffc658'][:len(agent_types)])
        ax2.set_ylabel('States Explored per Step')
        ax2.set_title('Exploration Efficiency')
        
        # Add value labels
        for i, v in enumerate(efficiency_data):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{output_dir}/exploration_metrics.png", dpi=150, bbox_inches='tight')
        plt.close()