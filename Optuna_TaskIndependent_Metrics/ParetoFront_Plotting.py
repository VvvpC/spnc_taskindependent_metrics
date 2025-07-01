"""
Pareto Front Plotting Module

This module provides functionality to visualize Pareto fronts and related data points
from optimization trials. It reads data from CSV files and creates comprehensive
plots showing the Pareto front and dominated points.

Author: Generated for SPNC Optuna Heterogeneous project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from scipy.spatial.distance import cdist
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ParetoFrontPlotter:
    """
    A class for plotting Pareto fronts and analyzing dominated solutions.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the ParetoFrontPlotter.
        
        Args:
            data_dir (str): Directory containing the CSV data files. If None, will auto-detect.
        """
        if data_dir is None:
            # Try to find the data directory automatically
            self.data_dir = self._find_data_directory()
        else:
            self.data_dir = Path(data_dir)
        self.pareto_front_df = None
        self.all_trials_df = None
    
    def _find_data_directory(self) -> Path:
        """
        Automatically find the data directory by searching common locations.
        
        Returns:
            Path: Path to the data directory
            
        Raises:
            FileNotFoundError: If data directory cannot be found
        """
        # Common possible locations for the data directory
        possible_paths = [
            Path("ParetoFront_CQandMC/data"),           # From project root
            Path("../ParetoFront_CQandMC/data"),        # From subdirectory
            Path("../../ParetoFront_CQandMC/data"),     # From deeper subdirectory
            Path(__file__).parent.parent / "ParetoFront_CQandMC/data",  # Relative to this file
        ]
        
        for path in possible_paths:
            if path.exists() and (path / "pareto_front.csv").exists():
                print(f"Found data directory at: {path.absolute()}")
                return path
                
        # If none found, provide helpful error message
        current_dir = Path.cwd()
        error_msg = f"""
Data directory not found. Searched in:
{chr(10).join(f'  - {p.absolute()}' for p in possible_paths)}

Current working directory: {current_dir}

Please ensure the data files exist at one of these locations:
  - pareto_front.csv
  - all_trials.csv

Or specify the data directory explicitly:
  plotter = ParetoFrontPlotter(data_dir="path/to/your/data")
"""
        raise FileNotFoundError(error_msg)
        
    def load_data(self) -> None:
        """Load Pareto front and all trials data from CSV files."""
        try:
            pareto_path = self.data_dir / "pareto_front.csv"
            all_trials_path = self.data_dir / "all_trials.csv"
            
            if not pareto_path.exists():
                raise FileNotFoundError(f"Pareto front data not found at {pareto_path}")
            if not all_trials_path.exists():
                raise FileNotFoundError(f"All trials data not found at {all_trials_path}")
                
            self.pareto_front_df = pd.read_csv(pareto_path)
            self.all_trials_df = pd.read_csv(all_trials_path)
            
            print(f"Loaded {len(self.pareto_front_df)} Pareto front points")
            print(f"Loaded {len(self.all_trials_df)} total trial points")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def find_near_pareto_points(self, distance_threshold: float = 0.5, 
                               max_points: int = 50) -> pd.DataFrame:
        """
        Find points from all trials that are close to the Pareto front.
        
        Args:
            distance_threshold (float): Maximum distance to Pareto front
            max_points (int): Maximum number of near-Pareto points to return
            
        Returns:
            pd.DataFrame: Points near the Pareto front
        """
        if self.pareto_front_df is None or self.all_trials_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Extract CQ and MC values for distance calculation
        pareto_points = self.pareto_front_df[['CQ', 'MC']].values
        all_points = self.all_trials_df[['CQ', 'MC']].values
        
        # Normalize the data for distance calculation
        cq_range = all_points[:, 0].max() - all_points[:, 0].min()
        mc_range = all_points[:, 1].max() - all_points[:, 1].min()
        
        pareto_normalized = pareto_points.copy()
        pareto_normalized[:, 0] = (pareto_points[:, 0] - all_points[:, 0].min()) / cq_range
        pareto_normalized[:, 1] = (pareto_points[:, 1] - all_points[:, 1].min()) / mc_range
        
        all_normalized = all_points.copy()
        all_normalized[:, 0] = (all_points[:, 0] - all_points[:, 0].min()) / cq_range
        all_normalized[:, 1] = (all_points[:, 1] - all_points[:, 1].min()) / mc_range
        
        # Calculate minimum distance from each point to the Pareto front
        distances = cdist(all_normalized, pareto_normalized, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        
        # Find points within distance threshold, excluding Pareto front points
        pareto_numbers = set(self.pareto_front_df['number'].values)
        near_indices = []
        
        for i, (distance, trial_number) in enumerate(zip(min_distances, self.all_trials_df['number'])):
            if distance <= distance_threshold and trial_number not in pareto_numbers:
                near_indices.append(i)
        
        # Sort by distance and limit to max_points
        near_indices = sorted(near_indices, key=lambda i: min_distances[i])[:max_points]
        
        return self.all_trials_df.iloc[near_indices].copy()
    
    def plot_pareto_front_2d(self, figsize: Tuple[int, int] = (12, 8),
                            distance_threshold: float = 0.5,
                            max_near_points: int = 50,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a 2D plot of the Pareto front with CQ vs MC.
        
        Args:
            figsize (tuple): Figure size (width, height)
            distance_threshold (float): Distance threshold for near-Pareto points
            max_near_points (int): Maximum number of near-Pareto points to show
            save_path (str, optional): Path to save the figure
            
        Returns:
            plt.Figure: The created figure
        """
        if self.pareto_front_df is None or self.all_trials_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Find near-Pareto points
        near_pareto_df = self.find_near_pareto_points(distance_threshold, max_near_points)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot all trials as background points
        ax.scatter(self.all_trials_df['CQ'], self.all_trials_df['MC'], 
                  alpha=0.3, s=20, c='lightgray', label=f'All Trials (n={len(self.all_trials_df)})')
        
        # Plot near-Pareto points
        if len(near_pareto_df) > 0:
            ax.scatter(near_pareto_df['CQ'], near_pareto_df['MC'], 
                      alpha=0.7, s=40, c='orange', label=f'Near Pareto (n={len(near_pareto_df)})')
        
        # Plot Pareto front
        pareto_sorted = self.pareto_front_df.sort_values('CQ')
        ax.plot(pareto_sorted['CQ'], pareto_sorted['MC'], 
               'r-', linewidth=2, alpha=0.7, label='Pareto Front Connection')
        ax.scatter(self.pareto_front_df['CQ'], self.pareto_front_df['MC'], 
                  s=80, c='red', edgecolors='darkred', linewidth=1, 
                  label=f'Pareto Front (n={len(self.pareto_front_df)})', zorder=5)
        
        # Annotate Pareto front points with their trial numbers
        for _, row in self.pareto_front_df.iterrows():
            ax.annotate(f'{int(row["number"])}', 
                       (row['CQ'], row['MC']), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Computational Quality (CQ)', fontsize=12)
        ax.set_ylabel('Memory Capacity (MC)', fontsize=12)
        ax.set_title('Pareto Front: Computational Quality vs Memory Capacity', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Add distance threshold info to the plot
        ax.text(0.02, 0.98, f'Distance threshold: {distance_threshold}', 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_parameter_distribution(self, parameters: List[str] = None,
                                   figsize: Tuple[int, int] = (15, 10),
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot parameter distributions for Pareto front vs all trials.
        
        Args:
            parameters (list): List of parameters to plot. If None, plots all numeric parameters
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the figure
            
        Returns:
            plt.Figure: The created figure
        """
        if self.pareto_front_df is None or self.all_trials_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if parameters is None:
            parameters = ['gamma', 'theta', 'm0', 'h', 'beta_prime', 'Nvirt']
        
        n_params = len(parameters)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, param in enumerate(parameters):
            ax = axes[i]
            
            # Plot histograms
            ax.hist(self.all_trials_df[param], bins=30, alpha=0.6, 
                   color='lightblue', label='All Trials', density=True)
            ax.hist(self.pareto_front_df[param], bins=15, alpha=0.8, 
                   color='red', label='Pareto Front', density=True)
            
            ax.set_xlabel(param)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {param}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(parameters), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Parameter Distributions: Pareto Front vs All Trials', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter distribution plot saved to {save_path}")
        
        return fig
    
    def plot_3d_pareto(self, third_param: str = 'gamma',
                      figsize: Tuple[int, int] = (12, 10),
                      distance_threshold: float = 0.5,
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a 3D plot of CQ, MC, and a third parameter.
        
        Args:
            third_param (str): Name of the third parameter to plot
            figsize (tuple): Figure size
            distance_threshold (float): Distance threshold for near-Pareto points
            save_path (str, optional): Path to save the figure
            
        Returns:
            plt.Figure: The created figure
        """
        if self.pareto_front_df is None or self.all_trials_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Find near-Pareto points
        near_pareto_df = self.find_near_pareto_points(distance_threshold, 50)
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all trials
        ax.scatter(self.all_trials_df['CQ'], self.all_trials_df['MC'], 
                  self.all_trials_df[third_param],
                  alpha=0.3, s=20, c='lightgray', label='All Trials')
        
        # Plot near-Pareto points
        if len(near_pareto_df) > 0:
            ax.scatter(near_pareto_df['CQ'], near_pareto_df['MC'], 
                      near_pareto_df[third_param],
                      alpha=0.7, s=40, c='orange', label='Near Pareto')
        
        # Plot Pareto front
        ax.scatter(self.pareto_front_df['CQ'], self.pareto_front_df['MC'], 
                  self.pareto_front_df[third_param],
                  s=80, c='red', edgecolors='darkred', linewidth=1, 
                  label='Pareto Front', zorder=5)
        
        ax.set_xlabel('Computational Quality (CQ)')
        ax.set_ylabel('Memory Capacity (MC)')
        ax.set_zlabel(third_param)
        ax.set_title(f'3D Pareto Analysis: CQ vs MC vs {third_param}')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D plot saved to {save_path}")
        
        return fig
    
    def create_summary_report(self) -> None:
        """Print a summary report of the Pareto front analysis."""
        if self.pareto_front_df is None or self.all_trials_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("="*60)
        print("PARETO FRONT ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nTotal trials: {len(self.all_trials_df)}")
        print(f"Pareto front points: {len(self.pareto_front_df)}")
        print(f"Pareto front efficiency: {len(self.pareto_front_df)/len(self.all_trials_df)*100:.1f}%")
        
        print("\nPareto Front Statistics:")
        print("-" * 30)
        for metric in ['CQ', 'MC']:
            print(f"{metric:3s}: min={self.pareto_front_df[metric].min():6.1f}, "
                  f"max={self.pareto_front_df[metric].max():6.1f}, "
                  f"mean={self.pareto_front_df[metric].mean():6.1f}")
        
        print("\nAll Trials Statistics:")
        print("-" * 30)
        for metric in ['CQ', 'MC']:
            print(f"{metric:3s}: min={self.all_trials_df[metric].min():6.1f}, "
                  f"max={self.all_trials_df[metric].max():6.1f}, "
                  f"mean={self.all_trials_df[metric].mean():6.1f}")
        
        print("\nTop 5 Pareto Front Points (by CQ):")
        print("-" * 40)
        top_cq = self.pareto_front_df.nlargest(5, 'CQ')[['number', 'CQ', 'MC']]
        for _, row in top_cq.iterrows():
            print(f"Trial {int(row['number']):3d}: CQ={row['CQ']:6.1f}, MC={row['MC']:6.1f}")
        
        print("\nTop 5 Pareto Front Points (by MC):")
        print("-" * 40)
        top_mc = self.pareto_front_df.nlargest(5, 'MC')[['number', 'CQ', 'MC']]
        for _, row in top_mc.iterrows():
            print(f"Trial {int(row['number']):3d}: CQ={row['CQ']:6.1f}, MC={row['MC']:6.1f}")


def main():
    """Main function demonstrating the usage of ParetoFrontPlotter."""
    # Initialize plotter
    plotter = ParetoFrontPlotter()
    
    # Load data
    plotter.load_data()
    
    # Create summary report
    plotter.create_summary_report()
    
    # Create plots
    print("\nGenerating plots...")
    
    # 2D Pareto front plot
    fig1 = plotter.plot_pareto_front_2d(
        distance_threshold=0.3,
        max_near_points=30,
        save_path="pareto_front_2d.png"
    )
    
    # Parameter distribution plot
    fig2 = plotter.plot_parameter_distribution(
        save_path="parameter_distributions.png"
    )
    
    # 3D plot with gamma
    fig3 = plotter.plot_3d_pareto(
        third_param='gamma',
        save_path="pareto_front_3d_gamma.png"
    )
    
    # Show plots
    plt.show()
    
    print("\nPlotting complete! Check the generated PNG files.")


if __name__ == "__main__":
    main() 