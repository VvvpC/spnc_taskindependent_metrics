#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-resolution, flexible heatmap plotting functions for beta-gamma data.

Created on 2025-07-22
@author: Claude Code

This module provides flexible heatmap plotting functions that can handle:
- Various data file formats containing beta-gamma heatmap data
- High-resolution output with customizable DPI
- Multiple colormaps and styling options
- Automatic or manual axis labeling
- Support for different matrix dimensions
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import sys


def convert_data_for_compatibility(data_file: Union[str, Path], 
                                  output_file: Union[str, Path] = None) -> None:
    """
    Convert a pickle file to be compatible with older numpy versions.
    
    This function loads data with the current numpy version and re-saves it,
    making it compatible with the current environment.
    
    Parameters:
    -----------
    data_file : str or Path
        Input pickle file path
    output_file : str or Path, optional
        Output file path. If None, will overwrite the input file.
    """
    data_file = Path(data_file)
    if output_file is None:
        output_file = data_file
    else:
        output_file = Path(output_file)
        
    print(f"Converting {data_file} for numpy {np.__version__} compatibility...")
    
    try:
        # Load with current numpy
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            
        # Re-save with current numpy
        with open(output_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(f"Successfully converted and saved to {output_file}")
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        raise


class HeatmapPlotter:
    """
    High-resolution, flexible heatmap plotter for beta-gamma performance data.
    
    Features:
    - High DPI output for publication quality
    - Multiple colormap options
    - Flexible data loading from pickle files
    - Customizable styling and annotations
    - Support for different matrix dimensions
    """
    
    def __init__(self, dpi: int = 300, figsize: Tuple[float, float] = (12, 10)):
        """
        Initialize the heatmap plotter.
        
        Parameters:
        -----------
        dpi : int, default=300
            Resolution for output figures
        figsize : Tuple[float, float], default=(12, 10)
            Figure size in inches (width, height)
        """
        self.dpi = dpi
        self.figsize = figsize
        self.default_cmaps = {
            'cq': 'viridis',
            'mc': 'plasma', 
            'performance': 'RdYlBu_r',
            'diverging': 'RdBu_r'
        }
        
    def load_data(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load heatmap data from pickle file with numpy compatibility handling.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the pickle file containing heatmap data
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing the loaded data
            
        Raises:
        -------
        FileNotFoundError
            If the specified file doesn't exist
        ValueError
            If the file format is not supported
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
            
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        except ModuleNotFoundError as e:
            if "numpy._core" in str(e):
                # Handle numpy version compatibility issue
                error_msg = (
                    f"Numpy compatibility issue: {e}\n"
                    f"The pickle file was created with a newer numpy version. "
                    f"Current numpy version: {np.__version__}\n"
                    f"Try upgrading numpy: pip install numpy>=1.26.0\n"
                    f"Or regenerate the data file with current numpy version."
                )
                raise ValueError(error_msg)
            else:
                raise ValueError(f"Missing module: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load pickle file: {e}")
            
        # Validate data structure
        required_keys = ['beta_prime_range', 'gamma_range']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            warnings.warn(f"Missing expected keys in data: {missing_keys}")
            
        return data
        
    def _prepare_matrices(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Prepare and validate matrices from loaded data.
        
        Parameters:
        -----------
        data : Dict[str, Any]
            Loaded data dictionary
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary containing prepared matrices
        """
        matrices = {}
        
        # Common matrix keys to look for
        matrix_keys = ['cq_matrix', 'mc_matrix', 'performance_matrix', 'kr_matrix', 'gr_matrix']
        
        for key in matrix_keys:
            if key in data:
                matrix = np.array(data[key])
                if matrix.ndim == 2:
                    matrices[key] = matrix
                else:
                    warnings.warn(f"Matrix {key} is not 2D, skipping")
                    
        return matrices
        
    def _get_axis_labels(self, data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract and prepare axis labels from data.
        
        Parameters:
        -----------
        data : Dict[str, Any]
            Loaded data dictionary
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Beta prime range and gamma range arrays
        """
        beta_range = np.array(data.get('beta_prime_range', []))
        gamma_range = np.array(data.get('gamma_range', []))
        
        return beta_range, gamma_range
        
    def plot_single_heatmap(self, 
                           matrix: np.ndarray,
                           beta_range: np.ndarray,
                           gamma_range: np.ndarray,
                           title: str = "Heatmap",
                           xlabel: str = "Beta Prime", 
                           ylabel: str = "Gamma",
                           cmap: str = 'viridis',
                           colorbar_label: str = "Value",
                           ax: Optional[plt.Axes] = None,
                           **kwargs) -> plt.Axes:
        """
        Plot a single heatmap with high resolution and flexible styling.
        
        Parameters:
        -----------
        matrix : np.ndarray
            2D matrix to plot as heatmap
        beta_range : np.ndarray
            Beta prime values for x-axis
        gamma_range : np.ndarray  
            Gamma values for y-axis
        title : str, default="Heatmap"
            Plot title
        xlabel : str, default="Beta Prime"
            X-axis label
        ylabel : str, default="Gamma"
            Y-axis label
        cmap : str, default='viridis'
            Colormap name
        colorbar_label : str, default="Value"
            Colorbar label
        ax : plt.Axes, optional
            Existing axes to plot on
        **kwargs
            Additional arguments for imshow
            
        Returns:
        --------
        plt.Axes
            The plot axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
        # Default imshow parameters for high quality
        imshow_params = {
            'aspect': 'auto',
            'origin': 'lower',
            'interpolation': 'bilinear',
            'cmap': cmap
        }
        imshow_params.update(kwargs)
        
        # Create the heatmap
        im = ax.imshow(matrix, **imshow_params)
        
        # Set title and labels
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        
        # Configure ticks and labels
        if len(beta_range) > 0 and len(gamma_range) > 0:
            # Set tick positions
            n_beta, n_gamma = len(beta_range), len(gamma_range)
            
            # Determine tick spacing for readability
            max_ticks = 10
            beta_step = max(1, n_beta // max_ticks)
            gamma_step = max(1, n_gamma // max_ticks)
            
            beta_tick_indices = range(0, n_beta, beta_step)
            gamma_tick_indices = range(0, n_gamma, gamma_step)
            
            ax.set_xticks(beta_tick_indices)
            ax.set_yticks(gamma_tick_indices)
            
            # Format labels with appropriate precision
            beta_labels = [f'{beta_range[i]:.1f}' for i in beta_tick_indices]
            gamma_labels = [f'{gamma_range[i]:.4f}' for i in gamma_tick_indices]
            
            ax.set_xticklabels(beta_labels, fontsize=10)
            ax.set_yticklabels(gamma_labels, fontsize=10)
            
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(colorbar_label, fontsize=11, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
        
        # Grid for better readability
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        return ax
        
    def plot_comparison_heatmaps(self,
                                data: Dict[str, Any],
                                matrices_to_plot: List[str] = None,
                                titles: Dict[str, str] = None,
                                cmaps: Dict[str, str] = None,
                                save_path: Optional[Union[str, Path]] = None,
                                show_plot: bool = True) -> plt.Figure:
        """
        Plot multiple heatmaps for comparison.
        
        Parameters:
        -----------
        data : Dict[str, Any]
            Loaded data dictionary
        matrices_to_plot : List[str], optional
            List of matrix keys to plot. If None, will plot all available matrices
        titles : Dict[str, str], optional
            Custom titles for each matrix
        cmaps : Dict[str, str], optional
            Custom colormaps for each matrix
        save_path : str or Path, optional
            Path to save the figure
        show_plot : bool, default=True
            Whether to display the plot
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        matrices = self._prepare_matrices(data)
        beta_range, gamma_range = self._get_axis_labels(data)
        
        if not matrices:
            raise ValueError("No valid matrices found in data")
            
        if matrices_to_plot is None:
            matrices_to_plot = list(matrices.keys())
        else:
            matrices_to_plot = [key for key in matrices_to_plot if key in matrices]
            
        n_matrices = len(matrices_to_plot)
        if n_matrices == 0:
            raise ValueError("No valid matrices to plot")
            
        # Determine subplot layout
        if n_matrices == 1:
            n_rows, n_cols = 1, 1
        elif n_matrices == 2:
            n_rows, n_cols = 1, 2
        elif n_matrices <= 4:
            n_rows, n_cols = 2, 2
        elif n_matrices <= 6:
            n_rows, n_cols = 2, 3
        else:
            n_rows = int(np.ceil(np.sqrt(n_matrices)))
            n_cols = int(np.ceil(n_matrices / n_rows))
            
        # Create figure with high DPI
        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize, dpi=self.dpi)
        if n_matrices == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
            
        # Default titles and cmaps
        default_titles = {
            'cq_matrix': 'CQ Performance Heatmap',
            'mc_matrix': 'MC Performance Heatmap',
            'performance_matrix': 'Performance Heatmap',
            'kr_matrix': 'KR Performance Heatmap',
            'gr_matrix': 'GR Performance Heatmap'
        }
        
        if titles is None:
            titles = default_titles
        if cmaps is None:
            cmaps = self.default_cmaps
            
        # Plot each matrix
        for i, matrix_key in enumerate(matrices_to_plot):
            matrix = matrices[matrix_key]
            title = titles.get(matrix_key, matrix_key.replace('_', ' ').title())
            cmap = cmaps.get(matrix_key.split('_')[0], 'viridis')
            colorbar_label = matrix_key.replace('_', ' ').title()
            
            self.plot_single_heatmap(
                matrix=matrix,
                beta_range=beta_range,
                gamma_range=gamma_range,
                title=title,
                cmap=cmap,
                colorbar_label=colorbar_label,
                ax=axes[i]
            )
            
        # Hide unused subplots
        for i in range(n_matrices, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout(pad=2.0)
        
        # Save figure if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Heatmap saved to: {save_path}")
            
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
            
        return fig
        
    def plot_from_file(self, 
                      filepath: Union[str, Path],
                      matrices_to_plot: List[str] = None,
                      save_path: Optional[Union[str, Path]] = None,
                      show_plot: bool = True,
                      **kwargs) -> plt.Figure:
        """
        Convenience function to load data and plot heatmaps in one step.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the data pickle file
        matrices_to_plot : List[str], optional
            List of matrix keys to plot
        save_path : str or Path, optional
            Path to save the figure
        show_plot : bool, default=True
            Whether to display the plot
        **kwargs
            Additional arguments for plot_comparison_heatmaps
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        data = self.load_data(filepath)
        return self.plot_comparison_heatmaps(
            data=data,
            matrices_to_plot=matrices_to_plot,
            save_path=save_path,
            show_plot=show_plot,
            **kwargs
        )


# Convenience functions for quick usage
def quick_heatmap(data_file: Union[str, Path], 
                 output_file: Optional[Union[str, Path]] = None,
                 dpi: int = 300,
                 show_plot: bool = True) -> plt.Figure:
    """
    Quick function to generate heatmaps from a data file.
    
    Parameters:
    -----------
    data_file : str or Path
        Path to the pickle data file
    output_file : str or Path, optional
        Path to save the output figure
    dpi : int, default=300
        Output resolution
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    plt.Figure
        The created figure
    """
    plotter = HeatmapPlotter(dpi=dpi)
    return plotter.plot_from_file(
        filepath=data_file,
        save_path=output_file,
        show_plot=show_plot
    )


def batch_heatmaps(data_directory: Union[str, Path],
                  output_directory: Union[str, Path],
                  file_pattern: str = "*heatmap_data*.pkl",
                  dpi: int = 300) -> List[Path]:
    """
    Generate heatmaps for multiple data files in a directory.
    
    Parameters:
    -----------
    data_directory : str or Path
        Directory containing data files
    output_directory : str or Path
        Directory to save output figures
    file_pattern : str, default="*heatmap_data*.pkl"
        Glob pattern to match data files
    dpi : int, default=300
        Output resolution
        
    Returns:
    --------
    List[Path]
        List of generated figure file paths
    """
    data_dir = Path(data_directory)
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plotter = HeatmapPlotter(dpi=dpi)
    generated_files = []
    
    for data_file in data_dir.glob(file_pattern):
        output_file = output_dir / f"{data_file.stem}_heatmap.png"
        try:
            plotter.plot_from_file(
                filepath=data_file,
                save_path=output_file,
                show_plot=False
            )
            generated_files.append(output_file)
            print(f"Processed: {data_file.name} -> {output_file.name}")
        except Exception as e:
            print(f"Error processing {data_file.name}: {e}")
            
    return generated_files


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate high-resolution heatmaps from beta-gamma data")
    parser.add_argument("data_file", help="Path to the pickle data file")
    parser.add_argument("--output", "-o", help="Output figure path")
    parser.add_argument("--dpi", type=int, default=300, help="Output resolution (default: 300)")
    parser.add_argument("--no-show", action="store_true", help="Don't display the plot")
    parser.add_argument("--matrices", nargs="+", help="Specific matrices to plot")
    parser.add_argument("--convert-only", action="store_true", 
                       help="Only convert data file for numpy compatibility, don't plot")
    parser.add_argument("--convert-output", help="Output path for converted data file")
    
    args = parser.parse_args()
    
    if args.convert_only:
        # Only convert the data file
        convert_data_for_compatibility(args.data_file, args.convert_output)
        print("Data conversion completed!")
    else:
        # Create heatmap
        try:
            quick_heatmap(
                data_file=args.data_file,
                output_file=args.output,
                dpi=args.dpi,
                show_plot=not args.no_show
            )
            print("Heatmap generation completed!")
        except ValueError as e:
            if "numpy._core" in str(e):
                print("\nNumpy compatibility issue detected!")
                print("Try converting the data file first:")
                print(f"python {sys.argv[0]} {args.data_file} --convert-only")
                print("Then try plotting again with the converted file.")
            else:
                raise