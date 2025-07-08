#!/usr/bin/env python3
"""
Simple script to plot results from TI46_parameter_sweep_20250708_154405.pkl
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_and_plot_results(filename):
    """
    Load pkl file and plot results
    """
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Loaded {len(results['accuracies'])} experimental results")
    print(f"Parameters tested: {results['param_names']}")
    
    # Extract data
    param_names = results['param_names']
    combinations = results['combinations']
    accuracies = results['accuracies']
    configurations = results['configurations']
    
    # Filter out None values
    valid_indices = [i for i, acc in enumerate(accuracies) if acc is not None]
    valid_combinations = [combinations[i] for i in valid_indices]
    valid_accuracies = [accuracies[i] for i in valid_indices]
    valid_configurations = [configurations[i] for i in valid_indices]
    
    if not valid_accuracies:
        print("No valid accuracy results")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('TI46 Parameter Sweep Results', fontsize=16)
    
    # Plot 1: Parameter vs Accuracy
    if len(param_names) > 0:
        param_name = param_names[0]
        param_values = [combo[0] for combo in valid_combinations]
        
        scatter = axes[0, 0].scatter(param_values, valid_accuracies, 
                                   c=valid_accuracies, cmap='viridis', 
                                   alpha=0.7, s=50)
        axes[0, 0].set_xlabel(param_name)
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title(f'{param_name} vs Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 0], label='Accuracy')
    
    # Plot 2: Accuracy distribution
    axes[0, 1].hist(valid_accuracies, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.max(valid_accuracies), color='red', linestyle='--', 
                       label=f'Best: {np.max(valid_accuracies):.4f}')
    axes[0, 1].set_xlabel('Accuracy')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Accuracy Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Accuracy vs experiment order
    axes[1, 0].plot(range(len(valid_accuracies)), valid_accuracies, 'bo-', markersize=4)
    axes[1, 0].set_xlabel('Experiment ID')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Accuracy vs Experiment Order')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Statistics table
    axes[1, 1].axis('off')
    stats_text = f"""
    Statistics Summary:
    
    Total experiments: {len(valid_accuracies)}
    Best accuracy: {np.max(valid_accuracies):.6f}
    Worst accuracy: {np.min(valid_accuracies):.6f}
    Mean accuracy: {np.mean(valid_accuracies):.6f}
    Std accuracy: {np.std(valid_accuracies):.6f}
    
    Best configuration:
    """
    
    # Find best configuration
    best_idx = valid_accuracies.index(np.max(valid_accuracies))
    best_config = valid_configurations[best_idx]
    for param, value in best_config.items():
        stats_text += f"    {param}: {value:.6f}\n"
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                    verticalalignment='top', fontsize=10, fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print(f"\n=== Detailed Statistics ===")
    print(f"Total experiments: {len(valid_accuracies)}")
    print(f"Best accuracy: {np.max(valid_accuracies):.6f}")
    print(f"Mean accuracy: {np.mean(valid_accuracies):.6f}")
    print(f"Std accuracy: {np.std(valid_accuracies):.6f}")
    
    print(f"\nBest parameter configuration:")
    for param, value in best_config.items():
        print(f"  {param}: {value}")
    print(f"  Accuracy: {np.max(valid_accuracies):.6f}")

if __name__ == "__main__":
    load_and_plot_results("TI46_parameter_sweep_20250708_013126.pkl")