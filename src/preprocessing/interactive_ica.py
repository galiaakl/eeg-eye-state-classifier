# src/preprocessing/interactive_ica.py
"""
Interactive ICA component removal - matches original code approach
"""

import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def interactive_component_removal(ica, components: np.ndarray, component_info: Dict[int, Dict[str, Any]], 
                                sorted_indices: List[int], channel_names: List[str]) -> tuple:
    """
    Interactive component removal matching original code approach.
    
    Shows component analysis, provides suggestions, and gets user input
    for which components to remove.
    
    Parameters:
    -----------
    ica : FastICA object
        Fitted ICA object
    components : np.ndarray
        ICA components array
    component_info : Dict[int, Dict[str, Any]]
        Component analysis information
    sorted_indices : List[int]
        Component indices sorted by discriminative score
    channel_names : List[str]
        Channel names
        
    Returns:
    --------
    tuple : (X_reconstructed, components_to_remove)
        - X_reconstructed: Reconstructed EEG data
        - components_to_remove: List of component indices that were removed
    """
    from .ica_analysis import suggest_components_to_remove, reconstruct_signals
    
    print("\n" + "="*60)
    print("ICA COMPONENT SELECTION FOR RECONSTRUCTION")
    print("="*60)
    
    print("\nComponent Analysis Summary:")
    print("-" * 40)
    
    # Show all components with their scores and types
    for rank, comp_idx in enumerate(sorted_indices):
        info = component_info[comp_idx]
        print(f"Rank {rank+1:2d}: Component {comp_idx+1:2d} - Score: {info['discriminative_score']:6.3f} - "
              f"Type: {info['component_type']:15s} - Region: {info['primary_region']}")
    
    print("\n" + "-" * 40)
    print("Component Details:")
    print("-" * 40)
    
    # Show detailed info for each component
    for comp_idx in sorted_indices:
        info = component_info[comp_idx]
        print(f"\nComponent {comp_idx+1}:")
        print(f"  Score: {info['discriminative_score']:.4f}")
        print(f"  Type: {info['component_type']}")
        print(f"  Primary Region: {info['primary_region']} ({info['region_score']*100:.1f}%)")
        print(f"  Max Weight Channel: {info['max_weight_channel']}")
        print(f"  Alpha Ratio (closed/open): {info['power_ratios']['alpha_ratio']:.3f}")
        print(f"  Beta Ratio (closed/open): {info['power_ratios']['beta_ratio']:.3f}")
        print(f"  Std Ratio (closed/open): {info['std_ratio']:.3f}")
    
    # Get automatic suggestions
    print("\n" + "="*60)
    print("AUTOMATIC COMPONENT REMOVAL SUGGESTIONS")
    print("="*60)
    
    suggested_removals = suggest_components_to_remove(component_info, threshold_score=1.0)
    
    if suggested_removals:
        print(f"\nSuggested components to remove: {[i+1 for i in suggested_removals]}")
        print("\nReasons for removal:")
        
        for comp_idx in suggested_removals:
            info = component_info[comp_idx]
            reasons = []
            
            if "artifact" in info['component_type'].lower():
                reasons.append("identified as artifact")
            if info['discriminative_score'] < 1.0:
                reasons.append(f"low discriminative score ({info['discriminative_score']:.3f})")
            if info['primary_region'] == 'posterior' and info['power_ratios'].get('alpha_ratio', 1) < 0.7:
                reasons.append("incorrect alpha pattern for posterior region")
            if info['primary_region'] == 'frontal' and info['power_ratios'].get('beta_ratio', 1) < 0.7:
                reasons.append("incorrect beta pattern for frontal region")
            
            print(f"  Component {comp_idx+1}: {', '.join(reasons)}")
        
        print(f"\nComponents recommended to KEEP (high scores):")
        keep_components = [idx for idx in sorted_indices if idx not in suggested_removals]
        for comp_idx in keep_components[:5]:  # Show top 5 to keep
            info = component_info[comp_idx]
            print(f"  Component {comp_idx+1}: Score {info['discriminative_score']:.3f} - {info['component_type']}")
    else:
        print("\nNo components suggested for removal - all components appear to be of good quality!")
        print("All components have reasonable discriminative scores and patterns.")
    
    # Interactive component selection
    print("\n" + "="*60)
    print("INTERACTIVE COMPONENT SELECTION")
    print("="*60)
    
    print("\nBased on the analysis above, decide which components to remove.")
    print("Components with lower discriminative scores and identified as artifacts")
    print("are good candidates for removal.")
    print("\nOptions:")
    print("  - Enter component numbers (e.g., '7,13' to remove components 7 and 13)")
    print("  - Enter 'auto' to use automatic suggestions")
    print("  - Enter 'none' or press Enter to keep all components")
    
    if suggested_removals:
        print(f"\nAutomatic suggestion: Remove components {[i+1 for i in suggested_removals]}")
    
    # Get user input with error handling
    while True:
        try:
            remove_input = input("\nEnter your choice: ").strip()
            
            if remove_input.lower() == 'auto':
                if suggested_removals:
                    components_to_remove = suggested_removals
                    print(f"✓ Using automatic suggestions: Remove components {[i+1 for i in components_to_remove]}")
                    break
                else:
                    print("No automatic suggestions available. Try 'none' or specify components.")
                    continue
                    
            elif remove_input.lower() == 'none' or remove_input == '':
                components_to_remove = []
                print("✓ No components will be removed - keeping all components")
                break
                
            else:
                # Parse user input
                component_nums = [int(x.strip()) for x in remove_input.split(',') if x.strip()]
                components_to_remove = [num - 1 for num in component_nums]  # Convert to 0-based indexing
                
                # Validate component numbers
                invalid_components = [num for num in component_nums if num < 1 or num > len(component_info)]
                if invalid_components:
                    print(f"Error: Invalid component numbers: {invalid_components}")
                    print(f"Valid range is 1-{len(component_info)}")
                    continue
                
                print(f"✓ User selected: Remove components {component_nums}")
                break
                
        except ValueError:
            print("Error: Please enter component numbers separated by commas, 'auto', or 'none'")
            print("Example: '7,13' or 'auto' or 'none'")
            continue
        except KeyboardInterrupt:
            print("\nOperation cancelled. Using automatic suggestions if available.")
            components_to_remove = suggested_removals if suggested_removals else []
            break
    
    # Reconstruct signal
    print("\n" + "="*60)
    print("SIGNAL RECONSTRUCTION")
    print("="*60)
    
    print(f"\nReconstructing signal...")
    if components_to_remove:
        print(f"Removing {len(components_to_remove)} components: {[i+1 for i in components_to_remove]}")
        
        # Show what's being removed
        for comp_idx in components_to_remove:
            info = component_info[comp_idx]
            print(f"  - Component {comp_idx+1}: {info['component_type']} (Score: {info['discriminative_score']:.3f})")
    else:
        print("Keeping all components (no removal)")
    
    X_reconstructed = reconstruct_signals(ica, components, components_to_remove)
    
    print("✓ Signal reconstruction completed")
    
    return X_reconstructed, components_to_remove
