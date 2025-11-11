# Greedy Modularity Optimization Approach
# This method uses a greedy algorithm to maximize the modularity score
# by iteratively splitting communities and tracking the best partition.
#
# Behavior: More aggressive and explores the partition space thoroughly
# Result: Detected 4 communities with modularity score of 0.4190

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# Load the Karate Club graph
G = nx.karate_club_graph()
print(G)

# Visualize the original graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300)
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=8)
plt.title("Karate Club Graph - Original")
plt.axis('off')
plt.tight_layout()
plt.show()

print(f"Nodes in the graph: {G.number_of_nodes()}")
print(f"Edges in the graph: {G.number_of_edges()}")
print(f"Nodes: {list(G.nodes())}")


# === GREEDY MODULARITY APPROACH ===

def find_split_greedy(graph, community, modularity_threshold=0.3):
    """
    Find if a community can be split using greedy modularity optimization.
    Uses NetworkX's greedy_modularity_communities algorithm.
    
    Args:
        graph: The original full graph
        community: List of nodes in the community to split
        modularity_threshold: Minimum modularity threshold
    
    Returns:
        Tuple of (comm1, comm2) if split is valid, None otherwise
    """
    if len(community) < 3:
        return None
    
    subgraph = graph.subgraph(community).copy()
    
    from networkx.algorithms import community as community_module
    detected_communities = list(community_module.greedy_modularity_communities(subgraph))
    
    # Return communities only if more than 1 community is detected
    if len(detected_communities) >= 2:
        # Return binary split (top 2 communities)
        sorted_comms = sorted(detected_communities, key=len, reverse=True)
        return (list(sorted_comms[0]), list(sorted_comms[1]))
    
    return None


def visualise_communities(graph, pos, communities, title):
    """
    Visualize communities with different colors.
    
    Args:
        graph: The network graph
        pos: Layout positions
        communities: List of communities (each community is a list of nodes)
        title: Title for the visualization
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create color map for communities
    colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
    node_colors = {}
    
    for comm_idx, community in enumerate(communities):
        for node in community:
            node_colors[node] = colors[comm_idx]
    
    node_color_list = [node_colors[node] for node in graph.nodes()]
    
    nx.draw_networkx_nodes(graph, pos, node_color=node_color_list, node_size=300, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def calculate_global_modularity(graph, communities):
    """
    Calculate the overall modularity of the current community partition.
    
    Args:
        graph: The full graph
        communities: List of communities
    
    Returns:
        Modularity score
    """
    from networkx.algorithms import community as community_module
    return community_module.modularity(graph, communities)


# === Main Iterative Analysis Loop with Greedy Modularity ===

print("\n" + "="*70)
print("GREEDY MODULARITY OPTIMIZATION APPROACH")
print("="*70)

# Initial setup
communities = [list(G.nodes())]  # Start with one community: the whole graph
iteration = 0
modularity_history = {}
best_modularity = 0.0
best_iteration = 0
best_communities = None

# Compute initial layout
pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)

print("\n--- Starting Greedy Modularity Optimization ---")

while True:
    # --- Visualize and Record Metrics for the current state ---
    title = f"Iteration {iteration}: {len(communities)} communities"
    print(f"\n--- {title} ---")
    visualise_communities(G, pos, communities, title)
    
    # Calculate global modularity
    global_modularity = calculate_global_modularity(G, communities)
    modularity_history[iteration] = global_modularity
    print(f"Global Modularity Score: {global_modularity:.4f}")
    
    # Track best modularity and communities
    if global_modularity > best_modularity:
        best_modularity = global_modularity
        best_iteration = iteration
        best_communities = [comm.copy() for comm in communities]
        print(f"*** NEW BEST MODULARITY: {best_modularity:.4f} ***")
    elif iteration > 0 and global_modularity < modularity_history[iteration - 1]:
        print(f"WARNING: Modularity decreased from {modularity_history[iteration - 1]:.4f} to {global_modularity:.4f}")
    
    # --- Find and perform the next split ---
    
    # Find the largest community that is still splittable
    community_to_split = None
    split_result = None
    
    # Sort communities by size to try splitting the largest ones first
    sorted_communities = sorted(communities, key=len, reverse=True)
    
    for community in sorted_communities:
        result = find_split_greedy(G, community, modularity_threshold=0.3)
        if result is not None:
            # We found a community we can split
            community_to_split = community
            split_result = result
            break  # Stop searching and perform the split
    
    # --- Check if we should stop the loop ---
    # Stop if modularity is decreasing significantly from the best
    if iteration > best_iteration and global_modularity < best_modularity * 0.95:
        print("\n" + "="*70)
        print("Modularity decreased significantly. Stopping at best partition.")
        print("="*70)
        communities = best_communities
        break
    
    if community_to_split is None:
        # If we went through all communities and none could be split, we're done.
        print("\n" + "="*70)
        print("No more splittable communities found. Halting.")
        print("="*70)
        break
    
    # --- If we found a split, update the list of communities ---
    split_result = result
    communities.remove(community_to_split)
    
    # Handle split communities (greedy returns exactly 2)
    if isinstance(split_result, tuple) and len(split_result) == 2:
        new_comm1, new_comm2 = split_result
        
        # Validate that all nodes from the original community are included
        combined_nodes = set(new_comm1) | set(new_comm2)
        original_nodes = set(community_to_split)
        
        if combined_nodes != original_nodes:
            missing_nodes = original_nodes - combined_nodes
            print(f"Warning: {len(missing_nodes)} nodes were lost, adding to first community")
            new_comm1 = list(set(new_comm1) | missing_nodes)
        
        communities.append(new_comm1)
        communities.append(new_comm2)
        
        print(f"\nSplit community of size {len(community_to_split)} using greedy modularity:")
        print(f"  -> Community 1: {sorted(new_comm1)} (size: {len(new_comm1)})")
        print(f"  -> Community 2: {sorted(new_comm2)} (size: {len(new_comm2)})")
    
    iteration += 1


# --- Final State ---
print("\n--- Community Detection Finished ---")
print(f"Final state: {len(communities)} communities found.")
print(f"\nFinal Communities:")
for idx, community in enumerate(communities):
    print(f"  Community {idx+1}: {sorted(community)} (size: {len(community)})")

print(f"\n" + "="*70)
print("BEST PARTITION SUMMARY - GREEDY MODULARITY APPROACH")
print("="*70)
print(f"Best partition found at Iteration {best_iteration} with {len(best_communities)} communities")
print(f"Best Global Modularity: {best_modularity:.4f}")
print(f"\nBest Communities:")
for idx, community in enumerate(best_communities):
    print(f"  Community {idx+1}: {sorted(community)} (size: {len(community)})")

print(f"\nModularity History:")
for iter_num, mod_score in modularity_history.items():
    print(f"  Iteration {iter_num}: {mod_score:.4f}")

print(f"\nApproach Characteristics:")
print(f"  - Algorithm: Greedy Modularity Optimization")
print(f"  - Behavior: Aggressive, explores partition space thoroughly")
print(f"  - Communities Detected: {len(best_communities)}")
print(f"  - Modularity Score: {best_modularity:.4f}")
print(f"  - Interpretation: Finer-grained partitions, continues until no significant improvement")
