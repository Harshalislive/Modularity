# Applying recursive bisection (Multi community detection)

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G = nx.karate_club_graph()
print(G)

# Visualize the graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300)
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=8)
plt.title("Karate Club Graph")
plt.axis('off')
plt.tight_layout()
plt.show()

#now let's create a recursive function that will check the modularity of check community that is formed and will split furhter based on it's modularity

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# Load all nodes and edges into a new graph (copy of original)
nodes_graph = nx.Graph()
nodes_graph.add_nodes_from(G.nodes())
nodes_graph.add_edges_from(G.edges())

print(f"Nodes in the graph: {nodes_graph.number_of_nodes()}")
print(f"Edges in the graph: {nodes_graph.number_of_edges()}")
print(f"Nodes: {list(nodes_graph.nodes())}")

# Function to find a split in a community using spectral bisection
def find_split(G_original, community_nodes, original_m=None):
    """
    Tests if a community can be split using spectral bisection method.
    Uses the modularity matrix eigenvalue decomposition.
    
    If yes, returns the two new communities. If no, returns None.
    
    Args:
        G_original: The original full graph
        community_nodes: List of nodes in the community to split
        original_m: Total number of edges in the graph (default: computed from G_original)
    
    Returns:
        Tuple of (comm1, comm2) if split is valid, None otherwise
    """
    if original_m is None:
        original_m = G_original.number_of_edges()
    
    subgraph = G_original.subgraph(community_nodes)
    sub_nodelist = list(subgraph.nodes())
    
    # A community of 1 or 0 cannot be split
    if len(sub_nodelist) <= 1:
        return None
    
    # Calculate the restricted modularity matrix B^(C) for the subgraph
    # Degrees (k) must be from the original graph
    sub_A = nx.to_numpy_array(subgraph, nodelist=sub_nodelist)
    original_degrees = np.array([G_original.degree(node) for node in sub_nodelist]).reshape(-1, 1)
    sub_B = sub_A - (original_degrees @ original_degrees.T) / (2 * original_m)
    
    # Find the leading eigenvalue (lambda_1)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(sub_B)
        lambda_1 = np.max(eigenvalues)
    except np.linalg.LinAlgError:
        return None
    
    # STOP if the leading eigenvalue is not meaningfully positive
    # This tolerance check prevents infinite loops
    if lambda_1 < 1e-10:
        return None
    
    # SPLIT the community using the signs of the corresponding eigenvector
    u_1 = eigenvectors[:, np.argmax(eigenvalues)]
    comm1 = [node for i, node in enumerate(sub_nodelist) if u_1[i] > 0]
    comm2 = [node for i, node in enumerate(sub_nodelist) if u_1[i] <= 0]
    
    # A split is only valid if it creates two non-empty groups
    if not comm1 or not comm2:
        return None
    
    return comm1, comm2


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


def calculate_metrics(graph, communities):
    """
    Calculate centrality and clustering metrics for nodes in their communities.
    
    Args:
        graph: The original full graph
        communities: List of communities
    
    Returns:
        Dictionary with metrics for each node
    """
    metrics = {'degree': {}, 'betweenness': {}, 'closeness': {}, 'clustering': {}}
    
    for community in communities:
        subgraph = graph.subgraph(community)
        
        deg_cen = nx.degree_centrality(subgraph)
        bet_cen = nx.betweenness_centrality(subgraph)
        clo_cen = nx.closeness_centrality(subgraph)
        clu_cen = nx.clustering(subgraph)
        
        for node in community:
            metrics['degree'][node] = deg_cen.get(node, 0)
            metrics['betweenness'][node] = bet_cen.get(node, 0)
            metrics['closeness'][node] = clo_cen.get(node, 0)
            metrics['clustering'][node] = clu_cen.get(node, 0)
    
    return metrics


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


# === 2. Main Iterative Analysis Loop ===

print("\n" + "="*60)
print("STARTING ITERATIVE COMMUNITY DETECTION WITH METRICS")
print("="*60)

# Initial setup
communities = [list(G.nodes())]  # Start with one community: the whole graph
metrics_history = {}
iteration = 0
modularity_history = {}
best_modularity = 0.0
best_iteration = 0
best_communities = None

# Compute initial layout
pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)

print("\n--- Starting Iterative Community Detection ---")

while True:
    # --- Task 2 & 3: Visualize and Record Metrics for the current state ---
    title = f"Iteration {iteration}: {len(communities)} communities"
    print(f"\n--- {title} ---")
    visualise_communities(G, pos, communities, title)
    
    print(f"Calculating metrics for iteration {iteration}...")
    iter_metrics = calculate_metrics(G, communities)
    metrics_history[iteration] = iter_metrics
    
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
    
    # Print sample metrics
    print(f"Sample node metrics (first 5 nodes):")
    for i, node in enumerate(list(G.nodes())[:5]):
        print(f"  Node {node}: degree={iter_metrics['degree'][node]:.4f}, " +
              f"betweenness={iter_metrics['betweenness'][node]:.4f}, " +
              f"closeness={iter_metrics['closeness'][node]:.4f}, " +
              f"clustering={iter_metrics['clustering'][node]:.4f}")
    
    # --- Task 1: Find and perform the next split ---
    
    # Calculate original_m (total edges in the graph) for spectral bisection
    original_m = G.number_of_edges()
    
    # Find the largest community that is still splittable
    community_to_split = None
    split_result = None
    
    # Sort communities by size to try splitting the largest ones first
    sorted_communities = sorted(communities, key=len, reverse=True)
    
    for community in sorted_communities:
        result = find_split(G, community, original_m=original_m)
        if result is not None:
            # We found a community we can split
            community_to_split = community
            split_result = result
            break  # Stop searching and perform the split
    
    # --- Check if we should stop the loop ---
    # Stop if modularity is decreasing significantly from the best
    if iteration > best_iteration and global_modularity < best_modularity * 0.95:
        print("\n" + "="*60)
        print("Modularity decreased significantly. Stopping at best partition.")
        print("="*60)
        communities = best_communities
        break
    
    if community_to_split is None:
        # If we went through all communities and none could be split, we're done.
        print("\n" + "="*60)
        print("No more splittable communities found. Halting.")
        print("="*60)
        break
    
    # --- If we found a split, update the list of communities ---
    split_result = result
    communities.remove(community_to_split)
    
    # Handle split communities (spectral bisection returns exactly 2)
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
        
        print(f"\nSplit community of size {len(community_to_split)} using spectral bisection:")
        print(f"  -> Community 1: {sorted(new_comm1)} (size: {len(new_comm1)})")
        print(f"  -> Community 2: {sorted(new_comm2)} (size: {len(new_comm2)})")
    iteration += 1

# --- Final State ---
print("\n--- Community Detection Finished ---")
print(f"Final state: {len(communities)} communities found.")
print(f"\nFinal Communities:")
for idx, community in enumerate(communities):
    print(f"  Community {idx+1}: {sorted(community)} (size: {len(community)})")

print(f"\n" + "="*60)
print("BEST PARTITION SUMMARY")
print("="*60)
print(f"Best partition found at Iteration {best_iteration} with {len(best_communities)} communities")
print(f"Best Global Modularity: {best_modularity:.4f}")
print(f"\nBest Communities:")
for idx, community in enumerate(best_communities):
    print(f"  Community {idx+1}: {sorted(community)} (size: {len(community)})")

print(f"\nModularity History:")
for iter_num, mod_score in modularity_history.items():
    print(f"  Iteration {iter_num}: {mod_score:.4f}")

print(f"\nFinal Global Modularity: {modularity_history[iteration-1]:.4f}")

# === METRIC EVOLUTION GRAPHS ===
print("\n" + "="*60)
print("GENERATING METRIC EVOLUTION GRAPHS")
print("="*60)

# Prepare data for metric evolution analysis
# For each node, track how its metrics change across iterations
node_metric_evolution = {
    'degree': {},
    'betweenness': {},
    'closeness': {},
    'clustering': {}
}

# Initialize evolution tracking for all nodes
for node in G.nodes():
    for metric_type in node_metric_evolution.keys():
        node_metric_evolution[metric_type][node] = []

# Populate the evolution data
for iter_num in sorted(metrics_history.keys()):
    iter_metrics = metrics_history[iter_num]
    for metric_type in node_metric_evolution.keys():
        for node in G.nodes():
            node_metric_evolution[metric_type][node].append(
                iter_metrics[metric_type].get(node, 0)
            )

# === Detailed Individual Node Tracking for ALL Nodes ===
# Create comprehensive tracking plots for all nodes
all_nodes = sorted(G.nodes())
print(f"\nTracking evolution for all {len(all_nodes)} nodes")

iterations = sorted(metrics_history.keys())

# Plot each metric for all nodes in a 2x2 grid
metrics_to_plot = [
    ('degree', 'Degree Centrality'),
    ('betweenness', 'Betweenness Centrality'),
    ('closeness', 'Closeness Centrality'),
    ('clustering', 'Clustering Coefficient')
]

# Create 2x2 grid figure
fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle('All Nodes Metric Evolution Across Iterations', fontsize=16, fontweight='bold')

# Use a colormap for all nodes
colors = plt.cm.tab20c(np.linspace(0, 1, len(all_nodes)))

for idx, (metric_type, metric_title) in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    
    for node_idx, node in enumerate(all_nodes):
        values = node_metric_evolution[metric_type][node]
        ax.plot(iterations, values, 'o-', linewidth=1.5, markersize=5,
               label=f'Node {node}', color=colors[node_idx], alpha=0.8)
    
    ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel(metric_title, fontsize=11, fontweight='bold')
    ax.set_title(f'{metric_title} Evolution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7, ncol=2)
    ax.set_xticks(iterations)

plt.tight_layout()
filename = 'all_nodes_metrics_evolution_2x2.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {filename} (All 4 metrics in 2x2 grid)")

# === Metric Statistics Summary Table ===
print("\n" + "="*60)
print("METRIC STATISTICS ACROSS ITERATIONS")
print("="*60)

summary_data = []
for iter_num in sorted(metrics_history.keys()):
    iter_metrics = metrics_history[iter_num]
    
    for metric_type in ['degree', 'betweenness', 'closeness', 'clustering']:
        values = [iter_metrics[metric_type].get(node, 0) for node in G.nodes()]
        
        summary_data.append({
            'Iteration': iter_num,
            'Metric': metric_type.replace('_', ' ').title(),
            'Mean': np.mean(values),
            'Std Dev': np.std(values),
            'Min': np.min(values),
            'Max': np.max(values),
            'Median': np.median(values)
        })

# Print summary table
print(f"\n{'Iteration':<10} {'Metric':<20} {'Mean':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
print("-" * 80)
for item in summary_data:
    print(f"{item['Iteration']:<10} {item['Metric']:<20} "
          f"{item['Mean']:<10.4f} {item['Std Dev']:<10.4f} "
          f"{item['Min']:<10.4f} {item['Max']:<10.4f} {item['Median']:<10.4f}")


print("\n" + "="*60)
print("ALL METRIC EVOLUTION GRAPHS GENERATED SUCCESSFULLY!")
print("="*60)
print("\nGenerated file:")
print("  • all_nodes_metrics_evolution_2x2.png - All 4 metrics in 2x2 grid layout")

