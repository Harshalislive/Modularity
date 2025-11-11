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

# Function to calculate modularity and determine if a subgraph can be further split
def calculate_modularity_and_split(subgraph, modularity_threshold=0.3):
    """
    Calculate modularity for a subgraph using community detection.
    If modularity is high enough, the subgraph can be split further.
    """
    if subgraph.number_of_nodes() < 2:
        return {
            'modularity': 0,
            'communities': [set(subgraph.nodes())],
            'can_split': False,
            'reason': 'Graph has less than 2 nodes'
        }

    from networkx.algorithms import community
    communities = list(community.greedy_modularity_communities(subgraph))

    modularity = community.modularity(subgraph, communities)

    can_split = len(communities) > 1 and modularity > modularity_threshold and subgraph.number_of_nodes() > 2

    return {
        'modularity': modularity,
        'communities': communities,
        'can_split': can_split,
        'num_communities': len(communities),
        'num_nodes': subgraph.number_of_nodes()
    }


def recursive_bisection_with_visualization(subgraph, iteration=0, modularity_threshold=0.3, max_iterations=10):
    if iteration >= max_iterations:
        print(f"\nReached maximum iterations ({max_iterations})")
        return

    result = calculate_modularity_and_split(subgraph, modularity_threshold)

    print(f"\n{'='*60}")
    print(f"ITERATION {iteration}")
    print(f"{'='*60}")
    print(f"Number of nodes: {result['num_nodes']}")
    print(f"Modularity Score: {result['modularity']:.4f}")
    print(f"Communities found: {result['num_communities']}")
    print(f"Can split further: {result['can_split']}")

    # For iteration 0, show the entire graph first, then the detected communities
    if iteration == 0:
        # First visualization: show the entire graph as one
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Left plot: entire graph
        pos = nx.spring_layout(subgraph, seed=42)
        nx.draw_networkx_nodes(subgraph, pos, node_color='lightblue', node_size=300, alpha=0.8, ax=axes[0])
        nx.draw_networkx_edges(subgraph, pos, alpha=0.5, ax=axes[0])
        nx.draw_networkx_labels(subgraph, pos, font_size=8, ax=axes[0])
        axes[0].set_title(f"Initial Graph\n({subgraph.number_of_nodes()} nodes)", fontsize=10, fontweight='bold')
        axes[0].axis('off')
        
        # Right plot: detected communities
        colors = plt.cm.Set3(np.linspace(0, 1, result['num_communities']))
        for idx, community_nodes in enumerate(result['communities']):
            community_graph = subgraph.subgraph(community_nodes).copy()
            node_colors = [colors[idx] if node in community_nodes else 'lightgray' for node in subgraph.nodes()]
            pos_full = nx.spring_layout(subgraph, seed=42)
            nx.draw_networkx_nodes(subgraph, pos_full, node_color=node_colors, node_size=300, alpha=0.8, ax=axes[1])
            nx.draw_networkx_edges(subgraph, pos_full, alpha=0.2, ax=axes[1])
        
        nx.draw_networkx_labels(subgraph, pos_full, font_size=8, ax=axes[1])
        axes[1].set_title(f"Detected Communities ({result['num_communities']} communities)", fontsize=10, fontweight='bold')
        axes[1].axis('off')
        
        plt.suptitle(f"Iteration {iteration} - Modularity: {result['modularity']:.4f}", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    else:
        # For iterations > 0, show each community separately
        num_comms = max(result['num_communities'], 1)
        fig, axes = plt.subplots(1, num_comms, figsize=(5 * num_comms, 5))
        if num_comms == 1:
            axes = [axes]

        colors = plt.cm.Set3(np.linspace(0, 1, num_comms))

        for idx, community_nodes in enumerate(result['communities']):
            ax = axes[idx]
            community_graph = subgraph.subgraph(community_nodes).copy()
            pos = nx.spring_layout(community_graph, seed=42)

            nx.draw_networkx_nodes(community_graph, pos, node_color=[colors[idx]], node_size=300, alpha=0.8, ax=ax)
            nx.draw_networkx_edges(community_graph, pos, alpha=0.5, ax=ax)
            nx.draw_networkx_labels(community_graph, pos, font_size=8, ax=ax)

            ax.set_title(f"Community {idx+1}\n({len(community_nodes)} nodes)", fontsize=10, fontweight='bold')
            ax.axis('off')

        plt.suptitle(f"Iteration {iteration} - Modularity: {result['modularity']:.4f}", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    if result['can_split']:
        for idx, community_nodes in enumerate(result['communities']):
            community_subgraph = subgraph.subgraph(community_nodes).copy()
            print(f"\n--- Processing Community {idx+1} with {len(community_nodes)} nodes ---")
            recursive_bisection_with_visualization(community_subgraph, iteration + 1, modularity_threshold, max_iterations)
    else:
        if iteration == 0:
            print("\nFinal communities (cannot be split further):")
            for idx, community_nodes in enumerate(result['communities']):
                print(f"  Final Community {idx+1}: {sorted(list(community_nodes))}")
        else:
            print(f"\nCommunity cannot be split further (modularity: {result['modularity']:.4f})")


# Run recursive bisection with visualization
print("\n" + "="*60)
print("STARTING RECURSIVE BISECTION")
print("="*60)
recursive_bisection_with_visualization(nodes_graph, modularity_threshold=0.3, max_iterations=5)
