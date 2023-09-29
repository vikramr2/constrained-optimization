import networkx as nx
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for running leiden.')
    # Todo: Can we make the input arguments similar to runleiden to maintain
    #  consistency. For e.g. "-i" is input file in runleiden and here it is
    #  "number of iterations" and "-n" is the input file
    parser.add_argument(
        '-i', metavar='ip_net', type=str, required=True,
        help='input network edge-list path'
        )
    parser.add_argument(
        '-o', metavar='output', type=str, required=True,
        help='output membership path'
        )
    args = parser.parse_args()

    # Load the edge list from a TSV file
    edge_list_file = args.i

    # Create an empty graph and add edges from the edge list
    G = nx.Graph()
    with open(edge_list_file, "r") as f:
        for line in f:
            source, target = line.strip().split("\t")
            G.add_edge(source, target)

    # Find connected components
    components = list(nx.connected_components(G))

    # Create a dictionary to map nodes to their connected component ID
    component_mapping = {}
    for component_id, component in enumerate(components):
        for node in component:
            component_mapping[node] = component_id

    # Write the node ID and component ID to a TSV file
    output_file = args.o
    with open(output_file, "w") as f:
        for node, component_id in component_mapping.items():
            f.write(f"{node}\t{component_id}\n")

    # print("Node-component mapping saved to", output_file)
