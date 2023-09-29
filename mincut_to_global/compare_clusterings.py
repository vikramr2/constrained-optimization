def get_grouping(mapping_file):
    # Initialize an empty dictionary to store the mapping
    cluster_to_nodes = {}

    # Read the TSV file and populate the dictionary
    with open(mapping_file, 'r') as file:
        for line in file:
            # Split each line into node_id and cluster_id
            node_id, cluster_id = line.strip().split('\t')
            node_id = int(node_id)

            # Check if cluster_id is already a key in the dictionary
            if cluster_id in cluster_to_nodes:
                cluster_to_nodes[cluster_id].append(node_id)
            else:
                # If cluster_id is not in the dictionary, create a new entry
                cluster_to_nodes[cluster_id] = [node_id]

    # Print the dictionary, where cluster IDs map to arrays of node IDs
    ret = []
    for cluster_id, node_ids in cluster_to_nodes.items():
        ret.append(node_ids)

    # Sort the inner arrays based on their elements
    sorted_array = [list(sorted(arr)) for arr in ret]

    # Sort the outer array based on the sorted inner arrays
    sorted_array.sort()

    return sorted_array

def compare_clustering(file1, file2):
    return get_grouping(file1) == get_grouping(file2)