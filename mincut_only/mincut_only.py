from __future__ import annotations

import networkit as nk
import treeswift as ts

from typing import Optional, Dict, cast
from os import path
from structlog import get_logger

import typer
import sys
import time
import jsonpickle

from connected_components import get_components

from common.graph import Graph
from common.mincut_requirement import MincutRequirement
from common.pruner import prune_graph
from common.to_universal import cm2universal
from common.json2membership import json2membership
from common.cluster_tree import ClusterTreeNode
from common.utils import update_cid_membership, annotate_tree_node


def algorithm_g(quiet, node2cids, node_mapping):
    # (VR) Main algorithm loop: Recursively cut clusters in stack until they have mincut above threshold
    while stack:
        if not quiet:
            log = get_logger()
            log.debug("entered next iteration of loop", queue_size=len(stack))

        # (VR) Get the next cluster to operate on
        subgraph = stack.pop()
        if not quiet:
            log.debug(
                "popped graph",
                graph_n=subgraph.n(),
                graph_index=subgraph.index,
            )

        # (VR) Mark nodes in popped cluster with their respective cluster ID
        update_cid_membership(subgraph, node2cids)
        if not quiet:
            log.debug(
                "updated cid membership",
            )

        # (VR) Get the current cluster tree node
        tree_node = node_mapping[subgraph.index]

        # (VR) If the current cluster is a singleton or empty, move on
        if subgraph.n() <= 1:
            continue

         # (VR) Log current cluster data after realization
        if not quiet:
            log = log.bind(
                g_id=subgraph.index,
                g_n=subgraph.n(),
                g_m=subgraph.m()
            )

        # (VR) Get minimum node degree in current cluster
        original_mcd = subgraph.mcd()
        if not quiet:
            log.debug(
                "computed mcd",
                g_mcd=subgraph.mcd()
            )

        # (VR) Pruning: Remove singletons with node degree under threshold until there exists none
        num_pruned = prune_graph(subgraph, requirement, None)

        if num_pruned > 0:
            # (VR) Set the cluster cut size to the degree of the removed node
            tree_node.cut_size = original_mcd
            tree_node.extant = False                        # (VR) Change: The current cluster has been changed, so its not extant or CM valid anymore
            tree_node.cm_valid = False

            if not quiet:
                log.info("pruned graph", num_pruned=num_pruned)

            # (VR) Create a TreeNodeCluster for the pruned cluster and set it as the current node's child
            new_child = ClusterTreeNode()
            subgraph.index = f"{subgraph.index}δ"
            annotate_tree_node(new_child, subgraph)
            tree_node.add_child(new_child)
            node_mapping[subgraph.index] = new_child

            # (VR) Iterate to the new node
            tree_node = new_child
            update_cid_membership(subgraph, node2cids)

        # (VR) Compute the mincut and validity threshold of the cluster
        mincut_res = subgraph.find_mincut()
        valid_threshold = requirement.validity_threshold(None, subgraph)
        if not quiet:
            log.debug("calculated validity threshold", validity_threshold=valid_threshold)
            log.debug(
                "mincut computed",
                a_side_size=len(mincut_res.get_light_partition()),
                b_side_size=len(mincut_res.get_heavy_partition()),
                cut_size=mincut_res.get_cut_size(),
            )
        
        # (VR) Set the current cluster's cut size
        tree_node.cut_size = mincut_res.get_cut_size()
        tree_node.validity_threshold = valid_threshold

        components = subgraph.find_connected_components()
        if len(components) > 1:
            tree_node.cm_valid = False                      # (VR) Change: The current cluster has been changed, so its not extant or CM valid anymore
            tree_node.extant = False
            for cluster in components:
                node = ClusterTreeNode()
                annotate_tree_node(node, cluster)
                tree_node.add_child(node)
                node_mapping[cluster.index] = node
            stack.extend(components)
            # (VR) Log the partitions
            if not quiet:
                log.info(
                    "cluster disconnected: split into connected components",
                    n_components=len(components)
                )
            continue
        else:
            subgraph = components[0]

        # (VR) If the cut size is below validity, split!
        if mincut_res.get_cut_size() <= valid_threshold:    # and mincut_res.get_cut_size >= 0: -> (VR) Change: Commented this out to handle disconnected clusters
            tree_node.cm_valid = False                      # (VR) Change: The current cluster has been changed, so its not extant or CM valid anymore
            tree_node.extant = False
            
            # (VR) Split partitions and set them as children nodes
            p1, p2 = subgraph.cut_by_mincut(mincut_res)

            node_a = ClusterTreeNode()
            node_b = ClusterTreeNode()

            annotate_tree_node(node_a, p1)
            annotate_tree_node(node_b, p2)

            tree_node.add_child(node_a)
            tree_node.add_child(node_b)

            node_mapping[p1.index] = node_a
            node_mapping[p2.index] = node_b

            stack.extend([p1, p2])

            # (VR) Log the partitions
            if not quiet:
                log.info(
                    "cluster split",
                    a_size=p1.n(),
                    b_size=p2.n()
                )
        else:
            if not quiet:
                log.info("cut valid, not splitting anymore")

    return (node_mapping, node2cids)

def main(
    input: str = typer.Option(..., "--input", "-i", help="The input network."),
    quiet: Optional[bool] = typer.Option(False, "--quiet", "-q", help="Silence output messages."),
    threshold: str = typer.Option("", "--threshold", "-t", help="Connectivity threshold which all clusters should be above."),
    output: str = typer.Option("", "--output", "-o", help="Output filename."),
    first_tsv: bool = typer.Option(False, "--firsttsv", "-f", help="Output the tsv file that comes before CM2Universal and json2membership.")
):
    # Set global variables
    global stack
    global requirement

    # Set default output name
    if output == "":
        base, _ = path.splitext(input)
        output = f"{base.split('/')[-1]}.mincut_only.tsv"

    if not quiet:
        log = get_logger()      # (VR) Change: removed working dir initialization
        log.debug(
            "Set output directory",
            output=output
        )

    # (VR) Setting a really high recursion limit to prevent stack overflow errors
    sys.setrecursionlimit(1231231234)

    # (VR) Start hm01
    if not quiet:
        log.info(
            f"starting mincut only heuristic",
            input=input
        )

    # (VR) Parse mincut threshold specification
    requirement = MincutRequirement.try_from_str(threshold)
    if not quiet:
        log.info(f"parsed connectivity requirement", requirement=requirement)

    # (VR) Get the initial time for reporting the time it took to load the graph
    time1 = time.time()

    # (VR) Load full graph into Graph object
    edgelist_reader = nk.graphio.EdgeListReader("\t", 0)
    nk_graph = edgelist_reader.read(input)
    global_graph = Graph(nk_graph, "")
    if not quiet:
        log.info(
            f"loaded graph",
            n=nk_graph.numberOfNodes(),
            m=nk_graph.numberOfEdges(),
            elapsed=time.time() - time1,
        )

    # (VR) Get the initial time for reporting the time it took to load the graph
    time1 = time.time()
    stack = get_components(global_graph)
    if not quiet:
        log.info(
            "computed connected components",
            components=len(stack),
            elapsed=time.time() - time1,
        )
    
    node2cids: Dict[int, str] = {}                      # (VR) node2cids: Mapping between nodes and cluster ID
    node_mapping: Dict[str, ClusterTreeNode] = {}       # (VR) node_mapping: maps cluster id to cluster tree node

    # Set the node to map the original graph
    n = ClusterTreeNode()
    n.extant = True
    node_mapping[''] = n

    tree = ts.Tree()                                    # (VR) tree: Recursion tree that keeps track of clusters created by serial mincut/reclusters

    n = ClusterTreeNode()
    n.extant = True
    annotate_tree_node(n, global_graph)
    tree.root = n
    node_mapping[""] = n

    for subgraph in stack:
        n = ClusterTreeNode()
        n.extant = True
        annotate_tree_node(n, subgraph)
        tree.root.add_child(n)
        node_mapping[subgraph.index] = n

    # (VR) Start the timer for the algorithmic stage of CM
    if not quiet:
        time1 = time.perf_counter()

    mapping, labels = algorithm_g(quiet, node2cids, node_mapping)


    # (VR) Log the output time for the algorithmic stage of CM
    if not quiet:
        log.info("CM algorithm completed", time_elapsed=time.perf_counter() - time1)

    # (VR) Retrieve output if we want the original tsv
    if first_tsv:
        with open(output, "w+") as f:
            for n, cid in labels.items():
                f.write(f"{n} {cid}\n")

    # (VR) Output the json data
    with open(output + ".tree.json", "w+") as f:
        f.write(cast(str, jsonpickle.encode(tree)))
        
    cm2universal(quiet, tree, labels, output)

    # (VR) Convert the 'after' json into a tsv file with columns (node_id, cluster_id)
    json2membership(output + ".after.json", output + ".after.tsv")

def entry_point():
    typer.run(main)

if __name__ == "__main__":
    entry_point()