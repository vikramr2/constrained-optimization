"""The main CLI logic, containing also the main algorithm"""
from __future__ import annotations

from typing import List, Union, Dict
from enum import Enum

import networkit as nk

# (VR) Change: I removed the context import since we do everything in memory
# (VR) Change 2: I brought back context just for IKC
from common.graph import Graph, IntangibleSubgraph, RealizedSubgraph
from common.cluster_tree import ClusterTreeNode


class ClustererSpec(str, Enum):
    """ (VR) Container for Clusterer Specification """  
    leiden = "leiden"
    ikc = "ikc"
    leiden_mod = "leiden_mod"

def annotate_tree_node(
    node: ClusterTreeNode, graph: Union[Graph, IntangibleSubgraph, RealizedSubgraph]
):
    """ (VR) Labels a ClusterTreeNode with its respective cluster """
    node.label = graph.index
    node.graph_index = graph.index
    node.num_nodes = graph.n()
    node.extant = False     # (VR) Def Extant: An input cluster that has remained untouched by CM (unpruned and uncut)
    node.cm_valid = True    # (VR) Def CM_Valid: A cluster that is in the final result, must have connectivity that fits the threshold

def update_cid_membership(
    subgraph: Union[Graph, IntangibleSubgraph, RealizedSubgraph],
    node2cids: Dict[int, str],
):
    """ (VR) Set nodes within current cluster to its respective cluster id """
    for n in subgraph.nodes():
        node2cids[n] = subgraph.index

def summarize_graphs(graphs: List[IntangibleSubgraph]) -> str:
    """ (VR) Summarize graphs for logging purposes """
    if not graphs:
        return "[](empty)"
    if len(graphs) > 3:
        return f"[{graphs[0].index}, ..., {graphs[-1].index}]({len(graphs)})"
    else:
        return f"[{', '.join([g.index for g in graphs])}]({len(graphs)})"