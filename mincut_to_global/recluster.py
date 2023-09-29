import sys
import time
from typing import Optional

import networkit as nk
import typer
from structlog import get_logger

from common.clusterers.ikc_wrapper import IkcClusterer
from common.clusterers.leiden_wrapper import LeidenClusterer, Quality
from common.graph import Graph, IntangibleSubgraph
from common.utils import ClustererSpec, summarize_graphs

    
def construct_components(graphs):
    g = nk.Graph(global_graph.n())
    for graph in graphs:
        # (VR) Realize the set of nodes contained by the graph (i.e. construct its adjacency list)
        if isinstance(graph, IntangibleSubgraph):
            subgraph = graph.realize(global_graph)
        else:
            subgraph = graph

        for u in subgraph.nodes():
            for v in subgraph.neighbors(u):
                g.addEdge(u, v)

    cc = nk.components.ConnectedComponents(g)
    cc.run()
    
    singleton_nodes = []
    for comp in cc.getComponents():
        if len(comp) == 1:
            singleton_nodes.append(comp[0])

    for node in singleton_nodes:
        g.removeNode(node)

    g.removeSelfLoops()

    return g

def main(
    input: str = typer.Option(..., "--input", "-i", help="The input network."),
    existing_clustering: str = typer.Option(..., "--existing-clustering", "-e", help="The existing clustering of the input network to be reclustered."),
    quiet: Optional[bool] = typer.Option(False, "--quiet", "-q", help="Silence output messages."), # (VR) Change: Removed working directory parameter since no FileIO occurs during runtime anymore
    clusterer_spec: ClustererSpec = typer.Option(..., "--clusterer", "-c", help="Clustering algorithm used to obtain the existing clustering."),
    k: int = typer.Option(-1, "--k", "-k", help="(IKC Only) k parameter."),
    resolution: float = typer.Option(-1, "--resolution", "-g", help="(Leiden Only) Resolution parameter."),
    output: str = typer.Option("", "--output", "-o", help="Output filename."),
    cores: int = typer.Option(4, "--nprocs", "-n", help="Number of cores to run in parallel.")
):
    global global_graph

    # (VR) Setting a really high recursion limit to prevent stack overflow errors
    sys.setrecursionlimit(1231231234)

    # (VR) Check -g and -k parameters for Leiden and IKC respectively
    if clusterer_spec == ClustererSpec.leiden:
        assert resolution != -1, "Leiden requires resolution"
        clusterer = LeidenClusterer(resolution)
    elif clusterer_spec == ClustererSpec.leiden_mod:
        assert resolution == -1, "Leiden with modularity does not support resolution"
        clusterer = LeidenClusterer(resolution, quality=Quality.modularity)
    else:
        assert k != -1, "IKC requires k"
        clusterer = IkcClusterer(k)

    # (VR) Start hm01
    if not quiet:
        log = get_logger()      # (VR) Change: removed working dir initialization
        log.info(
            f"starting recluster",
            input=input,
            clusterer=clusterer,
        )

    # (VR) Get the initial time for reporting the time it took to load the graph
    time1 = time.time()

    # (VR) Load full graph into Graph object
    edgelist_reader = nk.graphio.EdgeListReader("\t", 0)
    nk_graph = edgelist_reader.read(input)
    if not quiet:
        log.info(
            f"loaded graph",
            n=nk_graph.numberOfNodes(),
            m=nk_graph.numberOfEdges(),
            elapsed=time.time() - time1,
        )
    global_graph = Graph(nk_graph, "")

    # (VR) Load clustering
    if not quiet:
        log.info(f"loading existing clustering before algorithm-g", clusterer=clusterer)
    clusters = clusterer.from_existing_clustering(existing_clustering)
    if not quiet:
        log.info(
            f"first round of clustering obtained",
            num_clusters=len(clusters),
            summary=summarize_graphs(clusters),
        )

    mod_graph = construct_components(clusters)

    # Open the file for writing
    with open(f"{output}.newgraph.tsv", "w") as file:
        # Write the edges
        unique_edges = list(set(mod_graph.iterEdges()))
        for edge in unique_edges:
            file.write(f"{edge[0]}\t{edge[1]}\n")

    # (VR) Load clustering
    if not quiet:
        log.info(f"clustering new graph")

    new_clusters = clusterer.cluster_without_singletons(Graph(mod_graph, ""))
    
    with open(f"{output}.reclustering.tsv", "w") as file:
        for i, cluster in enumerate(new_clusters):
            for n in cluster.nodes():
                file.write(f"{n}\t{i}\n")

def entry_point():
    typer.run(main)

if __name__ == "__main__":
    entry_point()
