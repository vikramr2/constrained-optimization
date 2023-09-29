import logging

# from hm01.graph import Graph, RealizedSubgraph

from pymincut.pygraph import PyGraph
from mincut_wrapper import MincutResult

logger = logging.getLogger(__name__)

def viecut(graph):
    """ (VR) Compute the mincut result via VieCut """
    if graph.n() == 2 and graph.m() == 1:               # (VR) If we have a single edge, save the effort by splitting it
        nodes = list(graph.nodes())
        return MincutResult([nodes[0]], [nodes[1]], 1)
    pygraph = graph.as_pygraph()
    cut_result = run_viecut_command(pygraph)
    return cut_result


def run_viecut_command(pygraph):
    """ (VR) Run the viecut command and return the mincut result object """
    algorithm = 'cactus'                                   # (VR) Mincut algorithm from Nagamochi et. al.
    queue_implementation = 'bqueue'
    balanced = True

    light_partition, heavy_partition, cut_size = pygraph.mincut(algorithm, queue_implementation, balanced)

    # if cut_size == 0:
    #     return MincutResult([], [], 0) -> (VR) Change: we still want to split 0-connectivity clusters

    return MincutResult(light_partition, heavy_partition, cut_size)