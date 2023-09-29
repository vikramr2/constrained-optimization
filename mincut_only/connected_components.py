import networkit as nk

from common.graph import IntangibleSubgraph

def get_components(graph):
     # Steps 4 and 5: Compute and access the connected components
    cc = nk.components.ConnectedComponents(graph._data)
    cc.run()
    components = cc.getComponents()

    return [IntangibleSubgraph(list(component), str(i)).realize(graph) for i, component in enumerate(components) if len(component) > 1]  # Print the connected components

