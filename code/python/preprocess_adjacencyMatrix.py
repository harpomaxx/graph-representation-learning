from spektral.layers import GCNConv
from spektral.utils import gcn_filter


class GCNConv_preprocess_adjacencyMatrix(object):
    """
    Applies the `gcn_filter` function of a GCN Layer to the adjacency
    matrix. The result depends on symmetry of adjacency matrix.

    **Arguments**

    - `layer_class`: the class of a layer from `spektral.layers.convolutional`,
    in order to verify if it is a GCNConv.
    - `symmetric`: boolean, indicates if adjacency matrix is symmetric (undirected graph),
    or if it is non-symmetric (directed graph).
    """

    def __init__(self, layer_class, symmetric):
        self.layer_class = layer_class
        self.symmetric = symmetric

    def __call__(self, graph):
        if self.layer_class == GCNConv:
            if self.symmetric:
                graph.a = gcn_filter(graph.a)
            else:
                graph.a = gcn_filter(graph.a, symmetric=False)
            return graph
        else:
            raise ValueError('The parameter must be GCNConv. For other convolutional layers, find the appropriate preprocessing')




