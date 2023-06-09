import numpy as np
import scipy.sparse as sp

def _preprocess_features(features):
    """
    Copy from https://github.com/danielegrattarola/spektral/blob/39fe897c5c06ce8bd8100e10fe9d373b91958cc7/spektral/datasets/citation.py#L192
    """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


class GCNConv_preprocess_features(object):
    """
    Applies the `_preprocess_features` to the node features.
    
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, graph):
        graph.x = _preprocess_features(graph.x)
        return graph
       
