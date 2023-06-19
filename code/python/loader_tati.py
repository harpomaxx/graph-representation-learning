import numpy as np
import tensorflow as tf

from spektral.data.loaders import Loader
from spektral.data.utils import (
    #batch_generator,
    collate_labels_batch,
    #collate_labels_disjoint,
    #get_spec,
    prepend_none,
    sp_matrices_to_sp_tensors,
    to_batch,
    #to_disjoint,
    #to_mixed,
    #to_tf_signature,
)

#def add_signature(signature):
#    new_signature = signature.copy()
#    new_signature["sample_weight"] = {
#                                    "spec": signature["y"]["spec"],
#                                    "shape": signature["y"]["shape"],
#                                    "dtype": signature["x"]["dtype"],
#                                }
#    return new_signature


def to_tf_signature_Tati(signature, class_weights):
    """
    Converts a Dataset signature to a TensorFlow signature.
    :param signature: a Dataset signature.
    :return: a TensorFlow signature.
    """
    output = []
    keys = ["x", "a", "e", "i"]
    for k in keys:
        if k in signature:
            shape = signature[k]["shape"]
            dtype = signature[k]["dtype"]
            spec = signature[k]["spec"]
            output.append(spec(shape, dtype))
    output = tuple(output)
    if "y" in signature:
        shape = signature["y"]["shape"]
        dtype = signature["y"]["dtype"]
        spec = signature["y"]["spec"]
        output = (output, spec(shape, dtype))
        if class_weights is not None:
            shape = signature["y"]["shape"]
            dtype = signature["x"]["dtype"]
            spec = signature["y"]["spec"]
            output = (output, spec(shape, dtype))
    return output



class BatchLoader_Tati(Loader):
    """
    A Loader for [batch mode](https://graphneural.network/data-modes/#batch-mode).
    This loader returns batches of graphs stacked along an extra dimension,
    with all "node" dimensions padded to be equal among all graphs.
    If `n_max` is the number of nodes of the biggest graph in the batch, then
    the padding consist of adding zeros to the node features, adjacency matrix,
    and edge attributes of each graph so that they have shapes
    `[n_max, n_node_features]`, `[n_max, n_max]`, and
    `[n_max, n_max, n_edge_features]` respectively.
    The zero-padding is done batch-wise, which saves up memory at the cost of
    more computation. If latency is an issue but memory isn't, or if the
    dataset has graphs with a similar number of nodes, you can use
    the `PackedBatchLoader` that zero-pads all the dataset once and then
    iterates over it.
    Note that the adjacency matrix and edge attributes are returned as dense
    arrays.
    if `mask=True`, node attributes will be extended with a binary mask that indicates
    valid nodes (the last feature of each node will be 1 if the node was originally in
    the graph and 0 if it is a fake node added by zero-padding).
    Use this flag in conjunction with layers.base.GraphMasking to start the propagation
    of masks in a model (necessary for node-level prediction and models that use a
    dense pooling layer like DiffPool or MinCutPool).
    If `node_level=False`, the labels are interpreted as graph-level labels and
    are returned as an array of shape `[batch, n_labels]`.
    If `node_level=True`, then the labels are padded along the node dimension and are
    returned as an array of shape `[batch, n_max, n_labels]`.
    **Arguments**
    - `dataset`: a graph Dataset;
    - `mask`: bool, whether to add a mask to the node features;
    - `batch_size`: int, size of the mini-batches;
    - `epochs`: int, number of epochs to iterate over the dataset. By default (`None`)
    iterates indefinitely;
    - `shuffle`: bool, whether to shuffle the data at the start of each epoch;
    - `node_level`: bool, if `True` pad the labels along the node dimension;
    **Output**
    For each batch, returns a tuple `(inputs, labels)`.
    `inputs` is a tuple containing:
    - `x`: node attributes of shape `[batch, n_max, n_node_features]`;
    - `a`: adjacency matrices of shape `[batch, n_max, n_max]`;
    - `e`: edge attributes of shape `[batch, n_max, n_max, n_edge_features]`.
    `labels` have shape `[batch, n_labels]` if `node_level=False` or
    `[batch, n_max, n_labels]` otherwise.
    """
        
    def __init__(
        self,
        dataset,
        mask=False,
        batch_size=1,
        epochs=None,
        shuffle=True,
        node_level=False,
        class_weights=None
    ):
        self.mask = mask
        self.node_level = node_level
        self.signature = dataset.signature
        self.class_weights = class_weights
        super().__init__(dataset, batch_size=batch_size, epochs=epochs, shuffle=shuffle)
    

    def collate(self, batch):
        packed = self.pack(batch)

        y = packed.pop("y_list", None)
        if y is not None:
            y = collate_labels_batch(y, node_level=self.node_level)
            ## Create a tensor with class weights
            sample_weights = tf.gather(self.class_weights, tf.argmax(y, axis=-1))

        output = to_batch(**packed, mask=self.mask)
        output = sp_matrices_to_sp_tensors(output)

        if len(output) == 1:
            output = output[0]

        if y is None:
            return output
        else:
            if self.class_weights is not None:
                return output, y, sample_weights
            else: 
                return output, y

    def tf_signature(self):
        """
        Adjacency matrix has shape [batch, n_nodes, n_nodes]
        Node features have shape [batch, n_nodes, n_node_features]
        Edge features have shape [batch, n_nodes, n_nodes, n_edge_features]
        Labels have shape [batch, n_labels]
        """
        signature = self.signature
        for k in signature:
            signature[k]["shape"] = prepend_none(signature[k]["shape"])
        if "x" in signature and self.mask:
            # In case we have a mask, the mask is concatenated to the features
            signature["x"]["shape"] = signature["x"]["shape"][:-1] + (
                signature["x"]["shape"][-1] + 1,
            )
        if "a" in signature:
            # Adjacency matrix in batch mode is dense
            signature["a"]["spec"] = tf.TensorSpec
        if "e" in signature:
            # Edge attributes have an extra None dimension in batch mode
            signature["e"]["shape"] = prepend_none(signature["e"]["shape"])
        if "y" in signature and self.node_level:
            # Node labels have an extra None dimension
            signature["y"]["shape"] = prepend_none(signature["y"]["shape"])
        return to_tf_signature_Tati(signature, self.class_weight)
        
        