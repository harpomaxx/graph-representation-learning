## NOTA:
## Primera parte del código (y adaptación de la tercera parte) tomada de https://keras.io/examples/graph/gnn_citations/
## En la segunda parte, se toman algunos datos propios de CTU-13 para poner a andar la misma GNN del ejemplo de keras.
## El objetivo es lograr tener un demo andando, no obtener resultados de esta prueba. En efecto se prueba por 2 épocas y, siguiendo el ejemplo de keras,
## se usa accuracy como métrica, que para el problema de detección de botnets en la red no tiene sentido.

## Las features de cada nodo corresponden a la información del grado: ID, OD, IDW, ODW
## No tenemos en cuenta features que hacen a la centralidad, como BC, AC y LCC
 
## Lo próximo a realizar es una prueba con todos los datos de CTU-13, para más épocas, y tomando métricas más adecuadas.


###################################################################################################################


## Primera parte: clases y funciones del ejemplo de keras

import os
import pandas as pd
import numpy as np
#import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


hidden_units = [32, 32]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 2
batch_size = 256


# This function compiles and trains an input model using the given training data.
def run_experiment(model, x_train, y_train):
    # Compile the model.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # Tati: categorical_crossentropy, expects the labels to follow a categorical encoding. 
        #       With integer labels, you should use sparse_categorical_crossentropy.
        #       This new loss function is still mathematically the same as categorical_crossentropy; it just has a different interface.
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=50, restore_best_weights=True
    )
    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stopping],
    )

    return history


# This function displays the loss and accuracy curves of the model during training.
def display_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])
    ax1.legend(["train", "test"], loc="upper right")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2.plot(history.history["acc"])
    ax2.plot(history.history["val_acc"])
    ax2.legend(["train", "test"], loc="upper right")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.show()


def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(fnn_layers, name=name)


class GraphConvLayer(layers.Layer):
    def __init__(
        self,
        hidden_units,
        dropout_rate=0.2,
        aggregation_type="mean",
        combination_type="concat",
        normalize=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize

        self.ffn_prepare = create_ffn(hidden_units, dropout_rate)
        if self.combination_type == "gated":
            self.update_fn = layers.GRU(
                units=hidden_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                dropout=dropout_rate,
                return_state=True,
                recurrent_dropout=dropout_rate,
            )
        else:
            self.update_fn = create_ffn(hidden_units, dropout_rate)

    def prepare(self, node_repesentations, weights=None):
        # node_repesentations shape is [num_edges, embedding_dim].
        messages = self.ffn_prepare(node_repesentations)
        if weights is not None:
            messages = messages * tf.expand_dims(weights, -1)
        return messages

    def aggregate(self, node_indices, neighbour_messages, node_repesentations):
        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        # node_repesentations shape is [num_nodes, representation_dim]
        num_nodes = node_repesentations.shape[0]
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")

        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        # node_repesentations shape is [num_nodes, representation_dim].
        # aggregated_messages shape is [num_nodes, representation_dim].
        if self.combination_type == "gru":
            # Create a sequence of two elements for the GRU layer.
            h = tf.stack([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "concat":
            # Concatenate the node_repesentations and aggregated_messages.
            h = tf.concat([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "add":
            # Add node_repesentations and aggregated_messages.
            h = node_repesentations + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        # Apply the processing function.
        node_embeddings = self.update_fn(h)
        if self.combination_type == "gru":
            node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]

        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def call(self, inputs):
        """Process the inputs to produce the node_embeddings.

        inputs: a tuple of three elements: node_repesentations, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """

        node_repesentations, edges, edge_weights = inputs
        # Get node_indices (source) and neighbour_indices (target) from edges.
        node_indices, neighbour_indices = edges[0], edges[1]
        # neighbour_repesentations shape is [num_edges, representation_dim].
        neighbour_repesentations = tf.gather(node_repesentations, neighbour_indices)

        # Prepare the messages of the neighbours.
        neighbour_messages = self.prepare(neighbour_repesentations, edge_weights)
        # Aggregate the neighbour messages.
        aggregated_messages = self.aggregate(
            node_indices, neighbour_messages, node_repesentations
        )
        # Update the node embedding with the neighbour messages.
        return self.update(node_repesentations, aggregated_messages)


class GNNNodeClassifier(tf.keras.Model):
    def __init__(
        self,
        graph_info,
        num_classes,
        hidden_units,
        aggregation_type="sum",
        combination_type="concat",
        dropout_rate=0.2,
        normalize=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Unpack graph_info to three elements: node_features, edges, and edge_weight.
        node_features, edges, edge_weights = graph_info
        self.node_features = node_features
        self.edges = edges
        self.edge_weights = edge_weights
        # Set edge_weights to ones if not provided.
        if self.edge_weights is None:
            self.edge_weights = tf.ones(shape=edges.shape[1])
        # Scale edge_weights to sum to 1.
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights)

        # Create a process layer.
        self.preprocess = create_ffn(hidden_units, dropout_rate, name="preprocess")
        # Create the first GraphConv layer.
        self.conv1 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv1",
        )
        # Create the second GraphConv layer.
        self.conv2 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv2",
        )
        # Create a postprocess layer.
        self.postprocess = create_ffn(hidden_units, dropout_rate, name="postprocess")
        # Create a compute logits layer.
        self.compute_logits = layers.Dense(units=num_classes, name="logits")  
          # Tati: For Tensorflow: logits is a name that it is thought to imply that this Tensor is the quantity that is being mapped to probabilities by the Softmax

    def call(self, input_node_indices):
        # Preprocess the node_features to produce node representations.
        x = self.preprocess(self.node_features)
        # Apply the first graph conv layer.
        x1 = self.conv1((x, self.edges, self.edge_weights))
        # Skip connection.
        x = x1 + x
        # Apply the second graph conv layer.
        x2 = self.conv2((x, self.edges, self.edge_weights))
        # Skip connection.
        x = x2 + x
        # Postprocess node embedding.
        x = self.postprocess(x)
        # Fetch node embeddings for the input node_indices.
        node_embeddings = tf.gather(x, input_node_indices)
        # Compute logits
        return self.compute_logits(node_embeddings)


###################################################################################################################


## Segunda parte: cargar datos 

# Capture 16-2 (for training)
capture162 = pd.read_csv(
    "capture20110816-2_noZeroB.ncol",
    sep=" ",  
    header=None,  # no heading row
    names=["source", "target", "weight"],  # set our own names for the columns
)
capture162['source'] = '162-' + capture162['source'].astype(str)
capture162['target'] = '162-' + capture162['target'].astype(str)

capture162_features_tmp = pd.read_csv(
    "capture20110816-2_features.csv",
    sep=",",  
    header=0
)
capture162_features_tmp.iloc[:,8] = capture162_features_tmp.iloc[:,8].replace(to_replace="background", value="normal").copy()
capture162_features_tmp['node'] = '162-' + capture162_features_tmp['node'].astype(str)


# Capture 17 (for testing)
capture17 = pd.read_csv(
    "capture20110817_noZeroB.ncol",
    sep=" ",  
    header=None,  # no heading row
    names=["source", "target", "weight"],  # set our own names for the columns
)
capture17['source'] = '17-' + capture17['source'].astype(str)
capture17['target'] = '17-' + capture17['target'].astype(str)

capture17_features_tmp = pd.read_csv(
    "capture20110817_features.csv",
    sep=",",  
    header=0
)
capture17_features_tmp.iloc[:,8] = capture17_features_tmp.iloc[:,8].replace(to_replace="background", value="normal").copy()
capture17_features_tmp['node'] = '17-' + capture17_features_tmp['node'].astype(str)

capture17_sample = capture17.iloc[:70000,]
dirIP = pd.concat([capture17_sample.iloc[:,0], capture17_sample.iloc[:,1]], axis=0)
dirIP = pd.DataFrame(dirIP, columns=['node'])

node_values17 = sorted(dirIP["node"].unique())
capture17_sample_features_tmp = capture17_features_tmp[capture17_features_tmp["node"].isin(node_values17)]


# Ambas capturas
ambos = pd.concat([capture162, capture17_sample], axis=0)
ambos_features_tmp = pd.concat([capture162_features_tmp, capture17_sample_features_tmp], axis=0)

class_values_ambos = sorted(ambos_features_tmp["label"].unique())
class_idx_ambos = {name: id for id, name in enumerate(class_values_ambos)}
node_idx_ambos = {name: idx for idx, name in enumerate(sorted(ambos_features_tmp["node"].unique()))}

ambos_features = ambos_features_tmp.loc[:,["node","ID","OD","IDW","ODW","label"]].copy()
ambos_features["node"] = ambos_features_tmp["node"].apply(lambda name: node_idx_ambos[name])
ambos["source"] = ambos["source"].apply(lambda name: node_idx_ambos[name])
ambos["target"] = ambos["target"].apply(lambda name: node_idx_ambos[name])
ambos_features["label"] = ambos_features_tmp["label"].apply(lambda value: class_idx_ambos[value])

# Create an edges array (adjacency matrix) of shape [2, num_edges].
edges_ambos = ambos[["source", "target"]].to_numpy().T

# Create an edge weights array.
edge_weights_ambos = ambos[["weight"]].to_numpy().T 
edge_weights_ambos = edge_weights_ambos.reshape((edges_ambos.shape[-1],))
edge_weights_ambos = tf.convert_to_tensor(edge_weights_ambos)

# Create a node features array of shape [num_nodes, num_features].
feature_names_ambos = set(ambos_features.columns) - {"node", "label"}
node_features_ambos = tf.cast(
    ambos_features.sort_values("node")[feature_names_ambos].to_numpy(), dtype=tf.dtypes.float32
)

# Create graph info tuple with node_features, edges, and edge_weights.
graph_info_ambos = (node_features_ambos, edges_ambos, edge_weights_ambos)

print("Edges shape:", edges_ambos.shape)
print("Nodes shape:", node_features_ambos.shape)


###################################################################################################################


## Tercera parte: poner a andar la GNN

## GNN
gnn_model_ambos = GNNNodeClassifier(
    graph_info=graph_info_ambos,
    num_classes=len(class_idx_ambos),
    hidden_units=hidden_units,
    dropout_rate=dropout_rate,
    name="gnn_model_ambos",
)

print("GNN output shape:", gnn_model_ambos([1, 10, 100]))

print(gnn_model_ambos.summary())



train_data = ambos_features.iloc[:37943,:] # 37943 == capture162_features_tmp.shape[0]
test_data = ambos_features.iloc[37943:,:]

#train_data = pd.concat(train_data).sample(frac=1)
#test_data = pd.concat(test_data).sample(frac=1)

print("Train data shape:", train_data.shape) 
print("Test data shape:", test_data.shape) 

# Create train and test features as a numpy array.
x_train = train_data[feature_names_ambos].to_numpy()
x_test = test_data[feature_names_ambos].to_numpy()
# Create train and test targets as a numpy array.
y_train = train_data["label"]
y_test = test_data["label"]


x_train = train_data.node.to_numpy()
num_epochs = 2
history = run_experiment(gnn_model_ambos, x_train, y_train)


#display_learning_curves(history)


x_test = test_data.node.to_numpy()
_, test_accuracy = gnn_model_ambos.evaluate(x=x_test, y=y_test, verbose=1)
print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")

