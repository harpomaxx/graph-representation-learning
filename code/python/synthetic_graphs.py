import os
import shutil 

import numpy as np
import gc
import json

import tensorflow as tf

from spektral.data import Dataset, Graph
from spektral.datasets.utils import DATASET_FOLDER

from scipy import sparse
from scipy.special import softmax


def save_parameters(parameters, filename):
    with open(filename, 'w') as f:
        json.dump(parameters, f)
        
def load_parameters(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def generate_synthetic_graph_csbm(num_nodes, num_communities, num_features, edge_prob_matrix, node_features_mean,\
                                  feature_cov_matrix=None, directed=False, num_class1=None, seed=None):
    np.random.seed(seed)
    
    # Assign nodes to communities
    if num_class1 is None:
        communities = np.random.randint(0, num_communities, num_nodes) # caso balanceado
    else:
        indices = np.random.choice(num_nodes, num_class1, replace=False)
        communities = np.array([int(j in indices) for j in range(num_nodes)])
    
    # Generate node features
    if feature_cov_matrix is None:
        feature_cov_matrix = np.eye(num_features)
    features = np.zeros((num_nodes, num_features))
    for k in range(num_communities):
        nodes_in_community = np.where(communities == k)[0]
        features[nodes_in_community] = np.random.multivariate_normal(node_features_mean[k], feature_cov_matrix,\
                                                                     len(nodes_in_community))

    # Compute community membership probabilities based on node features
    community_membership_probs = softmax(features @ node_features_mean.T, axis=1)
    
    # Generate edges based on community membership probabilities
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    if directed:
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                community_i = communities[i]
                community_j = communities[j]
                edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                adjacency_matrix[i, j] = np.random.binomial(1, edge_prob)
    else: 
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                if i == j:
                    continue
                community_i = communities[i]
                community_j = communities[j]
                edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

    labels = tf.keras.utils.to_categorical(communities)
    adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
    return Graph(x=features, a=adjacency_matrix, y=labels)



class CacicSpringer_SyntheticGraphs(Dataset): # modificacion 2024 para cacic-springer
    
    def __init__(self, num_graphs, num_nodes, num_communities, num_features, edge_prob_matrix, \
                 node_features_mean, feature_cov_matrix=None, directed=False, class1_percent=None, \
                 seed=None, flattened=False, adjnul=False, **kwargs):
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.num_communities = num_communities
        self.num_features = num_features
        self.edge_prob_matrix = edge_prob_matrix
        self.node_features_mean = node_features_mean
        self.feature_cov_matrix = feature_cov_matrix
        self.directed = directed
        self.class1_percent = class1_percent
        self.seed = seed
        self.flattened = flattened
        self.adjnul = adjnul
        super().__init__(**kwargs)

    @property
    def path(self):
        edge_prob_matrix_str = '-'.join(map(str, self.edge_prob_matrix.flatten()))            
        node_features_mean_str = '-'.join(map(str, self.node_features_mean.flatten()))
        
        if self.feature_cov_matrix is None:
            feature_cov_matrix_str = "None"
        else:
            feature_cov_matrix_str = '-'.join(map(str, self.feature_cov_matrix.flatten()))
            
        if self.directed:
            dir_string = "Directed"
        else:
            dir_string = "Undirected"
        
        if self.class1_percent is None:
            class1_string = "Balanced"
        else:
            class1_string = f'{self.class1_percent}PercentClass1'

        dirname = f'{self.num_graphs}Graphs_{self.num_nodes}Nodes_{self.num_communities}Classes_\
{self.num_features}Features_{edge_prob_matrix_str}EdgeProbMatrix_{node_features_mean_str}NodeFeaturesMean_\
{dir_string}_{class1_string}_{feature_cov_matrix_str}FeaturesCov'

        return os.path.join(DATASET_FOLDER, "CacicSpringer_SyntheticGraphs", dirname)
        
        
    @property
    def num_class1(self):
        if self.class1_percent is None:
            return None
        else:
            return int( (self.class1_percent * self.num_nodes) / 100 )
    
    
    # Getter para obtener los argumentos utilizados
    def get_args(self):
        return {"num_graphs": self.num_graphs, "num_nodes": self.num_nodes, \
                "num_communities": self.num_communities, "num_features": self.num_features, \
                "edge_prob_matrix": self.edge_prob_matrix, "node_features_mean": self.node_features_mean, \
                "feature_cov_matrix": self.feature_cov_matrix, "directed": self.directed, \
                "class1_percent": self.class1_percent, "seed": self.seed, \
                "flattened": self.flattened, "adjnul": self.adjnul}

    def download(self):
        os.makedirs(self.path)
        
        parameters = self.get_args()
        parameters["edge_prob_matrix"] = self.edge_prob_matrix.tolist()
        parameters["node_features_mean"] = self.node_features_mean.tolist()
        
        if self.seed is None:
            vector_seed = [None] * self.num_graphs
        else:
            assert len(self.seed) >= self.num_graphs, f'There are not enough seeds ({len(self.seed)}) for graphs ({self.num_graphs})'
            vector_seed = self.seed[:self.num_graphs] 
            parameters["seed"] = vector_seed
        
        graphs = [generate_synthetic_graph_csbm(self.num_nodes, self.num_communities, self.num_features, \
                                                self.edge_prob_matrix, self.node_features_mean, \
                                                self.feature_cov_matrix, self.directed, \
                                                self.num_class1, vector_seed[i]) \
                  for i in range(self.num_graphs)]
        
        for j in range(self.num_graphs):
            filename = os.path.join(self.path, f'graph_{j:03d}.npz')
            np.savez(filename, x=graphs[j].x, a=graphs[j].a, y=graphs[j].y)
        
        save_parameters(parameters, os.path.join(self.path,"parameters.json"))
        # Free memory
        del graphs
        gc.collect()

        

    def read(self): 
        if os.path.exists(self.path):
            parameters = load_parameters(os.path.join(self.path,"parameters.json"))
        
        #### TATI: if self.seed==None entonces debería generar nuevos (no leer lo que ya está de antes).
        ####       Lo mismo si cambia alguna semilla ####
        if ((self.seed is None) or 
            (self.seed is not None and parameters["seed"] != self.get_args()["seed"][:self.num_graphs])):
            self.delete()
            self.download()           

        # We must return a list of Graph objects
        output = []

        for j in range(self.num_graphs):
            data = np.load(os.path.join(self.path, f'graph_{j:03d}.npz'), allow_pickle=True)
            #### if flattened, entonces en lugar de leer las features guardadas, asigna una matriz de todos 1 ####
            if self.flattened:
                x_features = np.ones((self.num_nodes, self.num_features))
            else:
                x_features = data['x']
            #### if adjnul, entonces los nodos no estan conectados. Solo influirian las features disponibles ####
            if self.adjnul:
                matrix = np.zeros((self.num_nodes, self.num_nodes))
                adj_matrix = sparse.csr_matrix(matrix)
            else:
                adj_matrix = data['a'][()] # también puede ser a=data['a'].item()
            output.append(
                Graph(x=x_features, a=adj_matrix, y=data['y']) 
            )

        return output
    
    
    def delete(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

