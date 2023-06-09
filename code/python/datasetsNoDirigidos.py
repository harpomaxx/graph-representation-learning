import os
import numpy as np
import gc
import tensorflow as tf

from spektral.data import Dataset, Graph
from spektral.datasets.utils import DATASET_FOLDER

from scipy import sparse
from scipy.special import softmax

"""
  Script que contiene todos los datasets NO DIRIGIDOS generados usando 
    la herramienta descripta en https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=10938&context=etd (ver discussion #56)
    
  Para una mayor descripción de los datasets, ver archivo read.me correspondiente.
"""


######################################################################################################################################
###                                No dirigido - 100 nodos - clases balanceadas                                                    ###
######################################################################################################################################


class synthetic_NoDir_100nodes_balanced_clasesSep(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * 100 nodes each
      * 2 balanced classes of nodes
      * "separated" classes (according to their graphical representation)
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    """    
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            communities = np.random.randint(0, n_communities, n_nodes)

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = 100
        n_features = 2
        n_classes = 2
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                        [0.8, 0.2],
                                        [0.3, 0.7]
                                    ])
        # Node feature means for each community: 2 communities (2 rows), 2 features (2 columns. For more features please add columns) 
        node_feature_means = np.array([
                                        [2, 1], 
                                        [1, 2] 
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 678, 789, 321, 654, 987]
        
        # Bien separadas las clases:
        graphs1 = [generate_synthetic_graph_csbm(n_nodes, n_classes, n_features, edge_prob_matrix, node_feature_means, semillas[i], 0, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_100nodes_balanced_clasesSep_0{j}.npz')
            np.savez(filename, x=graphs1[j].x, a=graphs1[j].a, y=graphs1[j].y)

        # Free memory
        del graphs1
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = 100
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_100nodes_balanced_clasesSep_0{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes, n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )     

        return output
    
    
###########################################################################################################################################################################################################


class synthetic_NoDir_100nodes_balanced_clasesJunt(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * 100 nodes each
      * 2 balanced classes of nodes
      * "more together" classes (according to their graphical representation)
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            communities = np.random.randint(0, n_communities, n_nodes)

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = 100
        n_features = 2
        n_classes = 2
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                    [0.5, 0.5],
                                    [0.5, 0.5]
                            ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [2, 1], #, 0, 0, 0],
                                        [1, 2] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 678, 789, 321, 654, 987]
        
        # Clases un poco más juntas:
        graphs2 = [generate_synthetic_graph_csbm(n_nodes, n_classes, n_features, edge_prob_matrix, node_feature_means, semillas[i], 1, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_100nodes_balanced_clasesJunt_1{j}.npz')
            np.savez(filename, x=graphs2[j].x, a=graphs2[j].a, y=graphs2[j].y)

        # Free memory
        del graphs2
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = 100
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_100nodes_balanced_clasesJunt_1{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes, n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output
    
    
###########################################################################################################################################################################################################


class synthetic_NoDir_100nodes_balanced_clasesMezcl(Dataset):  
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * 100 nodes each
      * 2 balanced classes of nodes
      * "mixed" classes (according to their graphical representation)
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            communities = np.random.randint(0, n_communities, n_nodes)

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = 100
        n_features = 2
        n_classes = 2
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                    [0.5, 0.5],
                                    [0.5, 0.5]
                            ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [1, 1], #, 0, 0, 0],
                                        [1, 1] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 678, 789, 321, 654, 987]
        
        # Clases completamente mezcladas:
        graphs3 = [generate_synthetic_graph_csbm(n_nodes, n_classes, n_features, edge_prob_matrix, node_feature_means, semillas[i], 2, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_100nodes_balanced_clasesMezcl_2{j}.npz')
            np.savez(filename, x=graphs3[j].x, a=graphs3[j].a, y=graphs3[j].y)
            
        # Free memory
        del graphs3
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = 100
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_100nodes_balanced_clasesMezcl_2{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes, n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output
    
    
###########################################################################################################################################################################################################


######################################################################################################################################
###                            No dirigido - diferente cantidad de nodos - clases balanceadas                                      ###
######################################################################################################################################


class synthetic_NoDir_diffSize_balanced_clasesSep(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * number of nodes: 25,50,100,200,400,800,1600,3200,6400,12800
      * 2 balanced classes of nodes
      * "separated" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            communities = np.random.randint(0, n_communities, n_nodes)

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_features = 2
        n_classes = 2
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                        [0.8, 0.2],
                                        [0.3, 0.7]
                                    ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [2, 1], #, 0, 0, 0],
                                        [1, 2] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 678, 789, 321, 654, 987]
        
        # Bien separadas las clases:
        graphs1 = [generate_synthetic_graph_csbm(n_nodes[i], n_classes, n_features, edge_prob_matrix, node_feature_means, semillas[i], 0, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_{n_nodes[j]}nodes_balanced_clasesSep_0{j}.npz')
            np.savez(filename, x=graphs1[j].x, a=graphs1[j].a, y=graphs1[j].y)

        # Free memory
        del graphs1
        gc.collect()


    def read(self):
         output = []
        
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_{n_nodes[j]}nodes_balanced_clasesSep_0{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes[j], n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output
        
    
###########################################################################################################################################################################################################


class synthetic_NoDir_diffSize_balanced_clasesJunt(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * number of nodes: 25,50,100,200,400,800,1600,3200,6400,12800
      * 2 balanced classes of nodes
      * "more together" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            communities = np.random.randint(0, n_communities, n_nodes)

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_features = 2
        n_classes = 2
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                    [0.5, 0.5],
                                    [0.5, 0.5]
                            ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [2, 1], #, 0, 0, 0],
                                        [1, 2] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 678, 789, 321, 654, 987]
        
        # Clases un poco más juntas:
        graphs2 = [generate_synthetic_graph_csbm(n_nodes[i], n_classes, n_features, edge_prob_matrix, node_feature_means, semillas[i], 1, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_{n_nodes[j]}nodes_balanced_clasesJunt_1{j}.npz')
            np.savez(filename, x=graphs2[j].x, a=graphs2[j].a, y=graphs2[j].y)

        # Free memory
        del graphs2
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_{n_nodes[j]}nodes_balanced_clasesJunt_1{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes[j], n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output
    
    
###########################################################################################################################################################################################################


class synthetic_NoDir_diffSize_balanced_clasesMezcl(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * number of nodes: 25,50,100,200,400,800,1600,3200,6400,12800
      * 2 balanced classes of nodes
      * "mixed" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            communities = np.random.randint(0, n_communities, n_nodes)

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_features = 2
        n_classes = 2
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                    [0.5, 0.5],
                                    [0.5, 0.5]
                            ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [1, 1], #, 0, 0, 0],
                                        [1, 1] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 678, 789, 321, 654, 987]
        
        # Clases completamente mezcladas:
        graphs3 = [generate_synthetic_graph_csbm(n_nodes[i], n_classes, n_features, edge_prob_matrix, node_feature_means, semillas[i], 2, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_{n_nodes[j]}nodes_balanced_clasesMezcl_2{j}.npz')
            np.savez(filename, x=graphs3[j].x, a=graphs3[j].a, y=graphs3[j].y)
            
        # Free memory
        del graphs3
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_{n_nodes[j]}nodes_balanced_clasesMezcl_2{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes[j], n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output
    
    
###########################################################################################################################################################################################################


######################################################################################################################################
###                                     No dirigido - 100 nodos - clases NO balanceadas                                            ###
######################################################################################################################################

    
class synthetic_NoDir_100nodes_NoBalanced_1a100_clasesSep(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * 100 nodes each
      * 2 unbalanced classes: 1% for the minority class
      * "separated" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  n_infected, semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            indices = [np.random.randint(0,n_nodes) for i in range(n_infected)]
            communities = np.array([int(j in indices) for j in range(n_nodes)])

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = 100
        n_features = 2
        n_classes = 2
        n_infected = 1
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                        [0.8, 0.2],
                                        [0.3, 0.7]
                                    ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [2, 1], #, 0, 0, 0],
                                        [1, 2] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 679, 789, 321, 654, 987]
        
        # Bien separadas las clases:
        graphs1 = [generate_synthetic_graph_csbm(n_nodes, n_classes, n_features, edge_prob_matrix, node_feature_means, n_infected, semillas[i], 0, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_100nodes_NoBalanced_1a100_clasesSep_0{j}.npz')
            np.savez(filename, x=graphs1[j].x, a=graphs1[j].a, y=graphs1[j].y)

        # Free memory
        del graphs1
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = 100
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_100nodes_NoBalanced_1a100_clasesSep_0{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes, n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output
    
  
###########################################################################################################################################################################################################


class synthetic_NoDir_100nodes_NoBalanced_1a100_clasesMezcl(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * 100 nodes each
      * 2 unbalanced classes: 1% for the minority class
      * "mixed" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  n_infected, semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            indices = [np.random.randint(0,n_nodes) for i in range(n_infected)]
            communities = np.array([int(j in indices) for j in range(n_nodes)])

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = 100
        n_features = 2
        n_classes = 2
        n_infected = 1
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                        [0.5, 0.5],
                                        [0.5, 0.5]
                                    ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [1, 1], #, 0, 0, 0],
                                        [1, 1] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 679, 789, 321, 654, 987]
        
        # Bien separadas las clases:
        graphs1 = [generate_synthetic_graph_csbm(n_nodes, n_classes, n_features, edge_prob_matrix, node_feature_means, n_infected, semillas[i], 0, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_100nodes_NoBalanced_1a100_clasesMezcl_2{j}.npz')
            np.savez(filename, x=graphs1[j].x, a=graphs1[j].a, y=graphs1[j].y)

        # Free memory
        del graphs1
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = 100
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_100nodes_NoBalanced_1a100_clasesMezcl_2{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes, n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output


###########################################################################################################################################################################################################


class synthetic_NoDir_100nodes_NoBalanced_2a100_clasesSep(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * 100 nodes each
      * 2 unbalanced classes: 2% for the minority class
      * "separated" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)
        
    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  n_infected, semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            indices = [np.random.randint(0,n_nodes) for i in range(n_infected)]
            communities = np.array([int(j in indices) for j in range(n_nodes)])

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = 100
        n_features = 2
        n_classes = 2
        n_infected = 2
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                        [0.8, 0.2],
                                        [0.3, 0.7]
                                    ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [2, 1], #, 0, 0, 0],
                                        [1, 2] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 679, 789, 321, 654, 987]
        
        # Bien separadas las clases:
        graphs1 = [generate_synthetic_graph_csbm(n_nodes, n_classes, n_features, edge_prob_matrix, node_feature_means, n_infected, semillas[i], 0, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_100nodes_NoBalanced_2a100_clasesSep_0{j}.npz')
            np.savez(filename, x=graphs1[j].x, a=graphs1[j].a, y=graphs1[j].y)

        # Free memory
        del graphs1
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = 100
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_100nodes_NoBalanced_2a100_clasesSep_0{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes, n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output


###########################################################################################################################################################################################################


class synthetic_NoDir_100nodes_NoBalanced_2a100_clasesMezcl(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * 100 nodes each
      * 2 unbalanced classes: 2% for the minority class
      * "mixed" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  n_infected, semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            indices = [np.random.randint(0,n_nodes) for i in range(n_infected)]
            communities = np.array([int(j in indices) for j in range(n_nodes)])

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = 100
        n_features = 2
        n_classes = 2
        n_infected = 2
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                    [0.5, 0.5],
                                    [0.5, 0.5]
                            ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [1, 1], #, 0, 0, 0],
                                        [1, 1] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 678, 789, 321, 654, 987]
        
        # Clases un poco más juntas:
        graphs2 = [generate_synthetic_graph_csbm(n_nodes, n_classes, n_features, edge_prob_matrix, node_feature_means, n_infected, semillas[i], 1, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_100nodes_NoBalanced_2a100_clasesMezcl_2{j}.npz')
            np.savez(filename, x=graphs2[j].x, a=graphs2[j].a, y=graphs2[j].y)

        # Free memory
        del graphs2
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = 100
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_100nodes_NoBalanced_2a100_clasesMezcl_2{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes, n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output


###########################################################################################################################################################################################################


class synthetic_NoDir_100nodes_NoBalanced_5a100_clasesSep(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * 100 nodes each
      * 2 unbalanced classes: 5% for the minority class
      * "separated" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  n_infected, semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            indices = [np.random.randint(0,n_nodes) for i in range(n_infected)]
            communities = np.array([int(j in indices) for j in range(n_nodes)])

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = 100
        n_features = 2
        n_classes = 2
        n_infected = 5
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                        [0.8, 0.2],
                                        [0.3, 0.7]
                                    ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [2, 1], #, 0, 0, 0],
                                        [1, 2] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 679, 789, 321, 654, 987]
        
        # Bien separadas las clases:
        graphs1 = [generate_synthetic_graph_csbm(n_nodes, n_classes, n_features, edge_prob_matrix, node_feature_means, n_infected, semillas[i], 0, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_100nodes_NoBalanced_5a100_clasesSep_0{j}.npz')
            np.savez(filename, x=graphs1[j].x, a=graphs1[j].a, y=graphs1[j].y)

        # Free memory
        del graphs1
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = 100
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_100nodes_NoBalanced_5a100_clasesSep_0{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes, n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output
        

###########################################################################################################################################################################################################


class synthetic_NoDir_100nodes_NoBalanced_5a100_clasesMezcl(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * 100 nodes each
      * 2 unbalanced classes: 5% for the minority class
      * "mixed" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)
        
    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  n_infected, semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            indices = [np.random.randint(0,n_nodes) for i in range(n_infected)]
            communities = np.array([int(j in indices) for j in range(n_nodes)])

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = 100
        n_features = 2
        n_classes = 2
        n_infected = 5
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                    [0.5, 0.5],
                                    [0.5, 0.5]
                            ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [1, 1], #, 0, 0, 0],
                                        [1, 1] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 678, 789, 321, 654, 987]
        
        # Clases un poco más juntas:
        graphs2 = [generate_synthetic_graph_csbm(n_nodes, n_classes, n_features, edge_prob_matrix, node_feature_means, n_infected, semillas[i], 1, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_100nodes_NoBalanced_5a100_clasesMezcl_2{j}.npz')
            np.savez(filename, x=graphs2[j].x, a=graphs2[j].a, y=graphs2[j].y)

        # Free memory
        del graphs2
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = 100
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_100nodes_NoBalanced_5a100_clasesMezcl_2{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes, n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output


###########################################################################################################################################################################################################


class synthetic_NoDir_100nodes_NoBalanced_10a100_clasesSep(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * 100 nodes each
      * 2 unbalanced classes: 10% for the minority class
      * "separated" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  n_infected, semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            indices = [np.random.randint(0,n_nodes) for i in range(n_infected)]
            communities = np.array([int(j in indices) for j in range(n_nodes)])

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = 100
        n_features = 2
        n_classes = 2
        n_infected = 10
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                        [0.8, 0.2],
                                        [0.3, 0.7]
                                    ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [2, 1], #, 0, 0, 0],
                                        [1, 2] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 679, 789, 321, 654, 987]
        
        # Bien separadas las clases:
        graphs1 = [generate_synthetic_graph_csbm(n_nodes, n_classes, n_features, edge_prob_matrix, node_feature_means, n_infected, semillas[i], 0, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_100nodes_NoBalanced_10a100_clasesSep_0{j}.npz')
            np.savez(filename, x=graphs1[j].x, a=graphs1[j].a, y=graphs1[j].y)

        # Free memory
        del graphs1
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = 100
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_100nodes_NoBalanced_10a100_clasesSep_0{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes, n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output


###########################################################################################################################################################################################################


class synthetic_NoDir_100nodes_NoBalanced_10a100_clasesMezcl(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * 100 nodes each
      * 2 unbalanced classes: 10% for the minority class
      * "mixed" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  n_infected, semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            indices = [np.random.randint(0,n_nodes) for i in range(n_infected)]
            communities = np.array([int(j in indices) for j in range(n_nodes)])

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = 100
        n_features = 2
        n_classes = 2
        n_infected = 10
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                    [0.5, 0.5],
                                    [0.5, 0.5]
                            ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [1, 1], #, 0, 0, 0],
                                        [1, 1] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 678, 789, 321, 654, 987]
        
        # Clases un poco más juntas:
        graphs2 = [generate_synthetic_graph_csbm(n_nodes, n_classes, n_features, edge_prob_matrix, node_feature_means, n_infected, semillas[i], 1, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_100nodes_NoBalanced_10a100_clasesMezcl_2{j}.npz')
            np.savez(filename, x=graphs2[j].x, a=graphs2[j].a, y=graphs2[j].y)

        # Free memory
        del graphs2
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = 100
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_100nodes_NoBalanced_10a100_clasesMezcl_2{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes, n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output


###########################################################################################################################################################################################################


class synthetic_NoDir_100nodes_NoBalanced_20a100_clasesSep(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * 100 nodes each
      * 2 unbalanced classes: 20% for the minority class
      * "separated" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  n_infected, semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            indices = [np.random.randint(0,n_nodes) for i in range(n_infected)]
            communities = np.array([int(j in indices) for j in range(n_nodes)])

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = 100
        n_features = 2
        n_classes = 2
        n_infected = 20
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                        [0.8, 0.2],
                                        [0.3, 0.7]
                                    ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [2, 1], #, 0, 0, 0],
                                        [1, 2] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 679, 789, 321, 654, 987]
        
        # Bien separadas las clases:
        graphs1 = [generate_synthetic_graph_csbm(n_nodes, n_classes, n_features, edge_prob_matrix, node_feature_means, n_infected, semillas[i], 0, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_100nodes_NoBalanced_20a100_clasesSep_0{j}.npz')
            np.savez(filename, x=graphs1[j].x, a=graphs1[j].a, y=graphs1[j].y)

        # Free memory
        del graphs1
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = 100
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_100nodes_NoBalanced_20a100_clasesSep_0{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes, n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output
        

###########################################################################################################################################################################################################


class synthetic_NoDir_100nodes_NoBalanced_20a100_clasesMezcl(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * 100 nodes each
      * 2 unbalanced classes: 20% for the minority class
      * "mixed" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  n_infected, semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            indices = [np.random.randint(0,n_nodes) for i in range(n_infected)]
            communities = np.array([int(j in indices) for j in range(n_nodes)])

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = 100
        n_features = 2
        n_classes = 2
        n_infected = 20
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                    [0.5, 0.5],
                                    [0.5, 0.5]
                            ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [1, 1], #, 0, 0, 0],
                                        [1, 1] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 678, 789, 321, 654, 987]
        
        # Clases un poco más juntas:
        graphs2 = [generate_synthetic_graph_csbm(n_nodes, n_classes, n_features, edge_prob_matrix, node_feature_means, n_infected, semillas[i], 1, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_100nodes_NoBalanced_20a100_clasesMezcl_2{j}.npz')
            np.savez(filename, x=graphs2[j].x, a=graphs2[j].a, y=graphs2[j].y)

        # Free memory
        del graphs2
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = 100
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_100nodes_NoBalanced_20a100_clasesMezcl_2{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes, n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output
    
    
###########################################################################################################################################################################################################


######################################################################################################################################
###                           No dirigido - diferente cantidad de nodos - clases NO balanceadas                                    ###
######################################################################################################################################


class synthetic_NoDir_diffSize_NoBalanced_1percent_clasesSep(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * number of nodes: 25,50,100,200,400,800,1600,3200,6400,12800
      * 2 unbalanced classes: 1% for the minority class
      * "separated" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  n_infected, semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            indices = [np.random.randint(0,n_nodes) for i in range(n_infected)]
            communities = np.array([int(j in indices) for j in range(n_nodes)])

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_features = 2
        n_classes = 2
        n_infected = [round((1/100)*n_nodes[i]) if (1/100)*n_nodes[i]>=1  else 1 for i in range(len(n_nodes))]
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                        [0.8, 0.2],
                                        [0.3, 0.7]
                                    ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [2, 1], #, 0, 0, 0],
                                        [1, 2] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 679, 789, 321, 654, 987]
        
        # Bien separadas las clases:
        graphs1 = [generate_synthetic_graph_csbm(n_nodes[i], n_classes, n_features, edge_prob_matrix, node_feature_means, n_infected[i], semillas[i], 0, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_NoBalanced_{n_infected[j]}a{n_nodes[j]}_clasesSep_0{j}.npz')
            np.savez(filename, x=graphs1[j].x, a=graphs1[j].a, y=graphs1[j].y)

        # Free memory
        del graphs1
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_infected = [round((1/100)*n_nodes[i]) if (1/100)*n_nodes[i]>=1  else 1 for i in range(len(n_nodes))]
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_NoBalanced_{n_infected[j]}a{n_nodes[j]}_clasesSep_0{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes[j], n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output


###########################################################################################################################################################################################################


class synthetic_NoDir_diffSize_NoBalanced_1percent_clasesMezcl(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * number of nodes: 25,50,100,200,400,800,1600,3200,6400,12800
      * 2 unbalanced classes: 1% for the minority class
      * "mixed" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  n_infected, semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            indices = [np.random.randint(0,n_nodes) for i in range(n_infected)]
            communities = np.array([int(j in indices) for j in range(n_nodes)])

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_features = 2
        n_classes = 2
        n_infected = [round((1/100)*n_nodes[i]) if (1/100)*n_nodes[i]>=1  else 1 for i in range(len(n_nodes))]
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                    [0.5, 0.5],
                                    [0.5, 0.5]
                            ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [1, 1], #, 0, 0, 0],
                                        [1, 1] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 678, 789, 321, 654, 987]
        
        # Clases un poco más juntas:
        graphs2 = [generate_synthetic_graph_csbm(n_nodes[i], n_classes, n_features, edge_prob_matrix, node_feature_means, n_infected[i], semillas[i], 1, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_NoBalanced_{n_infected[j]}a{n_nodes[j]}_clasesMezcl_2{j}.npz')
            np.savez(filename, x=graphs2[j].x, a=graphs2[j].a, y=graphs2[j].y)

        # Free memory
        del graphs2
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_infected = [round((1/100)*n_nodes[i]) if (1/100)*n_nodes[i]>=1  else 1 for i in range(len(n_nodes))]
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_NoBalanced_{n_infected[j]}a{n_nodes[j]}_clasesMezcl_2{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes[j], n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output


###########################################################################################################################################################################################################


class synthetic_NoDir_diffSize_NoBalanced_2percent_clasesSep(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * number of nodes: 25,50,100,200,400,800,1600,3200,6400,12800
      * 2 unbalanced classes: 2% for the minority class
      * "separated" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  n_infected, semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            indices = [np.random.randint(0,n_nodes) for i in range(n_infected)]
            communities = np.array([int(j in indices) for j in range(n_nodes)])

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_features = 2
        n_classes = 2
        n_infected = [round((2/100)*n_nodes[i]) if (2/100)*n_nodes[i]>=1  else 1 for i in range(len(n_nodes))]
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                        [0.8, 0.2],
                                        [0.3, 0.7]
                                    ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [2, 1], #, 0, 0, 0],
                                        [1, 2] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 679, 789, 321, 654, 987]
        
        # Bien separadas las clases:
        graphs1 = [generate_synthetic_graph_csbm(n_nodes[i], n_classes, n_features, edge_prob_matrix, node_feature_means, n_infected[i], semillas[i], 0, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_NoBalanced_{n_infected[j]}a{n_nodes[j]}_clasesSep_0{j}.npz')
            np.savez(filename, x=graphs1[j].x, a=graphs1[j].a, y=graphs1[j].y)

        # Free memory
        del graphs1
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_infected = [round((2/100)*n_nodes[i]) if (2/100)*n_nodes[i]>=1  else 1 for i in range(len(n_nodes))]
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_NoBalanced_{n_infected[j]}a{n_nodes[j]}_clasesSep_0{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes[j], n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output


###########################################################################################################################################################################################################


class synthetic_NoDir_diffSize_NoBalanced_2percent_clasesMezcl(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * number of nodes: 25,50,100,200,400,800,1600,3200,6400,12800
      * 2 unbalanced classes: 2% for the minority class
      * "mixed" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  n_infected, semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            indices = [np.random.randint(0,n_nodes) for i in range(n_infected)]
            communities = np.array([int(j in indices) for j in range(n_nodes)])

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_features = 2
        n_classes = 2
        n_infected = [round((2/100)*n_nodes[i]) if (2/100)*n_nodes[i]>=1  else 1 for i in range(len(n_nodes))]
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                    [0.5, 0.5],
                                    [0.5, 0.5]
                            ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [1, 1], #, 0, 0, 0],
                                        [1, 1] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 678, 789, 321, 654, 987]
        
        # Clases un poco más juntas:
        graphs2 = [generate_synthetic_graph_csbm(n_nodes[i], n_classes, n_features, edge_prob_matrix, node_feature_means, n_infected[i], semillas[i], 1, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_NoBalanced_{n_infected[j]}a{n_nodes[j]}_clasesMezcl_2{j}.npz')
            np.savez(filename, x=graphs2[j].x, a=graphs2[j].a, y=graphs2[j].y)

        # Free memory
        del graphs2
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_infected = [round((2/100)*n_nodes[i]) if (2/100)*n_nodes[i]>=1  else 1 for i in range(len(n_nodes))]
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_NoBalanced_{n_infected[j]}a{n_nodes[j]}_clasesMezcl_2{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes[j], n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output


###########################################################################################################################################################################################################


class synthetic_NoDir_diffSize_NoBalanced_5percent_clasesSep(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * number of nodes: 25,50,100,200,400,800,1600,3200,6400,12800
      * 2 unbalanced classes: 5% for the minority class
      * "separated" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  n_infected, semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            indices = [np.random.randint(0,n_nodes) for i in range(n_infected)]
            communities = np.array([int(j in indices) for j in range(n_nodes)])

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_features = 2
        n_classes = 2
        n_infected = [round((5/100)*n_nodes[i]) if (5/100)*n_nodes[i]>=1  else 1 for i in range(len(n_nodes))]
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                        [0.8, 0.2],
                                        [0.3, 0.7]
                                    ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [2, 1], #, 0, 0, 0],
                                        [1, 2] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 679, 789, 321, 654, 987]
        
        # Bien separadas las clases:
        graphs1 = [generate_synthetic_graph_csbm(n_nodes[i], n_classes, n_features, edge_prob_matrix, node_feature_means, n_infected[i], semillas[i], 0, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_NoBalanced_{n_infected[j]}a{n_nodes[j]}_clasesSep_0{j}.npz')
            np.savez(filename, x=graphs1[j].x, a=graphs1[j].a, y=graphs1[j].y)

        # Free memory
        del graphs1
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_infected = [round((5/100)*n_nodes[i]) if (5/100)*n_nodes[i]>=1  else 1 for i in range(len(n_nodes))]
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_NoBalanced_{n_infected[j]}a{n_nodes[j]}_clasesSep_0{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes[j], n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output


###########################################################################################################################################################################################################


class synthetic_NoDir_diffSize_NoBalanced_5percent_clasesMezcl(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * number of nodes: 25,50,100,200,400,800,1600,3200,6400,12800
      * 2 unbalanced classes: 5% for the minority class
      * "mixed" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  n_infected, semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            indices = [np.random.randint(0,n_nodes) for i in range(n_infected)]
            communities = np.array([int(j in indices) for j in range(n_nodes)])

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_features = 2
        n_classes = 2
        n_infected = [round((5/100)*n_nodes[i]) if (5/100)*n_nodes[i]>=1  else 1 for i in range(len(n_nodes))]
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                    [0.5, 0.5],
                                    [0.5, 0.5]
                            ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [1, 1], #, 0, 0, 0],
                                        [1, 1] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 678, 789, 321, 654, 987]
        
        # Clases un poco más juntas:
        graphs2 = [generate_synthetic_graph_csbm(n_nodes[i], n_classes, n_features, edge_prob_matrix, node_feature_means, n_infected[i], semillas[i], 1, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_NoBalanced_{n_infected[j]}a{n_nodes[j]}_clasesMezcl_2{j}.npz')
            np.savez(filename, x=graphs2[j].x, a=graphs2[j].a, y=graphs2[j].y)

        # Free memory
        del graphs2
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_infected = [round((5/100)*n_nodes[i]) if (5/100)*n_nodes[i]>=1  else 1 for i in range(len(n_nodes))]
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_NoBalanced_{n_infected[j]}a{n_nodes[j]}_clasesMezcl_2{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes[j], n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output


###########################################################################################################################################################################################################


class synthetic_NoDir_diffSize_NoBalanced_10percent_clasesSep(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * number of nodes: 25,50,100,200,400,800,1600,3200,6400,12800
      * 2 unbalanced classes: 10% for the minority class
      * "separated" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  n_infected, semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            indices = [np.random.randint(0,n_nodes) for i in range(n_infected)]
            communities = np.array([int(j in indices) for j in range(n_nodes)])

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_features = 2
        n_classes = 2
        n_infected = [round((10/100)*n_nodes[i]) if (10/100)*n_nodes[i]>=1  else 1 for i in range(len(n_nodes))]
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                        [0.8, 0.2],
                                        [0.3, 0.7]
                                    ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [2, 1], #, 0, 0, 0],
                                        [1, 2] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 679, 789, 321, 654, 987]
        
        # Bien separadas las clases:
        graphs1 = [generate_synthetic_graph_csbm(n_nodes[i], n_classes, n_features, edge_prob_matrix, node_feature_means, n_infected[i], semillas[i], 0, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_NoBalanced_{n_infected[j]}a{n_nodes[j]}_clasesSep_0{j}.npz')
            np.savez(filename, x=graphs1[j].x, a=graphs1[j].a, y=graphs1[j].y)

        # Free memory
        del graphs1
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_infected = [round((10/100)*n_nodes[i]) if (10/100)*n_nodes[i]>=1  else 1 for i in range(len(n_nodes))]
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_NoBalanced_{n_infected[j]}a{n_nodes[j]}_clasesSep_0{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes[j], n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output
        

###########################################################################################################################################################################################################


class synthetic_NoDir_diffSize_NoBalanced_10percent_clasesMezcl(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * number of nodes: 25,50,100,200,400,800,1600,3200,6400,12800
      * 2 unbalanced classes: 10% for the minority class
      * "mixed" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  n_infected, semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            indices = [np.random.randint(0,n_nodes) for i in range(n_infected)]
            communities = np.array([int(j in indices) for j in range(n_nodes)])

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_features = 2
        n_classes = 2
        n_infected = [round((10/100)*n_nodes[i]) if (10/100)*n_nodes[i]>=1  else 1 for i in range(len(n_nodes))]
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                    [0.5, 0.5],
                                    [0.5, 0.5]
                            ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [1, 1], #, 0, 0, 0],
                                        [1, 1] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 678, 789, 321, 654, 987]
        
        # Clases un poco más juntas:
        graphs2 = [generate_synthetic_graph_csbm(n_nodes[i], n_classes, n_features, edge_prob_matrix, node_feature_means, n_infected[i], semillas[i], 1, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_NoBalanced_{n_infected[j]}a{n_nodes[j]}_clasesMezcl_2{j}.npz')
            np.savez(filename, x=graphs2[j].x, a=graphs2[j].a, y=graphs2[j].y)

        # Free memory
        del graphs2
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_infected = [round((10/100)*n_nodes[i]) if (10/100)*n_nodes[i]>=1  else 1 for i in range(len(n_nodes))]
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_NoBalanced_{n_infected[j]}a{n_nodes[j]}_clasesMezcl_2{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes[j], n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output
        

###########################################################################################################################################################################################################


class synthetic_NoDir_diffSize_NoBalanced_20percent_clasesSep(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * number of nodes: 25,50,100,200,400,800,1600,3200,6400,12800
      * 2 unbalanced classes: 20% for the minority class
      * "separated" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  n_infected, semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            indices = [np.random.randint(0,n_nodes) for i in range(n_infected)]
            communities = np.array([int(j in indices) for j in range(n_nodes)])

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_features = 2
        n_classes = 2
        n_infected = [round((20/100)*n_nodes[i]) if (20/100)*n_nodes[i]>=1  else 1 for i in range(len(n_nodes))]
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                        [0.8, 0.2],
                                        [0.3, 0.7]
                                    ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [2, 1], #, 0, 0, 0],
                                        [1, 2] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 679, 789, 321, 654, 987]
        
        # Bien separadas las clases:
        graphs1 = [generate_synthetic_graph_csbm(n_nodes[i], n_classes, n_features, edge_prob_matrix, node_feature_means, n_infected[i], semillas[i], 0, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_NoBalanced_{n_infected[j]}a{n_nodes[j]}_clasesSep_0{j}.npz')
            np.savez(filename, x=graphs1[j].x, a=graphs1[j].a, y=graphs1[j].y)

        # Free memory
        del graphs1
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_infected = [round((20/100)*n_nodes[i]) if (20/100)*n_nodes[i]>=1  else 1 for i in range(len(n_nodes))]
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_NoBalanced_{n_infected[j]}a{n_nodes[j]}_clasesSep_0{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes[j], n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output


###########################################################################################################################################################################################################


class synthetic_NoDir_diffSize_NoBalanced_20percent_clasesMezcl(Dataset):
    """
    10 (ten) synthetic graphs:
      * undirected (i.e. symmetric adjacency matrix)
      * number of nodes: 25,50,100,200,400,800,1600,3200,6400,12800
      * 2 unbalanced classes: 20% for the minority class
      * "mixed" classes 
      
    **Arguments**
    
    - `flattened`: boolean, indicates whether the features will be flattened to 1.
    It is False by default.
    
    """   
    def __init__(self, flattened = False, **kwargs):
        self.flattened = flattened
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "syntheticGraphs", self.__class__.__name__)
    
    def download(self):
        os.makedirs(self.path)
        
        def generate_synthetic_graph_csbm(n_nodes, n_communities, n_features, edge_prob_matrix, node_feature_means,\
                                  n_infected, semilla, indice, feature_cov_matrix=None):
            # Assign nodes to communities
            np.random.seed(semilla*(indice+1))
            indices = [np.random.randint(0,n_nodes) for i in range(n_infected)]
            communities = np.array([int(j in indices) for j in range(n_nodes)])

            # Generate node features
            if feature_cov_matrix is None:
                feature_cov_matrix = np.eye(n_features)
            features = np.zeros((n_nodes, n_features))
            for k in range(n_communities):
                nodes_in_community = np.where(communities == k)[0]
                features[nodes_in_community] = np.random.multivariate_normal(node_feature_means[k], feature_cov_matrix,\
                                                                             len(nodes_in_community))

            # Compute community membership probabilities based on node features
            community_membership_probs = softmax(features @ node_feature_means.T, axis=1)

            # Generate edges based on community membership probabilities
            adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    if i == j:
                        continue
                    community_i = communities[i]
                    community_j = communities[j]
                    edge_prob = edge_prob_matrix[community_i, community_j] * community_membership_probs[i, community_j] * community_membership_probs[j, community_i]
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = np.random.binomial(1, edge_prob)

            labels = tf.keras.utils.to_categorical(communities)
            adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
            return Graph(x=features, a=adjacency_matrix, y=labels)

        
        n_graphs = 10
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_features = 2
        n_classes = 2
        n_infected = [round((20/100)*n_nodes[i]) if (20/100)*n_nodes[i]>=1  else 1 for i in range(len(n_nodes))]
        # Probability matrix for edges between communities
        edge_prob_matrix = np.array([
                                    [0.5, 0.5],
                                    [0.5, 0.5]
                            ])
        # Node feature means for each community
        node_feature_means = np.array([
                                        [1, 1], #, 0, 0, 0],
                                        [1, 1] #, 0, 0, 0]
                                    ])
        
        semillas = [123, 234, 345, 456, 567, 678, 789, 321, 654, 987]
        
        # Clases un poco más juntas:
        graphs2 = [generate_synthetic_graph_csbm(n_nodes[i], n_classes, n_features, edge_prob_matrix, node_feature_means, n_infected[i], semillas[i], 1, feature_cov_matrix=None) for i in range(n_graphs)]
        for j in range(10):
            filename = os.path.join(self.path, f'graph_NoDir_NoBalanced_{n_infected[j]}a{n_nodes[j]}_clasesMezcl_2{j}.npz')
            np.savez(filename, x=graphs2[j].x, a=graphs2[j].a, y=graphs2[j].y)

        # Free memory
        del graphs2
        gc.collect()


    def read(self):
        output = []
        
        n_nodes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
        n_infected = [round((20/100)*n_nodes[i]) if (20/100)*n_nodes[i]>=1  else 1 for i in range(len(n_nodes))]
        n_features = 2
        
        for j in range(10):
            data = np.load(os.path.join(self.path, f'graph_NoDir_NoBalanced_{n_infected[j]}a{n_nodes[j]}_clasesMezcl_2{j}.npz'), allow_pickle=True)
            if self.flattened:
                x_features = np.ones((n_nodes[j], n_features))
            else:
                x_features = data['x']
            output.append(
                Graph(x=x_features, a=data['a'][()], y=data['y']) # también puede ser a=data['a'].item()
            )

        return output
        

