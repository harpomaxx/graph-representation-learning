import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from synthetic_graphs import CacicSpringer_SyntheticGraphs
#from training_and_evaluation_GCN_cacicspringer import instancia
import sys

PATH_RDOS = str(sys.argv[1])

CLASE = eval(sys.argv[2])
print("CLASE = ", CLASE)

NUM_GRAPHS = int(sys.argv[3])
print("NUM_GRAPHS = ", NUM_GRAPHS)

NUM_NODES = int(sys.argv[4])
print("NUM_NODES = ", NUM_NODES)

NUM_COMMUNITIES = int(sys.argv[5])
print("NUM_COMMUNITIES = ", NUM_COMMUNITIES)

NUM_FEATURES = int(sys.argv[6])
print("NUM_FEATURES = ", NUM_FEATURES)

configuration = sys.argv[7]
config_str = list(map(str, str(sys.argv[7]).split('-')))
print("config_str = ", config_str)

feature_cov = str(sys.argv[8])
if feature_cov == "None":
    FEATURE_COV_MATRIX = None
else:
    FEATURE_COV_MATRIX = np.reshape(list(map(int, feature_cov.split(','))), (NUM_FEATURES, NUM_FEATURES))
print("FEATURE_COV_MATRIX = ", FEATURE_COV_MATRIX)

DIRECTED = eval(sys.argv[9])
print("DIRECTED = ", DIRECTED)

class1_percent_str = str(sys.argv[10])
if class1_percent_str == "None":
    CLASS1_PERCENT = None
else:
    CLASS1_PERCENT = eval(class1_percent_str)
print("CLASS1_PERCENT = ", CLASS1_PERCENT)

seed_aux = list(map(int, str(sys.argv[11]).split('-')))
# para generar 90 semillas diferentes para armar 100 grafos, a partir de las 10 semillas dadas:
SEED = seed_aux.copy()
for i in range(6):
    seed_aux = list(np.array(seed_aux)*10)
    for j in range(10):
        SEED.append(int(seed_aux[j]))


for i in range(3):
    seed_aux = list(np.array(seed_aux)+100)
    for j in range(10):
        SEED.append(int(seed_aux[j]))
print("SEED = ", SEED)

FLATTENED = eval(sys.argv[12])
print("FLATTENED = ", FLATTENED)

ADJNUL = eval(sys.argv[13])
print("ADJNUL = ", ADJNUL)

SYMMETRIC_ADJACENCY = eval(sys.argv[14])
print("SYMMETRIC_ADJACENCY = ", SYMMETRIC_ADJACENCY)

PREPROC_ADJACENCY = eval(sys.argv[15])
print("PREPROC_ADJACENCY = ", PREPROC_ADJACENCY)

PREPROC_FEATURES = eval(sys.argv[16])
print("PREPROC_FEATURES = ", PREPROC_FEATURES)


def deco(string_list):
    edge = string_list[0]
    mean = string_list[1]
    if edge == "e3": #A
        edge_prob_matrix = np.array([[0.9,0.1], [0.2,0.8]])
    elif edge == "e2": #B
        edge_prob_matrix = np.array([[0.8,0.2], [0.3,0.7]])
    elif edge == "e1": #C
        edge_prob_matrix = np.array([[0.6,0.4], [0.4,0.6]])
    elif edge == "e0": #D
        edge_prob_matrix=np.array([[0.5,0.5], [0.5,0.5]])
    else:
        raise ValueError("no valid entry")
    if mean == "f3": #I
        features_mean_matrix = np.array([[3,0], [0,3]])
    elif mean == "f2": #II
        features_mean_matrix = np.array([[2,1], [1,2]])
    elif mean == "f1": #III
        features_mean_matrix = np.array([[1.5,1], [1,1.5]])
    elif mean == "f0": #IV
        features_mean_matrix = np.array([[1,1], [1,1]])
    else:
        raise ValueError("no valid entry")
    return edge_prob_matrix, features_mean_matrix


def instancia(clase, num_graphs, num_nodes, num_communities, num_features, edge_prob_matrix, \
                node_feature_means, feature_cov_matrix=None, directed=False, class1_percent=None, \
                seed=None, flattened=False, adjnul=False, symmetricAdjacency=False, \
                preprocAdjacency=True, preprocFeatures=True):
    """
    Función para instanciar la clase que define los diferentes conjuntos de grafos sintéticos
    """
    if preprocAdjacency and preprocFeatures:
        inst = clase(num_graphs, num_nodes, num_communities, num_features, edge_prob_matrix, \
                node_feature_means, feature_cov_matrix, directed, class1_percent, seed, flattened, adjnul, \
                     transforms=[GCNConv_preprocess_adjacencyMatrix(GCNConv, symmetric=symmetricAdjacency), \
                                 GCNConv_preprocess_features()])
    elif preprocAdjacency and ~preprocFeatures:
        inst = clase(num_graphs, num_nodes, num_communities, num_features, edge_prob_matrix, \
                node_feature_means, feature_cov_matrix, directed, class1_percent, seed, flattened, adjnul, \
                     transforms=[GCNConv_preprocess_adjacencyMatrix(GCNConv, symmetric=symmetricAdjacency)])
    elif ~preprocAdjacency and preprocFeatures:
        inst = clase(num_graphs, num_nodes, num_communities, num_features, edge_prob_matrix, \
                node_feature_means, feature_cov_matrix, directed, class1_percent, seed, flattened, adjnul, \
                     transforms=[GCNConv_preprocess_features()])
    else:
        inst = clase(num_graphs, num_nodes, num_communities, num_features, edge_prob_matrix, \
                node_feature_means, feature_cov_matrix, directed, class1_percent, seed, flattened, adjnul)
    return inst


def plot_graph_dir(G, configuration, class1_percent, communities=None):
    pos = nx.spring_layout(G, seed=42)
    
    if communities is not None:
        # Assign colors to nodes based on their communities
        colors = ['white' if community == 0 else 'black' for community in communities]
    else:
        colors = 'black'
    
    plt.grid(False)
    nx.draw_networkx(G, pos, arrows=True, node_color=colors, edgecolors='black', with_labels=True, font_color="grey")

    plt.savefig(os.path.join(PATH_RDOS, f'graph_{configuration}_{class1_percent}.eps'), format="eps", dpi=800)


EDGE_PROB_MATRIX, NODE_FEATURES_MEAN = deco(config_str)
print("EDGE_PROB_MATRIX = ", EDGE_PROB_MATRIX)
print("NODE_FEATURES_MEAN = ", NODE_FEATURES_MEAN)


# Se carga el conjunto de grafos
dataset = instancia(CLASE, NUM_GRAPHS, NUM_NODES, NUM_COMMUNITIES, NUM_FEATURES, EDGE_PROB_MATRIX, \
                    NODE_FEATURES_MEAN, FEATURE_COV_MATRIX, DIRECTED, CLASS1_PERCENT, SEED, \
                    FLATTENED, ADJNUL, SYMMETRIC_ADJACENCY, PREPROC_ADJACENCY, PREPROC_FEATURES)


numr = np.random.randint(NUM_GRAPHS)
G = dataset[numr]
nxG = nx.from_numpy_array(G.a, create_using=nx.DiGraph)
labels = np.argmax(G.y, axis=1)
plot_graph_dir(nxG, configuration, CLASS1_PERCENT, labels)

