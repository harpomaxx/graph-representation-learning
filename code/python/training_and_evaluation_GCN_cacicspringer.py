import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from spektral.data import SingleLoader, BatchLoader
from spektral.layers import GCNConv
from spektral.models.gcn import GCN 

from sklearn.metrics import confusion_matrix, roc_auc_score

import time
import sys

# Módulos armados para definir los grafos y realizar el pre-procesamiento adecuado
from synthetic_graphs import CacicSpringer_SyntheticGraphs
from preprocess_adjacencyMatrix import GCNConv_preprocess_adjacencyMatrix
from preprocess_features import GCNConv_preprocess_features


start = time.time()

# Variables pasadas por argumento al correr el script                 
PATH_RDOS = str(sys.argv[1]) 
print("PATH_RDOS = ", PATH_RDOS)

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

    


# Limitar memoria de la GPU a utilizar
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)


# Funciones a usar:
def deco(string_list):
    edge = string_list[0]
    mean = string_list[1]
    if edge == "A":
        edge_prob_matrix = np.array([[0.9,0.1], [0.2,0.8]])
    elif edge == "B":
        edge_prob_matrix = np.array([[0.8,0.2], [0.3,0.7]])
    elif edge == "C":
        edge_prob_matrix = np.array([[0.6,0.4], [0.4,0.6]])
    elif edge == "D":
        edge_prob_matrix=np.array([[0.5,0.5], [0.5,0.5]])
    else:
        raise ValueError("no valid entry")
    if mean == "I":
        features_mean_matrix = np.array([[3,0], [0,3]])
    elif mean == "II":
        features_mean_matrix = np.array([[2,1], [1,2]])
    elif mean == "III":
        features_mean_matrix = np.array([[1.5,1], [1,1.5]])
    elif mean == "IV":
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



def predicciones(loader, nombre):
    """
    Calcula las predicciones y métricas de interés
    """
    TP_list = []
    FN_list = []
    FP_list = []
    TN_list = []
    acc = []
    prec = []
    rec = []
    esp = []
    F1 = []
    auc = []
    
    for k in range(loader.steps_per_epoch):
        inputs,target = loader.__next__()
        y_prediction = model(inputs, training=False)
        y_prediction = np.argmax(np.vstack(y_prediction), axis = 1)
        y_true=np.argmax(np.vstack(target), axis=1)
        prediccion=pd.DataFrame({"true_label":y_true, "prediction":y_prediction})
        prediccion.to_csv(os.path.join(prediccionesDirectorio,f'prediccion_{str(nombre)}_{k:03d}.csv'), index = None)
        
        #Create confusion matrix and normalizes it over predicted (columns)
        result = tf.math.confusion_matrix(y_true, y_prediction, num_classes=NUM_COMMUNITIES) 

        # confusion_matrix = [[TP, FN],
        #                     [FP, TN]]
        TP = result[0,0].numpy()
        FN = result[0,1].numpy()
        FP = result[1,0].numpy()
        TN = result[1,1].numpy()

        accuracy = (TP+TN)/(TP+FP+FN+TN)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        specificity = TN/(TN+FP)
        f1 = (2*precision*recall)/(precision+recall)
        auc_score = roc_auc_score(y_true, y_prediction)
        
        TP_list.append(TP)
        FN_list.append(FN)
        FP_list.append(FP)
        TN_list.append(TN)
        acc.append(accuracy)
        prec.append(precision)
        rec.append(recall)
        esp.append(specificity)
        F1.append(f1)
        auc.append(auc_score)
        
    df = pd.DataFrame({"TP":TP_list, "FN":FN_list, "FP":FP_list, "TN":TN_list, "accuracy":acc, "precision":prec, "recall":rec, "specificity":esp, "f1":F1, "auc_score":auc})
    df.to_csv(os.path.join(metricasDirectorio, f'metricas_{str(nombre)}.csv'), index = None)



###################################################################################

EDGE_PROB_MATRIX, NODE_FEATURES_MEAN = deco(config_str)
print("EDGE_PROB_MATRIX = ", EDGE_PROB_MATRIX)
print("NODE_FEATURES_MEAN = ", NODE_FEATURES_MEAN)


# Se carga el conjunto de grafos
dataset = instancia(CLASE, NUM_GRAPHS, NUM_NODES, NUM_COMMUNITIES, NUM_FEATURES, EDGE_PROB_MATRIX, \
                    NODE_FEATURES_MEAN, FEATURE_COV_MATRIX, DIRECTED, CLASS1_PERCENT, SEED, \
                    FLATTENED, ADJNUL, SYMMETRIC_ADJACENCY, PREPROC_ADJACENCY, PREPROC_FEATURES)


# Se almacena el modelo para una sola de las corridas, elegida al azar
guardarModelo = np.random.randint(NUM_GRAPHS)
modeloDirectorio = os.path.join(PATH_RDOS,f'prueba_{guardarModelo:03d}/modelo')
os.makedirs(modeloDirectorio, exist_ok = True)


# k-fold, con k=NUM_GRAPHS
for i in range(NUM_GRAPHS):
    graficasDirectorio = os.path.join(PATH_RDOS,f'prueba_{i:03d}/graficas')
    prediccionesDirectorio = os.path.join(PATH_RDOS,f'prueba_{i:03d}/predicciones')
    metricasDirectorio = os.path.join(PATH_RDOS,f'prueba_{i:03d}/metricas')
    
    os.makedirs(graficasDirectorio, exist_ok = True)
    os.makedirs(prediccionesDirectorio, exist_ok = True)
    os.makedirs(metricasDirectorio, exist_ok = True)
    
    tf.keras.backend.clear_session() 

    indices = np.concatenate((np.arange(i), np.arange(i+1, NUM_GRAPHS)))
    graphs4train = dataset[indices]
    test_dataset = dataset[i:i+1]
    
    idxs = np.random.permutation(len(graphs4train))
    split_va = int(0.99 * len(graphs4train))        # para 100 grafos al multiplicar por 0.99 me aseguro dejar 1 para validacion (para 10 grafos multiplicar por 0.9)
    idx_tr, idx_va = np.split(idxs, [split_va])
    train_dataset = graphs4train[idx_tr]
    val_dataset = graphs4train[idx_va]
    
    batch_size = 1
    n_epochs = 200
    
    # Se crean data loaders
    train_loader = BatchLoader(train_dataset, batch_size=batch_size, epochs=n_epochs, shuffle=False, node_level=True)   
    val_loader = SingleLoader(val_dataset, epochs=n_epochs)
    test_loader = SingleLoader(test_dataset, epochs=n_epochs)

    n_classes = NUM_COMMUNITIES
    model = GCN(n_labels=n_classes, channels=16)

    # Compila el model
    model.compile(optimizer=Adam(learning_rate=0.01), loss="binary_crossentropy", metrics=["accuracy"])

    # Se define early stopping para prevenir overfitting   
    if i==guardarModelo:
        callbacks_list = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                verbose=1
                ),
            ModelCheckpoint(
                filepath=modeloDirectorio,
                monitor="val_loss",
                save_best_only=True,
                )
        ]
    else:
        callbacks_list = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                verbose=1
                )
        ]

    # Entrenamiento
    history = model.fit(
        train_loader.load(),
        steps_per_epoch=train_loader.steps_per_epoch,
        epochs=n_epochs,
        validation_data=val_loader.load(),
        validation_steps=val_loader.steps_per_epoch,
        callbacks=callbacks_list                            
    )
    
    ## GRAFICAR
    res = pd.DataFrame(history.history)
    res.reset_index(inplace=True)
    res.rename(columns={'index': 'epoch'}, inplace=True)
    res.to_csv(os.path.join(graficasDirectorio,f'epochsResults.csv'),index = None)          
    
    sns.set_theme(style="whitegrid")
    line1 = sns.lineplot(x="epoch", y='loss', data=res, label='Training Loss')
    line2 = sns.lineplot(x="epoch", y='val_loss', data=res, label='Test Loss')
    scatter1 = sns.scatterplot(x="epoch", y='loss', data=res, marker='o', color='skyblue')
    scatter2 = sns.scatterplot(x="epoch", y='val_loss', data=res, marker='o', color='orange')
    plt.ylabel("Loss Value")
    plt.legend()
    plt.savefig(os.path.join(graficasDirectorio,f'loss.eps'), format='eps', dpi=800)                             
    plt.clf()
    
    sns.set_theme(style="whitegrid")
    line1 = sns.lineplot(x="epoch", y="accuracy", data=res, label='Training Accuracy')
    line2 = sns.lineplot(x="epoch", y="val_accuracy", data=res, label='Test Accuracy')
    scatter1 = sns.scatterplot(x="epoch", y="accuracy", data=res, marker='o', color='skyblue')
    scatter2 = sns.scatterplot(x="epoch", y="val_accuracy", data=res, marker='o', color='orange')
    plt.ylabel("Accuracy Value")
    plt.legend()
    plt.savefig(os.path.join(graficasDirectorio,f'accuracy.eps'), format='eps', dpi=800)              
    plt.clf()
    
    # PREDICCION
    loaders = [test_loader, val_loader, train_loader]
    names = ["test", "val", "train"]
    for j in range(len(loaders)):
        predicciones(loaders[j], names[j])


###############################################################################################

# Se une los resultados obtenidos en cada fold, en un único archivo

joinDirectorio = os.path.join(PATH_RDOS, "join")
os.makedirs(joinDirectorio, exist_ok = True)

names = ["test", "val", "train"]

for j in range(len(names)):
    df1 = pd.read_csv(os.path.join(PATH_RDOS, f'prueba_000/metricas/metricas_{names[j]}.csv'))

    for i in range(1, NUM_GRAPHS):
        ARCHIVO = os.path.join(PATH_RDOS, f'prueba_{i:03d}/metricas/metricas_{names[j]}.csv')
        p = pd.read_csv(ARCHIVO)
        df1 = pd.concat([df1, p], ignore_index=True)
    
    df1.to_csv(os.path.join(joinDirectorio, f'join_{names[j]}.csv'), index = None)
    
    #sns.set_theme(style="whitegrid")
    #boxplot1 = sns.boxplot(data=df1.iloc[:,4:5]) 
    #plt.savefig(os.path.join(joinDirectorio,f'accuracy_{names[j]}.eps'), format='eps', dpi=800)                             
    #plt.clf()
    
    #sns.set_theme(style="whitegrid")
    #boxplot1 = sns.boxplot(data=df1.iloc[:,7:8]) 
    #plt.savefig(os.path.join(joinDirectorio,f'specificity_{names[j]}.eps'), format='eps', dpi=800)                             
    #plt.clf()
    
    #sns.set_theme(style="whitegrid")
    #boxplot2 = sns.boxplot(data=df1.iloc[:,9:10]) 
    #plt.savefig(os.path.join(joinDirectorio,f'auc_{names[j]}.eps'), format='eps', dpi=800)                             
    #plt.clf()
    
################################################################################################

end = time.time()
print("\nTime = ", end-start)

    
