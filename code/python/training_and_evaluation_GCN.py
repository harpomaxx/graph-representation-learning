import networkx as nx

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from spektral.data import Dataset, Graph, SingleLoader, BatchLoader, DisjointLoader
from spektral.layers import GCNConv
from spektral.utils import gcn_filter
from spektral.models.gcn import GCN 
from spektral.datasets.utils import DATASET_FOLDER

from scipy import sparse
from scipy.special import softmax

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score

from keras.callbacks import EarlyStopping, ModelCheckpoint

import datasetsDirigidos
from preprocess_adjacencyMatrix import GCNConv_preprocess_adjacencyMatrix
from preprocess_features import GCNConv_preprocess_features

import sys

PATH_RDO = str(sys.argv[1]) 
CLASE = eval(sys.argv[2])
FLATTENED = eval(sys.argv[3])
SYMMETRIC_ADJACENCY = eval(sys.argv[4])
PREPROC_ADJACENCY = eval(sys.argv[5])
PREPROC_FEATURES = eval(sys.argv[6])

if FLATTENED:
    PATH_RDOS = os.path.join(PATH_RDO, "flattened")
else: 
    PATH_RDOS = os.path.join(PATH_RDO, "NO_flattened")


# Limiting GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



#class SyntheticDataset(Dataset):
#    def __init__(self, graphs=None, **kwargs):
#        self.graphs = graphs
#        super().__init__(**kwargs)
#
#    def read(self):
#        return self.graphs


def instancia(clase, flatten=False, symmetricAdjacency=False, preprocAdjacency=True, preprocFeatures=True):
    if preprocAdjacency and preprocFeatures:
        inst = clase(flattened=flatten, transforms=[GCNConv_preprocess_adjacencyMatrix(GCNConv, symmetric=symmetricAdjacency), GCNConv_preprocess_features()])
    elif preprocAdjacency and ~preprocFeatures:
        inst = clase(flattened=flatten, transforms=[GCNConv_preprocess_adjacencyMatrix(GCNConv, symmetric=symmetricAdjacency)])
    elif ~preprocAdjacency and preprocFeatures:
        inst = clase(flattened=flatten, transforms=[GCNConv_preprocess_features()])
    else:
        inst = clase(flattened=flatten)
    return inst


def predicciones(loader, path, nombre):
    for _ in range(loader.steps_per_epoch):
        metricas = open(path, "a")
        inputs,target = loader.__next__()
        y_prediction = model(inputs, training=False)
        y_prediction = np.argmax(np.vstack(y_prediction), axis = 1)
        y_true=np.argmax(np.vstack(target), axis=1)
        prediccion=pd.DataFrame({"true_label":y_true, "prediction":y_prediction})
        prediccion.to_csv(os.path.join(prediccionesDirectorio,f'prediccion_{str(nombre)}.csv'),index = None)
        #Create confusion matrix and normalizes it over predicted (columns)
        result = tf.math.confusion_matrix(y_true, y_prediction, num_classes=2) 
        metricas.write(f'confusion matrix:\n {str(result)}\n')

        # confusion_matrix = [[TP, FN],
        #                     [FP, TN]]
        TP = result[0,0].numpy()
        FN = result[0,1].numpy()
        FP = result[1,0].numpy()
        TN = result[1,1].numpy()

        accuracy = (TP+TN)/(TP+FP+FN+TN)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        especificity = TN/(TN+FP)
        f1 = (2*precision*recall)/(precision+recall)
        auc_score = roc_auc_score(y_true, y_prediction)
        
        metricas.write(f'accuracy: {str(accuracy)}\n')
        metricas.write(f'precision: {str(precision)}\n')
        metricas.write(f'recall: {str(recall)}\n')
        metricas.write(f'especificity: {str(especificity)}\n')
        metricas.write(f'f1: {str(f1)}\n')
        metricas.write(f'auc_score: {str(auc_score)}\n')
        metricas.write("==============================\n")
        
        metricas.close()


###################################################################################

dataset = instancia(CLASE, flatten=FLATTENED, symmetricAdjacency=SYMMETRIC_ADJACENCY, preprocAdjacency=PREPROC_ADJACENCY, preprocFeatures=PREPROC_FEATURES)

guardarModelo = np.random.randint(10)
modeloDirectorio = os.path.join(PATH_RDOS,f'prueba_0{guardarModelo}/modelo')
os.makedirs(modeloDirectorio, exist_ok = True)


for i in range(10):
    graficasDirectorio = os.path.join(PATH_RDOS,f'prueba_0{i}/graficas')
    prediccionesDirectorio = os.path.join(PATH_RDOS,f'prueba_0{i}/predicciones')
    metricasDirectorio = os.path.join(PATH_RDOS,f'prueba_0{i}/metricas')
    
    os.makedirs(graficasDirectorio, exist_ok = True)
    os.makedirs(prediccionesDirectorio, exist_ok = True)
    os.makedirs(metricasDirectorio, exist_ok = True)
    
    tf.keras.backend.clear_session() ### NO SE SI ESTO ESTA ANDANDO REALMENTE

    indices = np.concatenate((np.arange(i), np.arange(i+1,10)))
    graphs4train = dataset[indices]
    test_dataset = dataset[i:i+1]
    
    idxs = np.random.permutation(len(graphs4train))
    split_va = int(0.9 * len(graphs4train))
    idx_tr, idx_va = np.split(idxs, [split_va])
    train_dataset = graphs4train[idx_tr]
    val_dataset = graphs4train[idx_va]
    
    batch_size = 1
    n_epochs = 200
    # Create data loaders for training and testing data
    train_loader = BatchLoader(train_dataset, batch_size=batch_size, epochs=n_epochs, shuffle=False, node_level=True)   ####### ATENCION: mask y shuffle
    val_loader = SingleLoader(val_dataset, epochs=n_epochs)
    test_loader = SingleLoader(test_dataset, epochs=n_epochs)

    n_classes=2
    model = GCN(n_labels=n_classes)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.01), loss="binary_crossentropy", metrics=["accuracy"])

    # Define early stopping to prevent overfitting   
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

    # Train the model
    history = model.fit(
        train_loader.load(),
        steps_per_epoch=train_loader.steps_per_epoch,
        epochs=n_epochs,
        validation_data=val_loader.load(),
        validation_steps=val_loader.steps_per_epoch,
        callbacks=callbacks_list                            
    )
    
    ## GRAFICAR
    res=pd.DataFrame(history.history)
    # Add row index as a new column
    res.reset_index(inplace=True)
    # Rename the new column to 'row_id'
    res.rename(columns={'index': 'epoch'}, inplace=True)
    res.to_csv(os.path.join(graficasDirectorio,"epochsResults.csv"),index = None)          
    
    sns.set_theme(style="whitegrid")
    line1 = sns.lineplot(x="epoch", y='loss', data=res, label='Training Loss')
    line2 = sns.lineplot(x="epoch", y='val_loss', data=res, label='Test Loss')
    # Add points to each observation
    scatter1 = sns.scatterplot(x="epoch", y='loss', data=res, marker='o', color='skyblue')
    scatter2 = sns.scatterplot(x="epoch", y='val_loss', data=res, marker='o', color='orange')
    # Change the y-axis label
    plt.ylabel("Loss Value")
    # Create a legend for the lines
    plt.legend()
    # Show the plot
    plt.savefig(os.path.join(graficasDirectorio,"loss.png"))                             
    plt.clf()
    
    sns.set_theme(style="whitegrid")
    line1 = sns.lineplot(x="epoch", y="accuracy", data=res, label='Training Accuracy')
    line2 = sns.lineplot(x="epoch", y="val_accuracy", data=res, label='Test Accuracy')
    # Add points to each observation
    scatter1 = sns.scatterplot(x="epoch", y="accuracy", data=res, marker='o', color='skyblue')
    scatter2 = sns.scatterplot(x="epoch", y="val_accuracy", data=res, marker='o', color='orange')
    # Change the y-axis label
    plt.ylabel("Accuracy Value")
    # Create a legend for the lines
    plt.legend()
    # Show the plot
    plt.savefig(os.path.join(graficasDirectorio,"accuracy.png"))              
    plt.clf()
    
    # PREDICCION
    loaders = [test_loader, val_loader, train_loader]
    names = ["test", "val", "train"]
    for j in range(len(loaders)):
        predicciones(loaders[j], os.path.join(metricasDirectorio,f'metricas_{str(names[j])}.txt'), names[j])
        
