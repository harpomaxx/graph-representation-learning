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

import datasetsDirigidos as dirigidos
from preprocess_adjacencyMatrix import GCNConv_preprocess_adjacencyMatrix
from preprocess_features import GCNConv_preprocess_features

import sys

#PATH_RDOS = ########## COMPLETAR CON SYS

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



class SyntheticDataset(Dataset):
    def __init__(self, graphs=None, **kwargs):
        self.graphs = graphs
        super().__init__(**kwargs)

    def read(self):
        return self.graphs


dataset = dirigidos.synthetic_Dir_100nodes_balanced_clasesSep()#transforms=[preprocess_adjacencyMatrix(GCNConv,symmetric=False), preprocess_features()])



guardarModelo = np.random.randint(10)



for i in range(10):
    tf.keras.backend.clear_session() ### NO SE SI ESTO ESTA ANDANDO REALMENTE

    indices = np.concatenate((np.arange(i), np.arange(i+1,10)))
    graphs4train = dataset[indices]
    test_graphs = dataset[i:i+1]
    
    train_graphs, val_graphs = train_test_split(graphs4train, test_size=0.1, random_state=42)
    train_dataset = SyntheticDataset(train_graphs)
    val_dataset = SyntheticDataset(val_graphs)
    test_dataset = SyntheticDataset(test_graphs)
    
    batch_size = 1
    # Create data loaders for training and testing data
    train_loader = BatchLoader(train_dataset, batch_size=batch_size)
    val_loader = SingleLoader(val_dataset)
    test_loader = SingleLoader(test_dataset)

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
                filepath=f'resultados/Dir_100nodes_balanced_clasesSep/prueba_0{i}/modelo/',
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
        epochs=200,
        validation_data=val_loader.load(),
        validation_steps=val_loader.steps_per_epoch,
        callbacks=callbacks_list                             ### ARMAR UNA CARPETA PARA GUARDAR EL MODELO
    )
    
    ## GRAFICAR
    res=pd.DataFrame(history.history)

    # Add row index as a new column
    res.reset_index(inplace=True)

    # Rename the new column to 'row_id'
    res.rename(columns={'index': 'epoch'}, inplace=True)
    res.to_csv(f'resultados/Dir_100nodes_balanced_clasesSep/prueba_0{i}/graficas/epochsResults.csv')                ### PODRIA GUARDAR ESTA INFO EN csv

    #history.history['accuracy']
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
    plt.savefig(f'resultados/Dir_100nodes_balanced_clasesSep/prueba_0{i}/graficas/loss.png')                              ### GUARDAR IMAGEN

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
    plt.savefig(f'resultados/Dir_100nodes_balanced_clasesSep/prueba_0{i}/graficas/accuracy.png')               ### GUARDAR LA IMAGEN

    # PREDICCION EN TEST
    tutti = [test_loader, val_loader, train_loader]
    for _ in range(test_loader.steps_per_epoch):
        inputs,target = test_loader.__next__()
        y_prediction = model(inputs, training=False)
        y_prediction = np.argmax(np.vstack(y_prediction), axis = 1)
        y_test=np.argmax(np.vstack(target), axis=1)
        prediccion=pd.DataFrame({"true_label":y_test, "prediction":y_prediction})
        prediccion.to_csv(f'resultados/Dir_100nodes_balanced_clasesSep/prueba_0{i}/predicciones/prediccion.csv')
        #Create confusion matrix and normalizes it over predicted (columns)
        result = tf.math.confusion_matrix(y_test, y_prediction, num_classes=2) 
        print(result)

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
        auc_score = roc_auc_score(y_test, y_prediction)

        print("accuracy (exactitud) = ", accuracy) # cantidad de predicciones positivas que fueron correctas
        print("precision = ", precision) # proporcion de casos positivos detectados
        print("recall = ", recall) # casos positivos que fueron correctamente identificadas por el algoritmo
        print("especificity = ", especificity) # casos negativos que el algoritmo ha clasificado correctamente
        print("f1 = ", f1)
        print("auc_score = ", auc_score)
        print("==============================")


