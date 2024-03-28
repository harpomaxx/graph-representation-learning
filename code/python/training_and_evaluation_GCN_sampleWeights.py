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
from spektral.utils import gcn_filter, degree_matrix
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

PATH_RDO = os.path.join(str(sys.argv[1]), "resultados/sample_weights/Dir_100nodes_NoBalanced_20a100_clasesSep")
SAMPLE_WEIGHT = eval(sys.argv[2])

if SAMPLE_WEIGHT:
    PATH_RDOS = os.path.join(PATH_RDO, "CON_sample_weight")
    NOMBRE_PRUEBA = str(sys.argv[2]) + "_sampleWeight"
else:
    PATH_RDOS = os.path.join(PATH_RDO, "SIN_sample_weight")
    NOMBRE_PRUEBA = str(sys.argv[2]) + "_SINsampleWeight"
    


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


def predicciones(loader, nombre):
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
        prediccion.to_csv(os.path.join(prediccionesDirectorio,f'{str(NOMBRE_PRUEBA)}_prediccion_{str(nombre)}_0{k}.csv'),index = None)
        
        #Create confusion matrix and normalizes it over predicted (columns)
        result = tf.math.confusion_matrix(y_true, y_prediction, num_classes=2) 

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
        
        TP_list.append(TP)
        FN_list.append(FN)
        FP_list.append(FP)
        TN_list.append(TN)
        acc.append(accuracy)
        prec.append(precision)
        rec.append(recall)
        esp.append(especificity)
        F1.append(f1)
        auc.append(auc_score)
        
    df = pd.DataFrame({"TP":TP_list, "FN":FN_list, "FP":FP_list, "TN":TN_list, "accuracy":acc, "precision":prec, "recall":rec, "especificity":esp, "f1":F1, "auc_score":auc})
    df.to_csv(os.path.join(metricasDirectorio, f'{str(NOMBRE_PRUEBA)}_metricas_{str(nombre)}.csv'), index = None)

############################################################################################


graficasDirectorio = os.path.join(PATH_RDOS,"graficas")
prediccionesDirectorio = os.path.join(PATH_RDOS,"predicciones")
metricasDirectorio = os.path.join(PATH_RDOS,"metricas")
    
os.makedirs(graficasDirectorio, exist_ok = True)
os.makedirs(prediccionesDirectorio, exist_ok = True)
os.makedirs(metricasDirectorio, exist_ok = True)
   
dataset = datasetsDirigidos.synthetic_Dir_100nodes_NoBalanced_20a100_clasesSep(flattened=False, transforms=[GCNConv_preprocess_adjacencyMatrix(GCNConv, symmetric=False), GCNConv_preprocess_features()])

train_dataset = dataset[0:1]
test_dataset = dataset[0:1]
    
#batch_size = 1
n_epochs = 100

class_weight = np.array([0.001, 0.999])
sample_weight = tf.gather(class_weight, tf.argmax(train_dataset[0].y, axis=-1)) 

# Create data loaders for training and testing data
if SAMPLE_WEIGHT:
    train_loader = SingleLoader(train_dataset, epochs=n_epochs, sample_weights=sample_weight)
else:
    train_loader = SingleLoader(train_dataset, epochs=n_epochs)   ####### ATENCION: mask y shuffle

test_loader = SingleLoader(test_dataset, epochs=n_epochs)

n_classes=2
model = GCN(n_labels=n_classes, channels=16)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
        train_loader.load(),
        steps_per_epoch=train_loader.steps_per_epoch,
        epochs=n_epochs,
        #validation_data=val_loader.load(),
        #validation_steps=val_loader.steps_per_epoch,
        #callbacks=callbacks_list                            
)
    
## GRAFICAR
res=pd.DataFrame(history.history)
# Add row index as a new column
res.reset_index(inplace=True)
# Rename the new column to 'row_id'
res.rename(columns={'index': 'epoch'}, inplace=True)
res.to_csv(os.path.join(graficasDirectorio,f'{str(NOMBRE_PRUEBA)}_epochsResults.csv'),index = None)          
    
sns.set_theme(style="whitegrid")
line1 = sns.lineplot(x="epoch", y='loss', data=res, label='Training Loss')
#line2 = sns.lineplot(x="epoch", y='val_loss', data=res, label='Test Loss')
# Add points to each observation
scatter1 = sns.scatterplot(x="epoch", y='loss', data=res, marker='o', color='skyblue')
#scatter2 = sns.scatterplot(x="epoch", y='val_loss', data=res, marker='o', color='orange')
# Change the y-axis label
plt.ylabel("Loss Value")
# Create a legend for the lines
plt.legend()
# Show the plot
plt.savefig(os.path.join(graficasDirectorio,f'{str(NOMBRE_PRUEBA)}_loss.png'))                             
plt.clf()
    
sns.set_theme(style="whitegrid")
line1 = sns.lineplot(x="epoch", y="accuracy", data=res, label='Training Accuracy')
#line2 = sns.lineplot(x="epoch", y="val_accuracy", data=res, label='Test Accuracy')
# Add points to each observation
scatter1 = sns.scatterplot(x="epoch", y="accuracy", data=res, marker='o', color='skyblue')
#scatter2 = sns.scatterplot(x="epoch", y="val_accuracy", data=res, marker='o', color='orange')
# Change the y-axis label
plt.ylabel("Accuracy Value")
# Create a legend for the lines
plt.legend()
# Show the plot
plt.savefig(os.path.join(graficasDirectorio,f'{str(NOMBRE_PRUEBA)}_accuracy.png'))              
plt.clf()
  
# PREDICCION
predicciones(test_loader, "test")


###############################################################################################


    
