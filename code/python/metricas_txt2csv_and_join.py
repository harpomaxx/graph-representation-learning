import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

PATH_PRINC = str(sys.argv[1]) # path de interes, por ejemplo: "/home/tati/Nextcloud/BotChase/graph-representation-learning/rawdata/synthetic/resultados/Dir_100nodes_balanced_clasesSep/NO_flattened"
CUAL = str(sys.argv[2]) # cual archivo: "test", "val" o "train"


def quitar(x):
    return x.strip(("[]:,\n "))
    
    
for i in range(10):
    TP = []
    FN = []
    FP = []
    TN = []
    accuracy = []
    precision = []
    recall = []
    especificity = []
    f1 = []
    auc = []

    texto = open(os.path.join(PATH_PRINC, f'prueba_0{i}/metricas/metricas_{CUAL}.txt'), "r")
    
    for line in texto:
        depurar = [quitar(x) for x in line.split() if quitar(x)!='' and ")" not in quitar(x) and "=" not in quitar(x)]
        numeros = [y for y in depurar if y.replace('.','').isdigit()]   
        if depurar==numeros and depurar!=[] and numeros!=[]:
            if "[[" in line:
                TP.append(eval(numeros[0]))
                FN.append(eval(numeros[1]))
            else:
                FP.append(eval(numeros[0]))
                TN.append(eval(numeros[1]))
        elif depurar!=[] and depurar[0] == "accuracy":
            if depurar[1] == "nan":
                accuracy.append(np.nan)
            else:
                accuracy.append(eval(numeros[0]))
        elif depurar!=[] and depurar[0] == "precision":
            if depurar[1] == "nan":
                precision.append(np.nan)
            else:
                precision.append(eval(numeros[0]))
        elif depurar!=[] and depurar[0] == "recall":
            if depurar[1] == "nan":
                recall.append(np.nan)
            else:
                recall.append(eval(numeros[0]))
        elif depurar!=[] and depurar[0] == "especificity":
            if depurar[1] == "nan":
                especificity.append(np.nan)
            else:
                especificity.append(eval(numeros[0]))
        elif depurar!=[] and depurar[0] == "f1":
            if depurar[1] == "nan":
                f1.append(np.nan)
            else:
                f1.append(eval(numeros[0]))
        elif depurar!=[] and depurar[0] == "auc_score":
            if depurar[1] == "nan":
                auc.append(np.nan)
            else:
                auc.append(eval(numeros[0]))
        else:
            pass
           
    texto.close()
    
    df = pd.DataFrame({"TP":TP, "FN":FN, "FP":FP, "TN":TN, "accuracy":accuracy, "precision":precision, "recall":recall, "especificity":especificity, "f1":f1, "auc_score":auc})
    
    df.to_csv(os.path.join(PATH_PRINC, f'prueba_0{i}/metricas/metricas_{CUAL}.csv'),index = None)
      

######################################################################################

joinDirectorio = os.path.join(PATH_PRINC, "join")
os.makedirs(joinDirectorio, exist_ok = True)

df1 = pd.read_csv(os.path.join(PATH_PRINC, f'prueba_00/metricas/metricas_{CUAL}.csv'))

for i in range(1,10):
    ARCHIVO = os.path.join(PATH_PRINC, f'prueba_0{i}/metricas/metricas_{CUAL}.csv')
    p = pd.read_csv(ARCHIVO)
    df1 = pd.concat([df1, p], ignore_index=True)
    
df1.to_csv(os.path.join(joinDirectorio, f'join_{CUAL}.csv'),index = None)

    
#sns.set_theme(style="whitegrid")
boxplot1 = sns.boxplot(data=df1.iloc[:,7:8]) 
plt.savefig(os.path.join(joinDirectorio,f'especificity_{CUAL}.png'))                             
plt.clf()
    
#sns.set_theme(style="whitegrid")
boxplot2 = sns.boxplot(data=df1.iloc[:,9:10]) 
plt.savefig(os.path.join(joinDirectorio,f'auc_{CUAL}.png'))                             
plt.clf()
    
