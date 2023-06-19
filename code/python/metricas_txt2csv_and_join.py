import pandas as pd
import sys
import os

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
            accuracy.append(eval(numeros[0]))
        elif depurar!=[] and depurar[0] == "precision":
            precision.append(eval(numeros[0]))
        elif depurar!=[] and depurar[0] == "recall":
            recall.append(eval(numeros[0]))
        elif depurar!=[] and depurar[0] == "especificity":
            especificity.append(eval(numeros[0]))
        elif depurar!=[] and depurar[0] == "f1":
            f1.append(eval(numeros[0]))
        elif depurar!=[] and depurar[0] == "auc_score":
            auc.append(eval(numeros[0]))
        else:
            pass
        
    texto.close()

    df = pd.DataFrame({"TP":TP, "FN":FN, "FP":FP, "TN":TN, "accuracy":accuracy, "precision":precision, "recall":recall, "especificity":especificity, "f1":f1, "auc_score":auc})

    df.to_csv(os.path.join(PATH_PRINC, f'prueba_0{i}/metricas/metricas_{CUAL}.csv'),index = None)
      

######################################################################################

df1 = pd.read_csv(os.path.join(PATH_PRINC, f'prueba_00/metricas/metricas_{CUAL}.csv'))

for i in range(1,10):
    ARCHIVO = os.path.join(PATH_PRINC, f'prueba_0{i}/metricas/metricas_{CUAL}.csv')
    p = pd.read_csv(ARCHIVO)
    df1 = pd.concat([df1, p], ignore_index=True)
    
df1.to_csv(os.path.join(PATH_PRINC, f'join_{CUAL}.csv'),index = None)


