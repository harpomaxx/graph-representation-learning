import pandas as pd
import sys

ARCHIVO = str(sys.argv[1]) # archivo de metricas en txt a convertir a csv (solo para archivos con metricas)


def quitar(x):
    return x.strip(("[]:,\n "))


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

texto = open(ARCHIVO, "r")

        
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

if ARCHIVO == "metricas_test.txt":
    df.to_csv("metricas_test.csv",index = None)
else:
    df.to_csv(f'{ARCHIVO.strip(".txt")}.csv',index = None)
    

