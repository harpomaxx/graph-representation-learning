#!/bin/bash

# Tuve que correr para el caso flattened no-balanceado casos 10,20,50 de nuevo


DATASETS="datasetsDirigidos"
DIRECCION="Dir"
NODOS=100
BALANCED="balanced"
CLASES="Mezcl"


FLATTENED_YES="True"
FLATTENED_NO="False"

SYMMETRIC_ADJACENCY="False"
PREPROC_ADJACENCY="True"
PREPROC_FEATURES="True"

PATH_ACTUAL=`pwd` # carpeta "synthetic" (en el docker, donde la estructura es synthetic/code synthetic/datasets synthetic/resultados)

CLASE1="${DATASETS}.synthetic_${DIRECCION}_${NODOS}nodes_${BALANCED}_clases${CLASES}"
PATH_RDOS1=${PATH_ACTUAL}/resultados/${DIRECCION}_${NODOS}nodes_${BALANCED}_clases${CLASES}

mkdir -p ${PATH_RDOS1}/code_used

python3 ${PATH_ACTUAL}/code/python/training_and_evaluation_GCN.py ${PATH_RDOS1} ${CLASE1} ${FLATTENED_NO} ${SYMMETRIC_ADJACENCY} ${PREPROC_ADJACENCY} ${PREPROC_FEATURES} |& tee ${PATH_RDOS1}/${BALANCED}_NOflattened.txt &
python3 ${PATH_ACTUAL}/code/python/training_and_evaluation_GCN.py ${PATH_RDOS1} ${CLASE1} ${FLATTENED_YES} ${SYMMETRIC_ADJACENCY} ${PREPROC_ADJACENCY} ${PREPROC_FEATURES} |& tee ${PATH_RDOS1}/${BALANCED}_flattened.txt &
cp ${PATH_ACTUAL}/code/python/training_and_evaluation_GCN.py ${PATH_RDOS1}/code_used/
cp ${PATH_ACTUAL}/code/bash/training_and_evaluation_GCN.sh ${PATH_RDOS1}/code_used/


BALANCED="NoBalanced"
CASOS=(1 2 5 10 20) # casos para 100 nodos
#CASOS=(10 20 50 100 200) # casos para 1000 nodos
for i in "${CASOS[@]}"; do
    CLASE2="${DATASETS}.synthetic_${DIRECCION}_${NODOS}nodes_${BALANCED}_${i}a${NODOS}_clases${CLASES}"
    PATH_RDOS2=${PATH_ACTUAL}/resultados/${DIRECCION}_${NODOS}nodes_${BALANCED}_${i}a${NODOS}_clases${CLASES}
    mkdir -p ${PATH_RDOS2}/code_used
    python3 ${PATH_ACTUAL}/code/python/training_and_evaluation_GCN.py ${PATH_RDOS2} ${CLASE2} ${FLATTENED_NO} ${SYMMETRIC_ADJACENCY} ${PREPROC_ADJACENCY} ${PREPROC_FEATURES} |& tee ${PATH_RDOS2}/${BALANCED}_${i}a${NODOS}_NOflattened.txt &
    python3 ${PATH_ACTUAL}/code/python/training_and_evaluation_GCN.py ${PATH_RDOS2} ${CLASE2} ${FLATTENED_YES} ${SYMMETRIC_ADJACENCY} ${PREPROC_ADJACENCY} ${PREPROC_FEATURES} |& tee ${PATH_RDOS2}/${BALANCED}_${i}a${NODOS}_flattened.txt &
    cp ${PATH_ACTUAL}/code/python/training_and_evaluation_GCN.py ${PATH_RDOS2}/code_used/
    cp ${PATH_ACTUAL}/code/bash/training_and_evaluation_GCN.sh ${PATH_RDOS2}/code_used/
done


#chown -R 1015:1015 ${PATH_ACTUAL}

