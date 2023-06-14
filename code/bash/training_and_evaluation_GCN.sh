#!/bin/bash


FLATTENED_YES="True"
FLATTENED_NO="False"

SYMMETRIC_ADJACENCY="False"
PREPROC_ADJACENCY="True"
PREPROC_FEATURES="True"

CLASE1="datasetsDirigidos.synthetic_Dir_100nodes_balanced_clasesSep"
PATH_ACTUAL=`pwd` 
PATH_RDOS1=${PATH_ACTUAL}/resultados/Dir_100nodes_balanced_clasesSep

mkdir -p ${PATH_RDOS1}

python3 ${PATH_ACTUAL}/code/python/training_and_evaluation_GCN.py ${PATH_RDOS1} ${CLASE1} ${FLATTENED_NO} ${SYMMETRIC_ADJACENCY} ${PREPROC_ADJACENCY} ${PREPROC_FEATURES} |& tee ${PATH_RDOS1}/balanced_NOflattened.txt &
python3 ${PATH_ACTUAL}/code/python/training_and_evaluation_GCN.py ${PATH_RDOS1} ${CLASE1} ${FLATTENED_YES} ${SYMMETRIC_ADJACENCY} ${PREPROC_ADJACENCY} ${PREPROC_FEATURES} |& tee ${PATH_RDOS1}/balanced_flattened.txt &

    
CASOS=(1 2 5 10 20)
for i in "${CASOS[@]}"; do
    CLASE2="datasetsDirigidos.synthetic_Dir_100nodes_NoBalanced_${i}a100_clasesSep"
    PATH_RDOS2=${PATH_ACTUAL}/resultados/Dir_100nodes_NoBalanced_${i}a100_clasesSep
    mkdir -p ${PATH_RDOS2}
    python3 ${PATH_ACTUAL}/code/python/training_and_evaluation_GCN.py ${PATH_RDOS2} ${CLASE2} ${FLATTENED_NO} ${SYMMETRIC_ADJACENCY} ${PREPROC_ADJACENCY} ${PREPROC_FEATURES} |& tee ${PATH_RDOS2}/NoBalanced_${i}a100_NOflattened.txt &
    python3 ${PATH_ACTUAL}/code/python/training_and_evaluation_GCN.py ${PATH_RDOS2} ${CLASE2} ${FLATTENED_YES} ${SYMMETRIC_ADJACENCY} ${PREPROC_ADJACENCY} ${PREPROC_FEATURES} |& tee ${PATH_RDOS2}/NoBalanced_${i}a100_flattened.txt &
done

#chown -R 1015:1015 ${PATH_ACTUAL}

