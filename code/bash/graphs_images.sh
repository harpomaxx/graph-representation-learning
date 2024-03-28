#!/bin/bash

CLASE="CacicSpringer_SyntheticGraphs"
NUM_GRAPHS=100
NUM_NODES=100
NUM_COMMUNITIES=2
NUM_FEATURES=2
CONFIG_STR=("e3-f3" "e2-f2" "e2-f0" "e1-f1" "e0-f2" "e0-f0")
FEATURE_COV_MATRIX="None"
DIRECTED="True"
CLASS1_PERCENT=("None" "1" "2" "5" "7.5" "10" "12.5" "15" "17.5" "20")
SEED="123-234-345-456-567-678-789-321-654-987"
FLATTENED="False"
ADJNUL="False"
SYMMETRIC_ADJACENCY="False" # porque son grafos dirigidos
PREPROC_ADJACENCY="False" # para que grafique el grafo original
PREPROC_FEATURES="False"


PATH_ACTUAL=`pwd` # carpeta "synthetic" (para eso, correr antes: `docker exec -it <nombre_container> /bin/bash`) 


for config_str in "${CONFIG_STR[@]}"; do
    for class1_percent in "${CLASS1_PERCENT[@]}"; do
        if [ "$class1_percent" == "None" ]; then
            dir_name_c1="Balanced"
        else
            dir_name_c1=${class1_percent}_percent
        fi
        PATH_RDOS=${PATH_ACTUAL}/resultados/CacicSpringer2023/graphs_images
        mkdir -p ${PATH_RDOS}
	python3 ${PATH_ACTUAL}/code/python/graphs_images.py ${PATH_RDOS} ${CLASE} ${NUM_GRAPHS} ${NUM_NODES} ${NUM_COMMUNITIES} ${NUM_FEATURES} ${config_str} ${FEATURE_COV_MATRIX} ${DIRECTED} ${class1_percent} ${SEED} ${FLATTENED} ${ADJNUL} ${SYMMETRIC_ADJACENCY} ${PREPROC_ADJACENCY} ${PREPROC_FEATURES}
    done
done


