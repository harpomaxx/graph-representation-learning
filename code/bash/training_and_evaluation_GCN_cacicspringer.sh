#!/bin/bash

CLASE="CacicSpringer_SyntheticGraphs"
NUM_GRAPHS=100
NUM_NODES=100
NUM_COMMUNITIES=2
NUM_FEATURES=2
CONFIG_STR=("A-I" "B-II" "B-IV" "C-III" "D-II" "D-IV")
FEATURE_COV_MATRIX="None"
DIRECTED="True"
CLASS1_PERCENT=("None" "1" "2" "5" "7.5" "10" "12.5" "15" "17.5" "20")
SEED="123-234-345-456-567-678-789-321-654-987"
FLATTENED="False"
ADJNUL="False"
SYMMETRIC_ADJACENCY="False" # porque son grafos dirigidos
PREPROC_ADJACENCY="True"
PREPROC_FEATURES="True"


PATH_ACTUAL=`pwd` # carpeta "synthetic" (para eso, correr antes: `docker exec -it <nombre_container> /bin/bash`) 


for config_str in "${CONFIG_STR[@]}"; do
    for class1_percent in "${CLASS1_PERCENT[@]}"; do
        if [ "$class1_percent" == "None" ]; then
            dir_name_c1="Balanced"
        else
            dir_name_c1=${class1_percent}_percent
        fi
        PATH_RDOS=${PATH_ACTUAL}/resultados/CacicSpringer2023/${config_str}/${dir_name_c1}/NO_flattened
        mkdir -p ${PATH_RDOS}
	python3 ${PATH_ACTUAL}/code/python/training_and_evaluation_GCN_cacicspringer.py ${PATH_RDOS} ${CLASE} ${NUM_GRAPHS} ${NUM_NODES} ${NUM_COMMUNITIES} ${NUM_FEATURES} ${config_str} ${FEATURE_COV_MATRIX} ${DIRECTED} ${class1_percent} ${SEED} ${FLATTENED} ${ADJNUL} ${SYMMETRIC_ADJACENCY} ${PREPROC_ADJACENCY} ${PREPROC_FEATURES} |& tee ${PATH_RDOS}/salida.txt
    done
done


# Caso flattened para la configuracion A-I
PATH_RDOS=${PATH_ACTUAL}/resultados/CacicSpringer2023/A-I/Balanced/flattened
config_str="A-I"
class1_percent="None"
FLATTENED="True"
mkdir -p ${PATH_RDOS}
python3 ${PATH_ACTUAL}/code/python/training_and_evaluation_GCN_cacicspringer.py ${PATH_RDOS} ${CLASE} ${NUM_GRAPHS} ${NUM_NODES} ${NUM_COMMUNITIES} ${NUM_FEATURES} ${config_str} ${FEATURE_COV_MATRIX} ${DIRECTED} ${class1_percent} ${SEED} ${FLATTENED} ${ADJNUL} ${SYMMETRIC_ADJACENCY} ${PREPROC_ADJACENCY} ${PREPROC_FEATURES} |& tee ${PATH_RDOS}/salida.txt


