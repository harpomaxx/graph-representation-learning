#!/bin/bash

# Para correr en /ctu-13 (directorio que tiene los archivos comprimidos)

mkdir ip_bytes

for filename in ./*.gz; do 
    gunzip $filename    # descomprime
    grep "\"" ${filename%.*}    # controla que no haya comas dentro de comillas (sino lo siguiente tendría errores)

    if [ $? -ne 0 ]; then
        awk -F "," '{print $4, $7, $13, $14}' ${filename%.*} > ${filename%.*.*.*}_tmp   # extrae la info SrcAddr DstAddr TotBytes SrcBytes
        awk -v OFS="," '$3-=$4' ${filename%.*.*.*}_tmp | sed '1iSrcAddr,DstAddr,DstBytes,SrcBytes'> ${filename%.*.*.*}_forGraph.csv     # resta TotBytes-SrcBytes, y coloca cabecera actualizada
    else
        echo "Usar otro método porque hay comas dentro de comillas" > ${filename%.*.*.*}_warning.txt
    fi
done    

mv *_forGraph.csv ip_bytes/


####################################################

# PRUEBA CON LA PRIMERA CAPTURA:

#gunzip capture20110810.binetflow.labels.gz
#grep "\"" capture20110810.binetflow.labels

#if [ $? -ne 0 ]; then
#    awk -F "," '{print $4, $7, $13, $14}' capture20110810.binetflow.labels > capture20110810_extraccionDatos
#    awk -v OFS="," '$3-=$4' capture20110810_extraccionDatos > capture20110810_restaBytes
#    sed '1iSrcAddr,DstAddr,DstBytes,SrcBytes' capture20110810_restaBytes > capture20110810_listo
#else
#    echo "Usar otro método porque hay comas dentro de comillas" > awkSalidaActualizada
#fi

#SrcAddr DstAddr TotBytes SrcBytes # cabecera original luego de hacer la extracción
