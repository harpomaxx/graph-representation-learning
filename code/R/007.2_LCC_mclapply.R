#!/bin/Rscript



suppressPackageStartupMessages(library(parallel))
suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(data.table))

args <- commandArgs(TRUE)
ncolName <- paste(args[1], "_noZeroB.ncol", sep = "")
csvName <- paste(args[1], "_LCC_mclapply.csv", sep = "")

numCores <- 16 #as.integer(args[2])

cat("CAPTURA: ", args[1], "\n")
cat("Crea grafo desde: ", ncolName, "\n")
cat("Almacena resultados de LCC en: ", csvName, "\n")
cat("Cantidad de cores: ", numCores, "\n")


#computeLocalClusteringCoefficient_new <- function(neighbourhood, neighbourhoodOUT, k, index) {
    # Calculate LCC for a vertex in graph
   
#    numberNeighbours <- k[index]
#    localNeighbourhood <- neighbourhood[index][[1]]
    
#    if (k[index] <= 1) {
#        return(0.0)
#    } else {
#        numberTriplets <- 0
#        for (i in seq(1, numberNeighbours)) {
#            if (k[localNeighbourhood[i]] > 1) {
#                for (m in neighbourhoodOUT[localNeighbourhood[i]][[1]]) {
#                    if ((m != vertices[index]) && (m %in% localNeighbourhood)) {
#                        numberTriplets <- numberTriplets + 1
#                    }
#                }
#            }
#        }
#        return(numberTriplets / (k[index] * (k[index] - 1)))
#    }
#}


## Función para calcular LCC para el vértice en la posición "index" de la lista con todos los vértices. 
## Esta función tiene cambios que optimizarían el cálculo en comparación de la función original presente en TP-final/scripts/R/00_LCC_lapply_original.R

computeLocalClusteringCoefficient_new <- function(neighbourhood, neighbourhoodOUT, k, index) {
    # Calcula LCC para el vértice en la posición "index" (llamado VERTICE) de la lista con todos los vértices de un gafo.
    #
    # Argumentos: "neighbourhood": lista con el entorno a distancia 1 de cada vértice de un grafo, teniendo en cuenta TODOS los enlaces 
    #             "neighbourhoodOUT": lista con el entorno de distancia 1 de cada vértice de un grafo, teniendo en cuenta sólo los enlaces SALIENTES 
    #             "k": lista con el número de vecinos presentes en cada entorno de "neighbourhood"
    #             "index": índice que indica a qué vértice de la lista de vértices, se calculará el valor de LCC 
    # Salida: valor de LCC (float)
   
    numberNeighbours <- k[index] # número de vecinos de VERTICE
    localNeighbourhood <- neighbourhood[index][[1]] # entorno a distancia 1 de VERTICE
    
    if (k[index] <= 1) {
        # Si el número de vecinos en el entorno de VERTICE es menor o igual a 1, el valor de LCC es 0
        return(0.0)
    } else {
        # Sino, el valor de LCC dependerá del número de "triángulos" formados entre nodos vecinos y VERTICE
        numberTriplets <- 0
        for (nodo in seq(1, numberNeighbours)) {
            # Para cada NODO vecino de VERTICE
            if (k[localNeighbourhood[nodo]] > 1) {
                # Si NODO tiene más de un vecino (es decir, VERTICE no es el único vecino de NODO)
                for (vecino in neighbourhoodOUT[localNeighbourhood[nodo]][[1]]) {
                    # Para cada VECINO en el entorno dado por los enlaces "salientes" a NODO
                    if ((vecino != vertices[index]) && (vecino %in% localNeighbourhood)) {
                        # Si VECINO no es VERTICE, y si VECINO es también vecino de VERTICE,entonces se suma 1 a numberTriplets
                        numberTriplets <- numberTriplets + 1
                    }
                }
            }
        }
        return(numberTriplets / (k[index] * (k[index] - 1))) # Valor de LCC de VERTICE
    }
}

##### Load graph #####
start_load <- proc.time()
g <- read_graph(file = ncolName, format = "ncol", directed = T)
end_load <- proc.time()

time_load = end_load - start_load
cat(" time_load : \n")
print(time_load)


##### Calculate Local Clustering Coefficient (LCC) #####
start_pre <- proc.time()

vertices <- V(g)

neighbourhood <- ego(g, order = 1, nodes = vertices, mode = "all", mindist = 1) 
names(neighbourhood) <- vertices$name

neighbourhoodOUT <- ego(g, order = 1, nodes = vertices, mode = "out", mindist = 1) 
names(neighbourhoodOUT) <- vertices$name

k <- ego_size(g, order = 1, nodes = vertices, mode = "all", mindist = 1) 

end_pre <- proc.time()
time_pre = end_pre - start_pre
cat(" time_pre : \n")
print(time_pre)

start_LCC <- proc.time()
LCC <- mclapply(seq(1,length(vertices)), function(x) computeLocalClusteringCoefficient_new(neighbourhood, neighbourhoodOUT, k, x), mc.cores = numCores)
end_LCC <- proc.time()

time_LCC = end_LCC - start_LCC
cat(" time_LCC : \n")
print(time_LCC)


##### Create a DataFrame and store it in csv file #####
start_store <- proc.time()

names(LCC) <- vertices$name
df <- unlist(LCC) %>% data.frame(LCC = .) %>% as_tibble(rownames = "node")

fwrite(df, csvName)
end_store <- proc.time()

time_store = end_store - start_store
cat(" time_store : \n")
print(time_store)

