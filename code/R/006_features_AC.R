#!/bin/Rscript

suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(dplyr))

args <- commandArgs(TRUE)
ncolName <- paste(args[1], "_noZeroB.ncol", sep = "")
csvName <- paste(args[1], "_AC.csv", sep = "")

sprintf("CAPTURE: %s", args[1])
sprintf("Create graph from: %s", ncolName)
sprintf("Store alpha centrality at: %s", csvName)


##### Load graph #####
start_load <- proc.time()
g <- read_graph(file = ncolName, format = "ncol", directed = T)
end_load <- proc.time()

time_load = end_load - start_load
print(" time_load : ")
print(time_load)


##### Calculate Alpha Centrality (AC) #####
# https://igraph.org/r/html/latest/alpha_centrality.html
start_AC <- proc.time()
AC <- alpha_centrality(g, alpha = 0.01, exo = 1)
end_AC <- proc.time()

time_AC = end_AC - start_AC
print(" time_AC : ")
print(time_AC)


##### Create a DataFrame and store it in csv file #####
start_store <- proc.time()
df <- data.frame(AC) %>% as_tibble(rownames = "node")
write.csv(df, csvName, row.names = FALSE)
end_store <- proc.time()

time_store = end_store - start_store
print(" time_store : ")
print(time_store)

