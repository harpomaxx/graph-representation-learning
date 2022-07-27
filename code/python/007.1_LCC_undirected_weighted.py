import pandas as pd
import igraph
from igraph import Graph
import time
import sys

cap = sys.argv[1]
ncolName = cap + "_noZeroB.ncol"
csvName = cap + "_LCC_undirected_weighted.csv"

print("CAPTURE: ", cap)
print("Create graph from: ", ncolName)
print("Store LCC (undirected and weighted, following Barrat et al) at: ", csvName)

start = time.time()
startPT = time.process_time()


##### Load graph #####
start_load = time.time()
startPT_load = time.process_time()

g = Graph.Read_Ncol(ncolName, weights=True, directed=True)

end_load = time.time()
endPT_load = time.process_time()

time_load = end_load - start_load
timePT_load = endPT_load - startPT_load
print(" time_load : " , time_load , " | process_time_load : " , timePT_load)




##### Calculate Local Clustering Coefficient (LCC) undirected and weighted, following Barrat et al #####
#https://igraph.org/python/api/latest/igraph._igraph.GraphBase.html#transitivity_local_undirected
start_LCC = time.time()
startPT_LCC = time.process_time()

LCC = g.transitivity_local_undirected(vertices = None, mode = "zero", weights = g.es["weight"])

end_LCC = time.time()
endPT_LCC = time.process_time()

time_LCC = end_LCC - start_LCC
timePT_LCC = endPT_LCC - startPT_LCC
print(" time_LCC : " , time_LCC , " | process_time_LCC : " , timePT_LCC)


##### Create a DataFrame and store in csv file #####
start_store = time.time()
startPT_store = time.process_time()

features = pd.DataFrame(list(zip(g.vs["name"], LCC)), columns=['node','LCC'])

features.to_csv(csvName, index = None)

end_store = time.time()
endPT_store = time.process_time()

time_store = end_store - start_store
timePT_store = endPT_store - startPT_store
print(" time_store : " , time_store , " | process_time_store : " , timePT_store)


##### Total time
end = time.time()
endPT = time.process_time()

time_total = end - start
timePT_total = endPT - startPT
print(" time_total : " , time_total , " | process_time_total : " , timePT_total)

