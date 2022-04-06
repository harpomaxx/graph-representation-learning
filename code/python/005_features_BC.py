import pandas as pd
import igraph
from igraph import Graph
import time
import sys

cap = sys.argv[1]
ncolName = cap + "_noZeroB.ncol"
csvName = cap + "_BC.csv"

print("CAPTURE: ", cap)
print("Create graph from: ", ncolName)
print("Store betweenness centrality at: ", csvName)

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


##### Calculate Betweenness Centrality (BC) ##### 
#https://igraph.org/python/api/latest/igraph._igraph.GraphBase.html#betweenness
start_BC = time.time()
startPT_BC = time.process_time()

BC = g.betweenness(vertices = None, directed = True, cutoff = None, weights = g.es["weight"])

end_BC = time.time()
endPT_BC = time.process_time()

time_BC = end_BC - start_BC
timePT_BC = endPT_BC - startPT_BC
print(" time_BC : " , time_BC , " | process_time_BC : " , timePT_BC)


##### Create a DataFrame and store it in csv file #####
start_store = time.time()
startPT_store = time.process_time()

features = pd.DataFrame(list(zip(g.vs["name"], BC)), columns=['node','BC'])

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


