import pandas as pd
import igraph
from igraph import Graph
import time
import sys

cap = sys.argv[1]
ncolName = cap + "_noZeroB.ncol"
csvName = cap + "_D_DW.csv"

print("CAPTURE: ", cap)
print("Create graph from: ", ncolName)
print("Store degree and weight degree at: ", csvName)

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


##### Calculate In-Degree (ID) and Out-Degree (OD) #####
# https://igraph.org/python/api/latest/igraph._igraph.GraphBase.html#degree
start_ID = time.time()
startPT_ID = time.process_time()

ID = g.vs.degree(mode = 'in')

end_ID = time.time()
endPT_ID = time.process_time()

time_ID = end_ID - start_ID
timePT_ID = endPT_ID - startPT_ID
print(" time_ID : " , time_ID , " | process_time_ID : " , timePT_ID)


start_OD = time.time()
startPT_OD = time.process_time()

OD = g.vs.degree(mode = 'out')

end_OD = time.time()
endPT_OD = time.process_time()

time_OD = end_OD - start_OD
timePT_OD = endPT_OD - startPT_OD
print(" time_OD : " , time_OD , " | process_time_OD : " , timePT_OD)


##### Calculate In-Degree Weight (IDW) and Out-Degree Weight (ODW) #####
# https://igraph.org/python/api/latest/igraph._igraph.GraphBase.html#strength
start_IDW = time.time()
startPT_IDW = time.process_time()

IDW = g.strength(g.vs, mode = 'in', weights = g.es["weight"])

end_IDW = time.time()
endPT_IDW = time.process_time()

time_IDW = end_IDW - start_IDW
timePT_IDW = endPT_IDW - startPT_IDW
print(" time_IDW : " , time_IDW , " | process_time_IDW : " , timePT_IDW)


start_ODW = time.time()
startPT_ODW = time.process_time()

ODW = g.strength(g.vs, mode = 'out', weights = g.es["weight"])

end_ODW = time.time()
endPT_ODW = time.process_time()

time_ODW = end_ODW - start_ODW
timePT_ODW = endPT_ODW - startPT_ODW
print(" time_ODW : " , time_ODW , " | process_time_ODW : " , timePT_ODW)



##### Create a DataFrame and store it in csv file #####
start_store = time.time()
startPT_store = time.process_time()

features = pd.DataFrame(list(zip(g.vs["name"], ID, OD, IDW, ODW)), columns=['node','ID','OD','IDW','ODW'])

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


