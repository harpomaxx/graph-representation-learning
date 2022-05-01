import graph_tool.all as gt
import pandas as pd
import json


g=gt.Graph()
ctuName="capture20110815-2.binetflow.labels.gz"
g= gt.load_graph("../../data/graphml/"+ctuName+".graphml")

#for v in g.vertices():
 #   print(g.vp.name[v])
#for e in g.edges():
 #   print(g.ep.Bytes[e])
 #g.list_properties()
 #gt.graph_draw(g, vertex_text=g.vertex_index)

bv, be = gt.betweenness(g,weight=g.ep.Bytes,norm=False)
# Notebook 2:12 min vs Notebook igraph aprox 9min

BC=[]
names=[]
for v in g.vertices():
    BC.append(bv[v])
    names.append(g.vp.name[v])


with open("BC_gt.json", 'w') as f:
    json.dump(BC, f, indent=2) 