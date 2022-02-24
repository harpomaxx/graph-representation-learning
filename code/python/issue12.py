import pandas as pd
import igraph
from igraph import Graph
import sys

#pruebo con el primer archivo

c20110810 = pd.read_csv("capture20110810_forGraph.csv")
#c20110810[["SrcAddr","DstAddr","SrcBytes","DstBytes"]] #podria cambiar el orden de las columnas

#controla si existe al menos una reversa 
control = False
for i in range(len(c20110810["SrcAddr"])):
    for j in range(len(c20110810["DstAddr"])):
        if (c20110810["SrcAddr"][i]==c20110810["DstAddr"][j]):
            print(c20110810["SrcAddr"][i],i,j)
            control = True
            break
    if control:
        break
            

ida=c20110810[["SrcAddr","DstAddr","SrcBytes"]].copy()
vuelta=c20110810[["DstAddr","SrcAddr","DstBytes"]].copy()
ida.rename(columns={'SrcAddr': 'origen', 'DstAddr': 'destino', 'SrcBytes': 'peso'}, inplace=True)
vuelta.rename(columns={'DstAddr': 'origen', 'SrcAddr': 'destino', 'DstBytes': 'peso'}, inplace=True)

tutti=pd.concat([ida, vuelta], ignore_index=True)
df=tutti.groupby(['origen','destino'], as_index=False)['peso'].sum().copy()

vertices=pd.DataFrame(list(set(df["origen"])))

g=Graph.DataFrame(df, directed=True)
g.degree()
g.degree(mode='in')
g.degree(mode='out')
sum(i>150 for i in grados)
