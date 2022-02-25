import pandas as pd
import igraph
from igraph import Graph
#import sys

#first capture (20110810)

#read csv file and store it as dataframe
c20110810 = pd.read_csv("capture20110810_forGraph.csv")

#c20110810[["SrcAddr","DstAddr","SrcBytes","DstBytes"]] #to change the order of columns

#check if there is at least one reverse
control = False
for i in range(len(c20110810["SrcAddr"])):
    for j in range(len(c20110810["DstAddr"])):
        if (c20110810["SrcAddr"][i]==c20110810["DstAddr"][j]):
            print(c20110810["SrcAddr"][i],i,j)
            control = True
            break
    if control:
        break
            
#create dataframes with conections SrcAddr->DstAddr and responses DstAddr->SrcAddr
ida=c20110810[["SrcAddr","DstAddr","SrcBytes"]].copy()
vuelta=c20110810[["DstAddr","SrcAddr","DstBytes"]].copy()
ida.rename(columns={'SrcAddr': 'origen', 'DstAddr': 'destino', 'SrcBytes': 'peso'}, inplace=True)
vuelta.rename(columns={'DstAddr': 'origen', 'SrcAddr': 'destino', 'DstBytes': 'peso'}, inplace=True)

#concatenate dataframes and add the weights if there are repeated links
tutti=pd.concat([ida, vuelta], ignore_index=True)
df=tutti.groupby(['origen','destino'], as_index=False)['peso'].sum().copy()

##################################################################################
#create an NCOL file to then generate the graph  
df.to_csv('cap20110810.ncol',sep=' ',header=None, index=None)
#at another time:
grafo=Graph.Read_Ncol('cap20110810.ncol',weights=True,directed=True)

##################################################################################

#create a graph at the moment
g=Graph.DataFrame(df, directed=True)


#calculate degree
grados=g.degree()
gradosIn=g.degree(mode='in')
gradosOut=g.degree(mode='out')
sum(i>150 for i in grados)

#calculate degree weight
vertices=pd.DataFrame(list(set(df["origen"]))) #tiene TODOS los vertices porque "origen" tiene SrcAddr y DestAddr
g.strength(vertices[0].tolist(), mode='all', weights=df["peso"])



#ANOTHER WAY TO CREATE THE GRAPH
G=Graph.TupleList(df.values,weights=True,directed=True)
G.strength(G.vs,mode='all',weights=G.es['weight']) # G.vs['name'].sort()==list(vertices).sort() 
                                                   # G.es['weight'].sort()==list(df["peso"]).sort()
G.write_ncol("c20110810.ncol") # to load later
