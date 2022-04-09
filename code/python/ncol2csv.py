import igraph as ig
import pandas as pd

g=ig.Graph()
ctuName="capture20110815-2.binetflow.labels.gz"
g=g.Read_Ncol("../../data/ncol/"+ctuName+".ncol", names=True, weights='if_present', directed=True)

ig.summary(g)

ID=g.degree(mode="in") 
OD=g.degree(mode="out")
IDW=g.strength(mode="in",weights="weight")
ODW=g.strength(mode="out",weights="weight")
LCC=g.transitivity_local_undirected(mode='zero') #There exists a Weighted alternative. In the paper they don't consider weights
#Mode zero because we can't make use of nan in a column for ML.(Verify)
BC=g.betweenness(directed=True,weights="weight")
botnets=["147.32.84.165"]
df = pd.DataFrame()
df['node']=g.vs['name']
df['ID']=ID
df['OD']=OD
df['IDW']=IDW
df['ODW']=ODW
df['BC']=BC
df['LCC']=LCC
#df['AC']=AC
df['label']="normal"
for ip in botnets:
    df.loc[df["node"] == ip, ["label"]] = "botnet"
df.to_csv("../../data/csv/"+ctuName+".features.csv") #pay attention not to save the dataframe ids 