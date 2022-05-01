import igraph as ig
import pandas as pd
import json

#We can either calculate all with graph-tool, or combine graph-tool with igraph use in one environment
#(not tried yet, BC is computed in another script and written to a json file).

def botnetIP(ctu_name):
    a=ctu_name=="capture20110810.binetflow.labels.gz" or ctu_name=="capture20110811.binetflow.labels.gz"
    b=ctu_name=="capture20110815.binetflow.labels.gz"or ctu_name=="capture20110812.binetflow.labels.gz"
    c=ctu_name=="capture20110815-3.binetflow.labels.gz" or ctu_name=="capture20110815-2.binetflow.labels.gz"
    d=ctu_name=="capture20110816.binetflow.labels.gz" or ctu_name=="capture20110816-2.binetflow.labels.gz"
    e=ctu_name=="capture20110816-3.binetflow.labels.gz"
    if  a or b or c or d or e:
        bots=["147.32.84.165"]
    elif ctu_name=="capture20110817.binetflow.labels.gz" or ctu_name=="capture20110818.binetflow.labels.gz":
        bots=["147.32.84.165","147.32.84.191","147.32.84.192","147.32.84.193","147.32.84.204","147.32.84.205","147.32.84.206","147.32.84.207","147.32.84.208","147.32.84.209"]
    elif ctu_name=="capture20110818-2.binetflow.labels.gz" or ctu_name=="capture20110819.binetflow.labels.gz":
        bots=["147.32.84.165","147.32.84.191","147.32.84.192"]
    return bots

g=ig.Graph()
ctuName="capture20110815-2.binetflow.labels.gz"
g=g.Read_Ncol("../../data/ncol/"+ctuName+".ncol", names=True, weights='if_present', directed=True)

#Feature calculations
ID=g.degree(mode="in") 
OD=g.degree(mode="out")
IDW=g.strength(mode="in",weights="weight")
ODW=g.strength(mode="out",weights="weight")
LCC=g.transitivity_local_undirected(mode='zero') #There exists a Weighted alternative. In the paper they don't consider weights
#Mode zero because we can't make use of nan in a column for ML.(Verify)

#BC=g.betweenness(directed=True,weights="weight")    #Calculation of BC with igraph (not optimal)
#Notebook:    9 min
#samson running time: 6min???
with open("BC_gt.json", 'r') as f:
    BC_gt = json.load(f)


#AC=

#Botnets in each capture
botnets=botnetIP(ctuName)

df = pd.DataFrame()
df['node']=g.vs['name']
df['ID']=ID
df['OD']=OD
df['IDW']=IDW
df['ODW']=ODW
df['LCC']=LCC
#df['AC']=AC
df['label']="normal"
for ip in botnets:
    df.loc[df["node"] == ip, ["label"]] = "botnet"
df.sort_values(by=['node'], inplace=True) #We have to sort BEFORE adding BC computed with graph-tools, because it is already sorted by nodes.
df['BC']=BC_gt 
df = df[['node','ID', 'OD', 'IDW', 'ODW','BC','LCC','label']] #Add AC when ready
df.to_csv("../../data/csv/"+ctuName+".features.csv",index=False) 