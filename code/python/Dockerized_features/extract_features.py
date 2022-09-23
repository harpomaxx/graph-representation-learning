import graph_tool.all as gt
import pandas as pd
import numpy as np
from optparse import OptionParser
import sys
import networkx as nx

def parsearg():
    parser = OptionParser(usage="usage: %prog [opt] ",
                          version="%prog 1.0")
    parser.add_option("-i", "--input",
                      action="store",
                      dest="input",
                      default=False,
                      help="Input must be the capture name")
    parser.add_option("-o", "--output",
                      action="store", # optional because action defaults to "store"
                      dest="output",
                      help="Output must be a csv file address",)
    (opt, args) = parser.parse_args()
    return opt

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

#Feature normalizer:
def normalize(graph,feature):
    """feature is a property map of the graph"""
    feature_norm=graph.new_vp("double")
    for v in graph.vertices():
        sum_feature_neigh=0
        for v_neigh in v.all_neighbors():
            sum_feature_neigh+=feature[v_neigh]
        if sum_feature_neigh==0:
            sum_feature_neigh=1
        feature_norm[v]=feature[v]/sum_feature_neigh
    return feature_norm


if __name__ == '__main__':
   opt = parsearg()
   if opt.input == None:
        print("[Py] Parameters missing. Please use --help for look at available parameters.")
        sys.exit()

   else:   
        #Importing the graph:
        g=gt.Graph()   #graph-tool
        g_nx=nx.DiGraph()   #networkX.
        #ctuName="capture20110815-2.binetflow.labels.gz"
        ctuName=opt.input
        print("capture:"+ctuName)
        g= gt.load_graph("./data/graphml/"+ctuName+".graphml")
        g_nx=nx.read_graphml("./data/graphml/"+ctuName+".graphml")
        print("[Py] Input graphml file read correctly")

        #FEATURES CALCULATION AND NORMALIZATION
        #Get in degree
        ID=g.degree_property_map("in")
        ID_norm=normalize(g,ID)
        ID_df=pd.DataFrame(ID)
        ID_norm_df=pd.DataFrame(ID_norm)
        #Get out degree
        OD=g.degree_property_map("out")
        OD_norm=normalize(g,OD)
        OD_df=pd.DataFrame(OD)
        OD_norm_df=pd.DataFrame(OD_norm)

        #Get edge property map of weights
        Bytes=g.edge_properties["Bytes"]
        #Get in degree with weight
        IDW=g.degree_property_map("in",weight=Bytes)
        IDW_norm=normalize(g,IDW)
        IDW_df=pd.DataFrame(IDW)
        IDW_norm_df=pd.DataFrame(IDW_norm)

        #Get out degree with weight
        ODW=g.degree_property_map("out",weight=Bytes)
        ODW_norm=normalize(g,ODW)
        ODW_df=pd.DataFrame(ODW)
        ODW_norm_df=pd.DataFrame(ODW_norm)

        #Alpha centrality (with networkX)
        # AC=nx.katz_centrality_numpy(g_nx, alpha=0.001, beta=1.0, normalized=False, weight=None)
        # AC_norm=normalize(g_nx,AC)
        AC=gt.katz(g,weight=None,norm=False,alpha=0.001)
        AC_norm=normalize(g,AC)
        
        AC_df=pd.DataFrame(AC)
        AC_norm_df=pd.DataFrame(AC_norm)

        #Betweenness centrality
        BV, _ = gt.betweenness(g,weight=Bytes,norm=False)
        BV_norm=normalize(g,BV)
        BC_df=pd.DataFrame(BV)
        BC_norm_df=pd.DataFrame(BV_norm)

        #Local clustering coefficient
        LCC=gt.local_clustering(g,weight=None)
        LCC_norm=normalize(g,LCC)
        LCC_df=pd.DataFrame(LCC)
        LCC_norm_df=pd.DataFrame(LCC_norm)
        


        #Get node names
        nodes=pd.DataFrame(g.vp.name)
        nodes_df=pd.DataFrame(nodes)

        #Base labels
        labels_df=pd.DataFrame(np.zeros(len(nodes_df)))


        #Concatenate dataframes
        df=pd.concat([nodes_df,ID_df,OD_df,IDW_df,ODW_df,BC_df,LCC_df,AC_df,labels_df],axis=1)
        norm_df=pd.concat([nodes_df,ID_norm_df,OD_norm_df,IDW_norm_df,ODW_norm_df,BC_norm_df,LCC_norm_df,AC_norm_df,labels_df],axis=1)
        df.columns=["node","ID","OD","IDW","ODW","BC","LCC","AC","label"]
        norm_df.columns=["node","ID","OD","IDW","ODW","BC","LCC","AC","label"]

        #Labeling the bots
        bots=botnetIP(ctuName) #Botnets in each capture
        for ip in bots:
            df.loc[df["node"] == ip, ["label"]] = 1
            norm_df.loc[norm_df["node"] == ip, ["label"]] = 1

        #Save the dataframe in a csv
        #df.to_csv("../../data/csv/"+ctuName+".csv")
        df.to_csv("./data/csv/"+ctuName+".feat.csv")
        norm_df.to_csv("./data/csv/"+ctuName+".normfeat.csv")
        print("[Py] Output csv file written")
