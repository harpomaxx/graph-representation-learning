import pandas as pd
import igraph as ig
import os
import ipaddress

def parsearg():
    parser = OptionParser(usage="usage: %prog [opt] ",
                          version="%prog 1.0")
    parser.add_option("-i", "--input",
                      action="store",
                      dest="input",
                      default=False,
                      help="set name of the input file")
    parser.add_option("-o", "--output",
                      action="store", # optional because action defaults to "store"
                      dest="output",
                      help="set name of the output file",)
    (opt, args) = parser.parse_args()
    return opt


def ip_read(ip_str):
    aux_str=ip_str 
    if aux_str.count(":")==5:#For ipaddress to work for 6 bytes directions too
        ip=int(ipaddress.ip_address('00:00:'+ip_str))
    else:
        ip=int(ipaddress.ip_address(ip_str))
    return ip


def capture2graphml(ctuName):
    df=pd.read_csv('../../rawdata/ctu-13/'+ctuName)
    #Keep the tcp and udp comunications only.
    df=df[(df['Proto']=='tcp') | (df['Proto']=='udp')]
    #Keep the columns we need
    df=df[["SrcAddr","DstAddr","TotBytes","SrcBytes"]]
    df["DstBytes"]=df["TotBytes"]-df["SrcBytes"]
    df.drop('TotBytes',inplace=True, axis=1)

            
    idx=df['SrcAddr'].transform(ip_read)>df['DstAddr'].transform(ip_read) #Indexes where src>dst ips

    aux=df[idx]['SrcBytes'].copy() #Reorder bytes according to new address order
    df.loc[idx,'SrcBytes']=df[idx]['DstBytes']
    df.loc[idx,'DstBytes']=aux

    aux=df[idx]['SrcAddr'].copy()  #Reorders addresses
    df.loc[idx,'SrcAddr']=df[idx]['DstAddr']
    df.loc[idx,'DstAddr']=aux

    df_gb=df.groupby(['SrcAddr','DstAddr'])
    df_agg=df_gb.agg({
        'SrcBytes':'sum',
        'DstBytes':'sum'
    })
    df=df_agg.reset_index()


    df21=df[["SrcAddr","DstAddr","SrcBytes"]].rename(columns={"SrcBytes":"Bytes"})#src to dst edges
    df22=df[["DstAddr","SrcAddr","DstBytes"]].rename(columns={"DstBytes":"Bytes","DstAddr":"SrcAddr","SrcAddr":"DstAddr"})#Aristas dst a src
    df = pd.concat([df21,df22], axis=0)
    df=df.reset_index(drop=True)
    df.drop(df[df["Bytes"]==0].index,inplace=True, axis=0)#Edges with weight=0 removed
    df=df.reset_index(drop=True)

    g = ig.Graph.DataFrame(df,directed=True)
    g.write_graphml("../../data/graphml/"+ctuName+".graphml")

#Iterative version
for root, dirs, files in os.walk("../../rawdata/ctu-13"):
    for ctuName in files:
        if(ctuName.find(".dvc")+1 or ctuName.find(".gitignore")+1 or ctuName.find(".md")+1):
            continue
        capture2graphml(ctuName)
        print(ctuName+".graphml"+" done")

