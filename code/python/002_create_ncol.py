# Note: 002_create_ncol.sh calls this script


import pandas as pd
import sys


cap = sys.argv[1]
csvName = cap + "_ip_bytes.csv"
ncolName = cap + ".ncol"


# read csv file and store it as dataframe
capture = pd.read_csv(csvName)

            
# create dataframes with conections SrcAddr->DstAddr and responses DstAddr->SrcAddr, with their respective weights (bytes)
src2dst = capture[["SrcAddr", "DstAddr", "SrcBytes"]].copy()
dst2src = capture[["DstAddr", "SrcAddr", "DstBytes"]].copy()
src2dst.rename(columns={'SrcAddr': 'origin', 'DstAddr': 'destination', 'SrcBytes': 'weight'}, inplace=True)
dst2src.rename(columns={'DstAddr': 'origin', 'SrcAddr': 'destination', 'DstBytes': 'weight'}, inplace=True)


# concatenate dataframes and add the weights if there are repeated edges
concatAll = pd.concat([src2dst, dst2src], ignore_index=True)
finalDF = concatAll.groupby(['origin','destination'], as_index=False)['weight'].sum().copy()


# create NCOL file to then generate the graph using igraph: g=Graph.Read_Ncol(ncolName, weights=True, directed=True)
finalDF.to_csv(ncolName, sep=' ', header=None, index=None)

