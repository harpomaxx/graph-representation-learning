import pandas as pd
import numpy as np
from optparse import OptionParser
import sys


def parsearg():
    parser = OptionParser(usage="usage: %prog [opt] ",
                          version="%prog 1.0")
    parser.add_option("-n", "--norm",
                      action="store",
                      dest="norm",
                      default=False,
                      help="True for normalized data, anything for raw data")
    (opt, args) = parser.parse_args()
    return opt

opt = parsearg()
if opt.norm != 'True' and opt.norm != 'False':
    print("[Py] normalization parameter wrong or empty. Please use --help for more information")
    sys.exit()

else:   
    txt=''
    if opt.norm == 'True':
        txt='norm'

    #Create a table for the results
    results=pd.DataFrame(columns=['k','HoB','HoB_per','BoB','BoB_per'])

    ar=np.arange(2,16)
    ar=ar**2
    for k in ar:
        #Include the path to the csv file
        df = pd.read_csv('./data/csv/'+txt+'kmeans/'+str(k)+'_results.csv')

        #I'm not sure if zero cluster is always the bigger so that explains the next:
        #Get the number of the benign cluster
        cluster_labels=df['cluster']
        labels=df['label']
        benign_cluster=np.bincount(cluster_labels).argmax()

        #Get number of normal nodes
        #Count the number of zero labels
        number_normal=len(labels[labels==0])
        number_bots=len(labels[labels==1])
        #Keep only the nodes with cluster!=0
        Benign_df=df[df['cluster']!=benign_cluster]

        #Sum of nodes with label=0
        HoB=(Benign_df['label']==0).sum()
        BoB=(Benign_df['label']==1).sum()

        HoB_per=100*HoB/number_normal
        BoB_per=100*BoB/number_bots

        #Add the results to the table
        results.loc[len(results)]=[k,HoB,HoB_per,BoB,BoB_per]

    #Save the table as a csv
    results.to_csv('./data/csv/'+txt+'hob_bob_table.csv', index=False)

