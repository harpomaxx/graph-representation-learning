import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os

#Create an empty dataframe
df_all = pd.DataFrame()

for root, dirs, files in os.walk("./data/csv"):
    for ctuName in files:
        if(ctuName.find('capture20110817')+1): #Excluding capture number 9 from training
            continue          
        
        if(ctuName.find('norm')+1): #Work only with normalized data
            df=pd.read_csv('./data/csv/'+ctuName)
            #Concat df to df_all
            df_all = pd.concat([df_all, df], ignore_index=True)
#Remove first column
df_all.drop(df_all.columns[0], axis=1, inplace=True)


#Get dataframe column labels
features=df_all.columns.values
X = np.array(df_all[features[1:8]])
#y = np.array(df_all[features[8]])

ar=np.arange(2,16)
ar=ar**2
for k in ar:
    #Run kmeans
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    cluster_labels = kmeans.predict(X)
    df_all['cluster'] = cluster_labels

    #Save only cluster, label and node columns
    df_results = df_all[['node','label','cluster']]
    #Save dataframe as a csv
    df_results.to_csv('./data/csv/normkmeans/'+str(k)+'_results.csv', index=False)

    print('kmeans normalized with k='+str(k)+' written to csv')