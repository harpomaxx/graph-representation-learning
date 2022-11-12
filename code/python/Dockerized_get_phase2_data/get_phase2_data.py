#Run the optimal kmeans to remove the benign cluster from data (without capture 9)
import pandas as pd
import os
from pycaret.clustering import *

#Load data
#Initialize empty df
df_all=pd.DataFrame()
for root, dirs, files in os.walk("./data/csv/captures_features"):
    for ctuName in files:
        #ignore capture 9
        if ctuName == 'capture20110817.binetflow.labels-positive-weights.labeled.csv':
            continue
        df=pd.read_csv("./data/csv/captures_features/"+ctuName)
        #add to general df as new rows
        df_all=df_all.append(df, ignore_index=True)
print(df_all.shape)

#Change background and normal to 0 and infected values to 1
df_all['label']=df_all['label'].replace(['background','normal'],0)
df_all['label']=df_all['label'].replace(['infected'],1)
#drop node column
df_all=df_all.drop(['node'], axis=1)


print(df_all)

df_nolabel=df_all.drop(['label'], axis=1)
cluster = setup(df_nolabel, session_id = 7652)

kmeans = create_model('kmeans',num_clusters=25,n_init=1,n_jobs=1)

kmeans_df = assign_model(kmeans)
#add original labels to the df 
kmeans_df['label']=df_all['label']



#get cluster centroids
#calculate the mean for each column for each cluster
Centroids_df=pd.DataFrame(kmeans_df.groupby('Cluster')['ID'].mean())
Centroids_df['OD']=kmeans_df.groupby('Cluster')['OD'].mean()
Centroids_df['IDW']=kmeans_df.groupby('Cluster')['IDW'].mean()
Centroids_df['ODW']=kmeans_df.groupby('Cluster')['ODW'].mean()
Centroids_df['BC']=kmeans_df.groupby('Cluster')['BC'].mean()
Centroids_df['LCC']=kmeans_df.groupby('Cluster')['LCC'].mean()
Centroids_df['AC']=kmeans_df.groupby('Cluster')['AC'].mean()


#drop data inside the benign cluster
#drop most common cluster
benign_str=kmeans_df['Cluster'].value_counts().idxmax()
phase2_data=kmeans_df[kmeans_df['Cluster']!=benign_str]
#drop Cluster column
phase2_data.drop(['Cluster'], axis=1, inplace=True)


#check hob and bob te verify right run
print(phase2_data['label'].value_counts())


#save data to csv
phase2_data.to_csv('./data/csv/phase2_data.csv', index=False)
Centroids_df.to_csv('./data/csv/phase1_centroids.csv', index=False)


