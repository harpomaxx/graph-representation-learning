import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os

#Create an empty dataframe
df_all = pd.DataFrame()

for root, dirs, files in os.walk("../../data/csv"):
    for ctuName in files:
        if(ctuName.find(".gz")+1 or ctuName.find('capture20110817')+1): #Excluding capture number 9 from training
            continue

        df=pd.read_csv('../../data/csv/'+ctuName)
        #BC already normalized
        #Normalize ID, OD, IDW, ODW, AC
        for col in ['ID','OD','IDW','ODW','AC']:
            df[col]=(df[col]-df[col].min())/(df[col].max()-df[col].min())
        #Add df values to df_all
        df_all = df_all.append(df)

#Get dataframe column labels
features=df_all.columns.values
X = np.array(df_all[features[1:8]])
y = np.array(df_all[features[8]])

print('Starting kmeans')
#Run kmeans
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
labels = kmeans.predict(X)

#Calcular la moda de labels
mode_labels=np.bincount(labels).argmax()
print(mode_labels)
#Contar la cantidad de veces que aparece la moda
count_mode_labels=np.bincount(labels).max()
print(count_mode_labels)