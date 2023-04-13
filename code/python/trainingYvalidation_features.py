import pandas as pd
import numpy as np


df_features = pd.read_csv("all_captures_except_9_FEATURES.pkts.SINnorm.csv")
df_features.iloc[:,8] = df_features.iloc[:,8].replace(to_replace="background", value="normal").copy()

np.random.seed(123)
for etiqueta,grupo in df_features.groupby("label"):
    if (etiqueta == "infected"):
        infected_index = np.random.choice(grupo.index, size=3, replace=False)
    if (etiqueta == "normal"):
        normal_index = np.random.choice(grupo.index, size=344905, replace=False)
     
validationIndices = list(infected_index)+list(normal_index)   

validation = df_features.iloc[validationIndices,:].loc[:,["node","ID","OD","IDW","ODW","label"]].copy()
validationSet = validation.sample(frac=1, random_state=123)

training = df_features.drop(validationIndices).copy()
trainingSet = training.sample(frac=1, random_state=123)

trainingSet.to_csv("training_features.csv", index=None)
validationSet.to_csv("validation_features.csv", index=None)


################################################################################################


df_ncol = pd.read_csv("all_captures_except_9_GRAFOS.pkts.ncol")

np.random.seed(123)
for etiqueta,grupo in df_features.groupby("label"):
    if (etiqueta == "infected"):
        infected_index = np.random.choice(grupo.index, size=3, replace=False)
    if (etiqueta == "normal"):
        normal_index = np.random.choice(grupo.index, size=344905, replace=False)
     
validationIndices = list(infected_index)+list(normal_index)   

validation = df_features.iloc[validationIndices,:].loc[:,["node","ID","OD","IDW","ODW","label"]].copy()
validationSet = validation.sample(frac=1, random_state=123)

training = df_features.drop(validationIndices).copy()
trainingSet = training.sample(frac=1, random_state=123)

trainingSet.to_csv("training_features.csv", index=None)
validationSet.to_csv("validation_features.csv", index=None)


####################

df_ncol = pd.read_csv("/home/tati/Nextcloud/BotChase/graph-representation-learning/rawdata/harpo/all_captures_except_9_GRAFOS.pkts.ncol", sep=" ", header=None)

algunos=df_ncol.sample(375000,random_state=789)
nodosSelec = sorted(pd.DataFrame(pd.concat([algunos.iloc[:,0], algunos.iloc[:,1]], axis=0),columns=["node"])["node"].unique()) #len(nodosSelec)=352996
len(nodosSelec)

validation = df_features[df_features["node"].isin(nodosSelec)]
len(validation.loc[validation["label"]=="infected"])



