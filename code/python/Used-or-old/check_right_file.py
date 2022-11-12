import pandas as pd

#Load data
#Initialize empty df
df=pd.DataFrame()
df=pd.read_csv("./data/csv/phase2_data.csv")
print(df)

#check hob and bob te verify right run
print(df['label'].value_counts())



