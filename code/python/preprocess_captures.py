import pandas as pd
import numpy as np
import pickle
import gc
import os

os.makedirs("more_graphs/naive/capturas_pkl", exis_ok=True)

cap_names = ["10", "11", "12", "15", "15-2", "15-3", "16", "16-2", "16-3", "17", "18", "18-2", "19"]

for i in cap_names:
    capturas = {}
    
    #cap = pd.read_csv(f"/content/unzipped_files/capture201108{str(i)}.csv", dtype={'Sport': object, 'Dport': object})
    cap = pd.read_csv(f"capture201108{str(i)}.csv", dtype={'Sport': object, 'Dport': object})
    cap_copy = cap.copy()
    #
    cap = cap.loc[(cap['Proto'] == "tcp") | (cap['Proto'] == "udp")]
    #
    keep_columns = [0,1,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    cap = cap.iloc[:, keep_columns]
    #
    cap_norm = cap[~cap['Label'].str.contains("From-Botnet")].copy()
    cap_norm['Label'] = cap_norm['Label'].str.contains('From-Botnet').astype(int)
    #
    cap_bot = cap[cap['Label'].str.contains("From-Botnet")].copy()
    cap_bot['Label'] = cap_bot['Label'].str.contains('From-Botnet').astype(int)
    #
    aux_bot=cap_bot.groupby(['SrcAddr', 'DstAddr']).agg({
        'Dur': 'mean',
        'TotPkts': 'mean',
        'TotBytes': 'mean',
        'TotAppByte': 'mean',
        'Rate': 'mean',
        'Label': 'max'
        }).reset_index()
    num_rows = 2*len(aux_bot)
    #
    capturas["original"] = cap_copy
    capturas["filter"] = cap
    capturas["normal"] = cap_norm
    capturas["bot"] = cap_bot
    
    # ESTO LO CAMBIO PORQUE NO ASEGURABA TOMAR FILAS DISTINTAS EN CADA SUBSET
    # for j in range(30):
    #     np.random.seed(42+j*10)
    #     cap_subset = pd.concat([cap_bot, cap_norm.sample(n=num_rows)]).sample(frac=1).reset_index(drop=True)
    #     capturas[f"cap_subset_{j:02}"] = cap_subset
    
    # ESTA ES LA NUEVA VERSION, PERO COMO cap_norm TIENE FILAS REPETIDAS, PUEDE HABER COINCIDENCIAS ENTRE DIFERENTES SUBSETS
    remaining_norm = cap_norm.copy()
    for j in range(30):
        np.random.seed(42+j*10)
        cap_norm_sample = remaining_norm.sample(n=num_rows)
        remaining_norm = remaining_norm.drop(cap_norm_sample.index)
        cap_subset = pd.concat([cap_bot, cap_norm_sample]).sample(frac=1).reset_index(drop=True)
        capturas[f"cap_subset_{j:02}"] = cap_subset
    #
    del cap, cap_copy, cap_norm, cap_bot, aux_bot, cap_subset
    with open(f'more_graphs/naive/capturas_pkl/capturas_{str(i)}.pkl', 'wb') as archivo:
        pickle.dump(capturas, archivo)
    #
    gc.collect()
