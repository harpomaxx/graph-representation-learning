import pandas as pd
import pickle
import torch
import os
import gc
import numpy as np

os.makedirs("more_graphs/5elem/subsets_for_run", exist_ok=True)

# SUBSETS POR CORRIDA

cap_names = ["10", "11", "12", "15", "15-2", "15-3", "16", "16-2", "16-3", "17", "18", "18-2", "19"]

for j in range(30):
    subsets = {}
    for i in cap_names:
        subsets[str(i)] = {}
        #
        with open(f'more_graphs/5elem/capturas_pkl/capturas_{str(i)}.pkl', 'rb') as archivo:
          capturas = pickle.load(archivo)
        
        src = capturas[f"cap_subset_{j:02}"][['SrcAddr', 'Sport']].copy()
        src.rename(columns={'SrcAddr':'ip','Sport':'port'}, inplace=True)
        
        dst = capturas[f"cap_subset_{j:02}"][['DstAddr', 'Dport']].copy()
        dst.rename(columns={'DstAddr':'ip','Dport':'port'}, inplace=True)
        
        unique_pairs = pd.concat([src,dst]).drop_duplicates()
        unique_pairs['pair_id'] = unique_pairs.groupby(['ip','port']).ngroup()
        unique_pairs = unique_pairs.sort_values('pair_id').set_index('pair_id').reset_index()
        unique_pairs['pairs'] = unique_pairs.apply(lambda x: str(x['ip']) + ',' + str(x['port']), axis=1)
        
        pair_to_index = {pair: idx for idx, pair in zip(unique_pairs['pair_id'], unique_pairs['pairs'])} 
        
        ##unique_pairs_src = unique_pairs.copy()
        ##unique_pairs_src.rename(columns={'ip':'SrcAddr','port':'Sport'}, inplace=True)
        
        ##unique_pairs_dst = unique_pairs.copy()
        ##unique_pairs_dst.rename(columns={'ip':'DstAddr','port':'Dport'}, inplace=True)
        
        # Group by SrcAddr to calculate sending-related features
        src_features = capturas[f"cap_subset_{j:02}"].groupby(['SrcAddr', 'Sport']).agg({        # AGREGO puerto, pero no se como considerar el protocolo
            'SrcPkts': 'mean',
            'SrcBytes': 'mean',
            'SAppBytes': 'mean',
            'SrcRate': 'mean'
        }).reset_index()
        src_features['src_pairs'] = src_features.apply(lambda x: str(x['SrcAddr']) + ',' + str(x['Sport']), axis=1)
        src_features = src_features.set_index('src_pairs').reindex(np.array(unique_pairs['pairs']), fill_value=0).reset_index()
        
        # Add missing columns with zeros for nodes that do not appear as SrcAddr
        ##full_src_features=unique_pairs_src.merge(src_features, on=['SrcAddr','Sport'],how='left').fillna(0)
        ##full_src_features.set_index('pair_id',inplace=True)
        
        #src_features = src_features.set_index('SrcAddr').reindex(unique_ips, fill_value=0).reset_index()
        #
        # Group by DstAddr to calculate receiving-related features
        dst_features = capturas[f"cap_subset_{j:02}"].groupby(['DstAddr', 'Dport']).agg({        # AGREGO puerto, pero no se como considerar el protocolo
            'DstPkts': 'mean',
            'DstBytes': 'mean',
            'DAppBytes': 'mean',
            'DstRate': 'mean'
        }).reset_index()
        dst_features['dst_pairs'] = dst_features.apply(lambda x: str(x['DstAddr']) + ',' + str(x['Dport']), axis=1)
        dst_features = dst_features.set_index('dst_pairs').reindex(np.array(unique_pairs['pairs']), fill_value=0).reset_index()

        # Add missing columns with zeros for nodes that do not appear as DstAddr
        ##full_dst_features=unique_pairs_dst.merge(dst_features, on=['DstAddr','Dport'],how='left').fillna(0)
        ##full_dst_features.set_index('pair_id',inplace=True)
        
        #dst_features = dst_features.set_index('DstAddr').reindex(unique_ips, fill_value=0).reset_index()
        #
        node_features = pd.merge(src_features, dst_features, left_on=['src_pairs'], right_on=['dst_pairs'], how='outer')
        # Merge src_features and dst_features based on unique IPs
        ##node_features = pd.merge(full_src_features, full_dst_features, left_on=['SrcAddr','Sport'], right_on=['DstAddr','Dport'], how='outer')#.fillna(0)
        # Reorganize columns to create the final node feature vector:
        # [SrcPkts, SrcBytes, SAppBytes, SrcRate, DstPkts, DstBytes, DAppBytes, DstRate]
        node_features = node_features[['SrcPkts', 'SrcBytes', 'SAppBytes', 'SrcRate', 'DstPkts', 'DstBytes', 'DAppBytes', 'DstRate']]
        # Convert node features to tensor
        node_features_tensor = torch.tensor(node_features.values, dtype=torch.float)
        #
        # Group by SrcAddr and DstAddr to create edge features
        edge_features = capturas[f"cap_subset_{j:02}"].groupby(['SrcAddr', 'DstAddr', 'Proto', 'Sport', 'Dport']).agg({
            'Dur': 'mean',
            'TotPkts': 'mean',
            'TotBytes': 'mean',
            'TotAppByte': 'mean',
            'Rate': 'mean',
            'Label': 'max'
        }).reset_index()
        edge_features['src_pairs'] = edge_features.apply(lambda x: str(x['SrcAddr']) + ',' + str(x['Sport']), axis=1)
        edge_features['dst_pairs'] = edge_features.apply(lambda x: str(x['DstAddr']) + ',' + str(x['Dport']), axis=1)
                
        edge_indices = torch.tensor([[pair_to_index[src], pair_to_index[dst]] for src, dst in zip(edge_features['src_pairs'], edge_features['dst_pairs'])], dtype=torch.long).t().contiguous()
        # Convert edge features to tensor
        edge_features_tensor = torch.tensor(edge_features[['Dur', 'TotPkts', 'TotBytes', 'TotAppByte', 'Rate']].values, dtype=torch.float)
        # Convert edge labels to tensor
        edge_labels_tensor = torch.tensor(edge_features['Label'].values, dtype=torch.long)
        #
        subsets[str(i)]["unique_ips"] = unique_ips
        subsets[str(i)]["ip_to_index"] = ip_to_index
        subsets[str(i)]["node_features_tensor"] = node_features_tensor
        subsets[str(i)]["edge_indices"] = edge_indices
        subsets[str(i)]["edge_features_tensor"] = edge_features_tensor
        subsets[str(i)]["edge_labels_tensor"] = edge_labels_tensor
        #
        del unique_ips, ip_to_index, node_features_tensor, edge_indices, edge_features_tensor, edge_labels_tensor
    with open(f'more_graphs/5elem/subsets_for_run/subsets_for_run_{str(j)}.pkl', 'wb') as archivo:
        pickle.dump(subsets, archivo)
    #
    gc.collect()
