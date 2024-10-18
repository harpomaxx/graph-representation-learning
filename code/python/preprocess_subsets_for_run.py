import pandas as pd
import pickle
import torch
import os
import gc

os.makedirs("more_graphs/naive/subsets_for_run", exist_ok=True)

# SUBSETS POR CORRIDA

cap_names = ["10", "11", "12", "15", "15-2", "15-3", "16", "16-2", "16-3", "17", "18", "18-2", "19"]

for j in range(30):
    subsets = {}
    for i in cap_names:
        subsets[str(i)] = {}
        #
        with open(f'more_graphs/naive/capturas_pkl/capturas_{str(i)}.pkl', 'rb') as archivo:
          capturas = pickle.load(archivo)
        #
        # Get all unique IPs from SrcAddr and DstAddr
        unique_ips = pd.unique(capturas[f"cap_subset_{j:02}"][['SrcAddr', 'DstAddr']].values.ravel('K'))
        # Create a mapping from IP address to a unique node index
        ip_to_index = {ip: idx for idx, ip in enumerate(unique_ips)}
        #
        # Group by SrcAddr to calculate sending-related features
        src_features = capturas[f"cap_subset_{j:02}"].groupby('SrcAddr').agg({
            'SrcPkts': 'mean',
            'SrcBytes': 'mean',
            'SAppBytes': 'mean',
            'SrcRate': 'mean'
        }).reset_index()
        # Add missing columns with zeros for nodes that do not appear as SrcAddr
        src_features = src_features.set_index('SrcAddr').reindex(unique_ips, fill_value=0).reset_index()
        #
        # Group by DstAddr to calculate receiving-related features
        dst_features = capturas[f"cap_subset_{j:02}"].groupby('DstAddr').agg({
            'DstPkts': 'mean',
            'DstBytes': 'mean',
            'DAppBytes': 'mean',
            'DstRate': 'mean'
        }).reset_index()
        # Add missing columns with zeros for nodes that do not appear as DstAddr
        dst_features = dst_features.set_index('DstAddr').reindex(unique_ips, fill_value=0).reset_index()
        #
        # Merge src_features and dst_features based on unique IPs
        node_features = pd.merge(src_features, dst_features, left_on='SrcAddr', right_on='DstAddr', how='outer')#.fillna(0)
        # Reorganize columns to create the final node feature vector:
        # [SrcPkts, SrcBytes, SAppBytes, SrcRate, DstPkts, DstBytes, DAppBytes, DstRate]
        node_features = node_features[['SrcPkts', 'SrcBytes', 'SAppBytes', 'SrcRate', 'DstPkts', 'DstBytes', 'DAppBytes', 'DstRate']]
        # Convert node features to tensor
        node_features_tensor = torch.tensor(node_features.values, dtype=torch.float)
        #
        # Group by SrcAddr and DstAddr to create edge features
        edge_features = capturas[f"cap_subset_{j:02}"].groupby(['SrcAddr', 'DstAddr']).agg({
            'Dur': 'mean',
            'TotPkts': 'mean',
            'TotBytes': 'mean',
            'TotAppByte': 'mean',
            'Rate': 'mean',
            'Label': 'max'
        }).reset_index()
        # Convert the SrcAddr and DstAddr pairs to edge indices
        edge_indices = torch.tensor([[ip_to_index[src], ip_to_index[dst]] for src, dst in zip(edge_features['SrcAddr'], edge_features['DstAddr'])], dtype=torch.long).t().contiguous()
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
    with open(f'more_graphs/naive/subsets_for_run/subsets_for_run_{str(j)}.pkl', 'wb') as archivo:
        pickle.dump(subsets, archivo)
    #
    gc.collect()
