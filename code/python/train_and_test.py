import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import pickle
import os

from gnn_models import *

import sys

SUBSET_PATH = sys.argv[1]
RESULTS_PATH = sys.argv[2]
BATCH_SIZE_TRAIN = int(sys.argv[3])
BATCH_SIZE_TEST = int(sys.argv[4])
HIDDEN_CHANNELS_M1 = int(sys.argv[5])
HIDDEN_CHANNELS_L1_M2 = int(sys.argv[6])
HIDDEN_CHANNELS_L2_M2 = int(sys.argv[7])
OUTPUT_CHANNELS = int(sys.argv[8])
LEARNING_RATE = float(sys.argv[9])
NUM_EPOCHS = int(sys.argv[10])
RUNS = int(sys.argv[11])

device = torch.device('cpu') #'cuda' if torch.cuda.is_available() else 'cpu')

cap_names_cv = ["10", "11", "12", "15", "15-2", "15-3", "16", "16-3", "18-2", "19"]

for run in range(RUNS):
    print("run:", run)
    #print(os.path.join(str(SUBSET_PATH), f"subsets_for_run_{run}.pkl"))
    #with open('/home/tati/ctu13_pyg/subsets_for_run/subsets_for_run_0.pkl','rb') as filename:
    with open(os.path.join(str(SUBSET_PATH), f"subsets_for_run_{run}.pkl"), 'rb') as filename:
        subsets = pickle.load(filename)

    # Graphs for train/validation
    graph10 = Data(
      x = subsets["10"]["node_features_tensor"],       # Node features
      edge_index = subsets["10"]["edge_indices"],      # Edge index
      edge_attr = subsets["10"]["edge_features_tensor"],  # Edge features
      edge_label = subsets["10"]["edge_labels_tensor"]   # Edge labels
    )
    graph11 = Data(
      x = subsets["11"]["node_features_tensor"],       # Node features
      edge_index = subsets["11"]["edge_indices"],      # Edge index
      edge_attr = subsets["11"]["edge_features_tensor"],  # Edge features
      edge_label = subsets["11"]["edge_labels_tensor"]   # Edge labels
    )
    graph12 = Data(
      x = subsets["12"]["node_features_tensor"],       # Node features
      edge_index = subsets["12"]["edge_indices"],      # Edge index
      edge_attr = subsets["12"]["edge_features_tensor"],  # Edge features
      edge_label = subsets["12"]["edge_labels_tensor"]   # Edge labels
    )
    graph15 = Data(
      x = subsets["15"]["node_features_tensor"],       # Node features
      edge_index = subsets["15"]["edge_indices"],      # Edge index
      edge_attr = subsets["15"]["edge_features_tensor"],  # Edge features
      edge_label = subsets["15"]["edge_labels_tensor"]   # Edge labels
    )
    graph152 = Data(
      x = subsets["15-2"]["node_features_tensor"],       # Node features
      edge_index = subsets["15-2"]["edge_indices"],      # Edge index
      edge_attr = subsets["15-2"]["edge_features_tensor"],  # Edge features
      edge_label = subsets["15-2"]["edge_labels_tensor"]   # Edge labels
    )
    graph153 = Data(
      x = subsets["15-3"]["node_features_tensor"],       # Node features
      edge_index = subsets["15-3"]["edge_indices"],      # Edge index
      edge_attr = subsets["15-3"]["edge_features_tensor"],  # Edge features
      edge_label = subsets["15-3"]["edge_labels_tensor"]   # Edge labels
    )
    graph16 = Data(
      x = subsets["16"]["node_features_tensor"],       # Node features
      edge_index = subsets["16"]["edge_indices"],      # Edge index
      edge_attr = subsets["16"]["edge_features_tensor"],  # Edge features
      edge_label = subsets["16"]["edge_labels_tensor"]   # Edge labels
    )
    graph163 = Data(
      x = subsets["16-3"]["node_features_tensor"],       # Node features
      edge_index = subsets["16-3"]["edge_indices"],      # Edge index
      edge_attr = subsets["16-3"]["edge_features_tensor"],  # Edge features
      edge_label = subsets["16-3"]["edge_labels_tensor"]   # Edge labels
    )
    graph182 = Data(
      x = subsets["18-2"]["node_features_tensor"],       # Node features
      edge_index = subsets["18-2"]["edge_indices"],      # Edge index
      edge_attr = subsets["18-2"]["edge_features_tensor"],  # Edge features
      edge_label = subsets["18-2"]["edge_labels_tensor"]   # Edge labels
    )
    graph19 = Data(
      x = subsets["19"]["node_features_tensor"],       # Node features
      edge_index = subsets["19"]["edge_indices"],      # Edge index
      edge_attr = subsets["19"]["edge_features_tensor"],  # Edge features
      edge_label = subsets["19"]["edge_labels_tensor"]   # Edge labels
    )

    # List of all graphs for training/validation
    graphs = [graph10, graph11, graph12, graph15, graph152, graph153, graph16, graph163, graph182, graph19]

    # Graphs for testing
    test1_data = Data(
          x = subsets["18"]["node_features_tensor"],       # Node features
          edge_index = subsets["18"]["edge_indices"],      # Edge index
          edge_attr = subsets["18"]["edge_features_tensor"],  # Edge features
          edge_label = subsets["18"]["edge_labels_tensor"]   # Edge labels
        )
    test1_loader = DataLoader([test1_data], batch_size=BATCH_SIZE_TEST)

    test2_data = Data(
          x = subsets["16-2"]["node_features_tensor"],       # Node features
          edge_index = subsets["16-2"]["edge_indices"],      # Edge index
          edge_attr = subsets["16-2"]["edge_features_tensor"],  # Edge features
          edge_label = subsets["16-2"]["edge_labels_tensor"]   # Edge labels
        )
    test2_loader = DataLoader([test2_data], batch_size=BATCH_SIZE_TEST)


    # Training and testing

    df_losses_m1 = pd.DataFrame()
    df_losses_m2 = pd.DataFrame()

    loss_t1_m1_l = []
    accuracy_t1_m1_l = []
    precision_t1_m1_l = []
    recall_t1_m1_l = []
    f1_t1_m1_l = []
    loss_t1_m2_l = []
    accuracy_t1_m2_l = []
    precision_t1_m2_l = []
    recall_t1_m2_l = []
    f1_t1_m2_l = []
    loss_t2_m1_l = []
    accuracy_t2_m1_l = []
    precision_t2_m1_l = []
    recall_t2_m1_l = []
    f1_t2_m1_l = []
    loss_t2_m2_l = []
    accuracy_t2_m2_l = []
    precision_t2_m2_l = []
    recall_t2_m2_l = []
    f1_t2_m2_l = []

    pred_test1 = {'labels': None, 'model1': {}, 'model2': {}}
    pred_test2 = {'labels': None, 'model1': {}, 'model2': {}}


    # leave-one-graph-out cross-validation

    num_graphs = len(graphs)

    for i_graph in range(num_graphs):
        print(i_graph)
        
        # Split data into training and validation sets
        train_graphs = [graphs[k] for k in range(num_graphs) if k != i_graph]  # All graphs except the i-th one
        val_graph = [graphs[i_graph]]  # The i-th graph for validation
        
        # Create DataLoaders for training and validation
        train_loader1 = DataLoader(train_graphs, batch_size=BATCH_SIZE_TRAIN, shuffle=False)
        train_loader2 = DataLoader(train_graphs, batch_size=BATCH_SIZE_TRAIN, shuffle=False)
        val_loader = DataLoader(val_graph, batch_size=BATCH_SIZE_TEST, shuffle=False)
        
        # Print which graph is being used for validation
        #print(f"Fold {i + 1}/{num_graphs}: Training on {num_graphs - 1} graphs, Validating on graph {i + 1}")
        
        # Parameters
        in_channels_node = train_graphs[0].x.size(1)  # Number of node features
        in_channels_edge = train_graphs[0].edge_attr.size(1)  # Number of edge features
        hidden_channels = HIDDEN_CHANNELS_M1  # Hidden dimension size model1
        hidden_channels1 = HIDDEN_CHANNELS_L1_M2 # Hidden dimension size model2
        hidden_channels2 = HIDDEN_CHANNELS_L2_M2 # Hidden dimension size model2
        out_channels = OUTPUT_CHANNELS
        
        # Initialize the model 1
        model1 = EGraphSAGE1(in_channels_node, in_channels_edge, hidden_channels, out_channels)
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=LEARNING_RATE)
        
        # Initialize the model 2
        model2 = EGraphSAGE2(in_channels_node, in_channels_edge, hidden_channels1, hidden_channels2, out_channels).to(device)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=LEARNING_RATE)
        
        # Training for multiple epochs
        loss_m1 = []
        loss_val_m1 = []
        
        loss_m2 = []
        loss_val_m2 = []
        
        for epoch in range(1, NUM_EPOCHS + 1):
            loss_m1.append(train(model1, optimizer1, train_loader1))
            loss_val_m1.append(test(model1, val_loader)[0])

            loss_m2.append(train(model2, optimizer2, train_loader2))
            loss_val_m2.append(test(model2, val_loader)[0])
            
        df_losses_m1 = pd.concat([df_losses_m1, pd.DataFrame({f"loss_{i_graph:02}": loss_m1, f"loss_val_{i_graph:02}": loss_val_m1})], axis=1)
        df_losses_m2 = pd.concat([df_losses_m2, pd.DataFrame({f"loss_{i_graph:02}": loss_m2, f"loss_val_{i_graph:02}": loss_val_m2})], axis=1)
        
        # Test 1
        loss_t1_m1, accuracy_t1_m1, precision_t1_m1, recall_t1_m1, f1_t1_m1, all_labels_t1, all_preds_t1_m1 = test(model1,test1_loader)
        loss_t1_m2, accuracy_t1_m2, precision_t1_m2, recall_t1_m2, f1_t1_m2, _, all_preds_t1_m2 = test(model2,test1_loader)
        
        # Test 2
        loss_t2_m1, accuracy_t2_m1, precision_t2_m1, recall_t2_m1, f1_t2_m1, all_labels_t2, all_preds_t2_m1 = test(model1,test2_loader)
        loss_t2_m2, accuracy_t2_m2, precision_t2_m2, recall_t2_m2, f1_t2_m2, _, all_preds_t2_m2 = test(model2,test2_loader)
        
        loss_t1_m1_l.append(loss_t1_m1)
        accuracy_t1_m1_l.append(accuracy_t1_m1)
        precision_t1_m1_l.append(precision_t1_m1)
        recall_t1_m1_l.append(recall_t1_m1)
        f1_t1_m1_l.append(f1_t1_m1)
        
        loss_t1_m2_l.append(loss_t1_m2)
        accuracy_t1_m2_l.append(accuracy_t1_m2)
        precision_t1_m2_l.append(precision_t1_m2)
        recall_t1_m2_l.append(recall_t1_m2)
        f1_t1_m2_l.append(f1_t1_m2)
        
        loss_t2_m1_l.append(loss_t2_m1)
        accuracy_t2_m1_l.append(accuracy_t2_m1)
        precision_t2_m1_l.append(precision_t2_m1)
        recall_t2_m1_l.append(recall_t2_m1)
        f1_t2_m1_l.append(f1_t2_m1)
        
        loss_t2_m2_l.append(loss_t2_m2)
        accuracy_t2_m2_l.append(accuracy_t2_m2)
        precision_t2_m2_l.append(precision_t2_m2)
        recall_t2_m2_l.append(recall_t2_m2)
        f1_t2_m2_l.append(f1_t2_m2)
        
        if i_graph==0:
            pred_test1["labels"] = all_labels_t1
            pred_test2["labels"] = all_labels_t2
            
        pred_test1["model1"][i_graph] = all_preds_t1_m1
        pred_test1["model2"][i_graph] = all_preds_t1_m2
        
        pred_test2["model1"][i_graph] = all_preds_t2_m1
        pred_test2["model2"][i_graph] = all_preds_t2_m2
        
    df_metrics_test1 = pd.DataFrame({'loss_m1': loss_t1_m1_l,
                                    'acc_m1': accuracy_t1_m1_l,
                                    'prec_m1': precision_t1_m1_l,
                                    'rec_m1': recall_t1_m1_l,
                                    'f1_m1': f1_t1_m1_l,
                                    'loss_m2': loss_t1_m2_l,
                                    'acc_m2': accuracy_t1_m2_l,
                                    'prec_m2': precision_t1_m2_l,
                                    'rec_m2': recall_t1_m2_l,
                                    'f1_m2': f1_t1_m2_l
                                    })
    
    df_metrics_test2 = pd.DataFrame({'loss_m1': loss_t2_m1_l,
                                    'acc_m1': accuracy_t2_m1_l,
                                    'prec_m1': precision_t2_m1_l,
                                    'rec_m1': recall_t2_m1_l,
                                    'f1_m1': f1_t2_m1_l,
                                    'loss_m2': loss_t2_m2_l,
                                    'acc_m2': accuracy_t2_m2_l,
                                    'prec_m2': precision_t2_m2_l,
                                    'rec_m2': recall_t2_m2_l,
                                    'f1_m2': f1_t2_m2_l
                                    })
    
    results_subdir = os.path.join(str(RESULTS_PATH), f"run_{run:02}")
    os.makedirs(results_subdir, exist_ok=True)
    
    df_metrics_test1.to_csv(os.path.join(results_subdir, "metrics_t1.csv"))
    df_metrics_test2.to_csv(os.path.join(results_subdir, "metrics_t2.csv"))
    
    df_losses_m1.to_csv(os.path.join(results_subdir, "losses_m1.csv"))
    df_losses_m2.to_csv(os.path.join(results_subdir, "losses_m2.csv"))

    with open(os.path.join(results_subdir, 'pred_test1.pkl'), 'wb') as filename:
        pickle.dump(pred_test1, filename)

    with open(os.path.join(results_subdir, 'pred_test2.pkl'), 'wb') as filename:
        pickle.dump(pred_test2, filename)


    # PLOTS:

    ################## Losses model 1  #################################
    
    # Configurar la grilla de 10 plots (2 filas x 5 columnas)
    fig, axs = plt.subplots(2, 5, figsize=(15, 8), sharey=True)

    # Contador de "runs" para los títulos
    run_counter = 0

    # Iterar sobre cada par de columnas (0-1, 2-3, ...)
    for i in range(2):
        for j in range(5):
            # Índices de las columnas a graficar
            col1 = (run_counter * 2)
            col2 = col1 + 1

            # Graficar los valores de las columnas col1 y col2 vs range(200)
            _=axs[i, j].plot(range(200), df_losses_m1.iloc[:, col1], label='train')
            _=axs[i, j].plot(range(200), df_losses_m1.iloc[:, col2], label='val')

            # Añadir título y leyenda
            _=axs[i, j].set_title(f'val cap_{cap_names_cv[j+i*5]}')
            _=axs[i, j].legend()

            # Incrementar el contador de "runs"
            run_counter += 1

    _=fig.suptitle(f"Run {run:02}: Training loss and validation loss for Model 1\n", fontsize=18)
    _=fig.text(0.5, 0.04, 'Epochs', ha='center', va='center', fontsize=14)
    _=fig.text(0.08, 0.5, 'Loss', ha='center', va='center', rotation='vertical', fontsize=14)

    # Ajustar el layout para evitar solapamientos
    # plt.tight_layout()
    
    plt.savefig(os.path.join(results_subdir,"plot_losses_m1.pdf"))
    plt.savefig(os.path.join(results_subdir,"plot_losses_m1.png"))


    ################## Losses model 2  #################################

    # Configurar la grilla de 10 plots (2 filas x 5 columnas)
    fig, axs = plt.subplots(2, 5, figsize=(15, 8), sharey=True)

    # Contador de "runs" para los títulos
    run_counter = 0

    # Iterar sobre cada par de columnas (0-1, 2-3, ...)
    for i in range(2):
        for j in range(5):
            # Índices de las columnas a graficar
            col1 = (run_counter * 2)
            col2 = col1 + 1

            # Graficar los valores de las columnas col1 y col2 vs range(200)
            _=axs[i, j].plot(range(200), df_losses_m2.iloc[:, col1], label='train')
            _=axs[i, j].plot(range(200), df_losses_m2.iloc[:, col2], label='val')

            # Añadir título y leyenda
            _=axs[i, j].set_title(f'val cap_{cap_names_cv[j+i*5]}')
            _=axs[i, j].legend()

            # Incrementar el contador de "runs"
            run_counter += 1

    _=fig.suptitle(f"Run {run:02}: Training loss and validation loss for Model 2\n", fontsize=18)
    _=fig.text(0.5, 0.04, 'Epochs', ha='center', va='center', fontsize=14)
    _=fig.text(0.08, 0.5, 'Loss', ha='center', va='center', rotation='vertical', fontsize=14)

    # Ajustar el layout para evitar solapamientos
    # plt.tight_layout()
    
    plt.savefig(os.path.join(results_subdir,"plot_losses_m2.pdf"))
    plt.savefig(os.path.join(results_subdir,"plot_losses_m2.png"))


    ################## Boxplot metrics test1 (capture 18) #################################
    df_metrics_test1=pd.read_csv(os.path.join(results_subdir,"metrics_t1.csv"))
    df_metrics_test2=pd.read_csv(os.path.join(results_subdir,"metrics_t2.csv"))
    
    data_model1 = df_metrics_test1.iloc[:, 2:6]
    data_model2 = df_metrics_test1.iloc[:, 7:11]
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the boxplot for Model 1 with slightly shifted positions to avoid overlap
    boxplot1 = ax.boxplot(data_model1.values, positions=[1,4,7,10], widths=0.6, patch_artist=True, medianprops={'color': 'black'})
    
    # Plot the boxplot for Model 2 with slightly shifted positions
    boxplot2 = ax.boxplot(data_model2.values, positions=[2,5,8,11], widths=0.6, patch_artist=True, medianprops={'color': 'black'})

    # Fill colors for Model 1 and Model 2
    for box in boxplot1['boxes']:
        _=box.set(facecolor='tab:blue', edgecolor='black')

    for box in boxplot2['boxes']:
        _=box.set(facecolor='tab:orange', edgecolor='black')

    # Customizing the plot
    _=ax.set_xticks([0, 1.5, 4.5, 7.5, 10.5])  # Center the labels between the pairs of boxes
    _=ax.set_xticklabels(['','Accuracy', 'Precision', 'Recall', 'F1-score'])
    _=ax.set_title(f'Run {run:02}: Distribution metrics test1 (capture20110818)')
    #_=ax.set_ylabel('Values')

    # Add a legend to indicate which boxplot represents which model
    legend_elements = [plt.Line2D([0], [0], color='tab:blue', lw=6, label='Model 1'),
                       plt.Line2D([0], [0], color='tab:orange', lw=6, label='Model 2')]
    _=ax.legend(handles=legend_elements, loc='upper right')

    # Show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(results_subdir,"boxplot_metrics_test1.pdf"))
    plt.savefig(os.path.join(results_subdir,"boxplot_metrics_test1.png"))
        

    ################## Boxplot metrics test2 (capture 16-2) #################################

    data_model1 = df_metrics_test2.iloc[:, 2:6]
    data_model2 = df_metrics_test2.iloc[:, 7:11]
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the boxplot for Model 1 with slightly shifted positions to avoid overlap
    boxplot1 = ax.boxplot(data_model1.values, positions=[1,4,7,10], widths=0.6, patch_artist=True, medianprops={'color': 'black'})
    
    # Plot the boxplot for Model 2 with slightly shifted positions
    boxplot2 = ax.boxplot(data_model2.values, positions=[2,5,8,11], widths=0.6, patch_artist=True, medianprops={'color': 'black'})

    # Fill colors for Model 1 and Model 2
    for box in boxplot1['boxes']:
        _=box.set(facecolor='tab:blue', edgecolor='black')

    for box in boxplot2['boxes']:
        _=box.set(facecolor='tab:orange', edgecolor='black')

    # Customizing the plot
    _=ax.set_xticks([0, 1.5, 4.5, 7.5, 10.5])  # Center the labels between the pairs of boxes
    _=ax.set_xticklabels(['','Accuracy', 'Precision', 'Recall', 'F1-score'])
    _=ax.set_title(f'Run {run:02}: Distribution metrics test2 (capture20110816-2)')
    #_=ax.set_ylabel('Values')

    # Add a legend to indicate which boxplot represents which model
    legend_elements = [plt.Line2D([0], [0], color='tab:blue', lw=6, label='Model 1'), plt.Line2D([0], [0], color='tab:orange', lw=6, label='Model 2')]
    _=ax.legend(handles=legend_elements, loc='upper right')

    # Show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(results_subdir,"boxplot_metrics_test2.pdf"))
    plt.savefig(os.path.join(results_subdir,"boxplot_metrics_test2.png"))
    

    ################## Metrics test1 (capture 18) #################################
    
    # Configurar la grilla de 10 plots (2 filas x 5 columnas)
    fig, axs = plt.subplots(2, 5, figsize=(15, 8), sharey=True)

    # Nombres para los xticks (las 4 comparaciones)
    xtick_labels = ['acc', 'prec', 'rec', 'f1']

    # Iterar sobre cada fila del DataFrame
    for idx, row in enumerate(df_metrics_test1.iterrows()):
        i, j = divmod(idx, 5)  # Calcular el índice para las subplots (filas y columnas)

        # Extraer los valores de cada fila para graficar
        group_1 = row[1].iloc[[2, 3, 4, 5]]
        group_2 = row[1].iloc[[7, 8, 9, 10]]

        # Crear el gráfico de barras comparando los grupos
        _=axs[i, j].bar(np.arange(4) - 0.2, group_1, width=0.4, label='Model 1')
        _=axs[i, j].bar(np.arange(4) + 0.2, group_2, width=0.4, label='Model 2')

        # Añadir título, xticks y leyenda
        _=axs[i, j].set_title(f'val cap_{cap_names_cv[j+i*5]}')
        _=axs[i, j].set_xticks(np.arange(4))  # Colocar los xticks en las posiciones correctas
        _=axs[i, j].set_xticklabels(xtick_labels, rotation=45, ha='right')  # Colocar los nombres de los xticks
        _=axs[i, j].legend()

    _=fig.suptitle(f"Run {run:02}: Metrics test1 (capture20110818)\n", fontsize=18)
    # Ajustar el layout para evitar solapamientos
    plt.tight_layout()

    plt.savefig(os.path.join(results_subdir,"metrics_test1.pdf"))
    plt.savefig(os.path.join(results_subdir,"metrics_test1.png"))
    

    ################## Metrics test2 (capture 16-2) #################################

    # Configurar la grilla de 10 plots (2 filas x 5 columnas)
    fig, axs = plt.subplots(2, 5, figsize=(15, 8), sharey=True)

    # Nombres para los xticks (las 4 comparaciones)
    xtick_labels = ['acc', 'prec', 'rec', 'f1']

    # Iterar sobre cada fila del DataFrame
    for idx, row in enumerate(df_metrics_test2.iterrows()):
        i, j = divmod(idx, 5)  # Calcular el índice para las subplots (filas y columnas)

        # Extraer los valores de cada fila para graficar
        group_1 = row[1].iloc[[2, 3, 4, 5]]
        group_2 = row[1].iloc[[7, 8, 9, 10]]

        # Crear el gráfico de barras comparando los grupos
        _=axs[i, j].bar(np.arange(4) - 0.2, group_1, width=0.4, label='Model 1')
        _=axs[i, j].bar(np.arange(4) + 0.2, group_2, width=0.4, label='Model 2')

        # Añadir título, xticks y leyenda
        _=axs[i, j].set_title(f'val cap_{cap_names_cv[j+i*5]}')
        _=axs[i, j].set_xticks(np.arange(4))  # Colocar los xticks en las posiciones correctas
        _=axs[i, j].set_xticklabels(xtick_labels, rotation=45, ha='right')  # Colocar los nombres de los xticks
        _=axs[i, j].legend()

    _=fig.suptitle(f"Run {run:02}: Metrics test2 (capture20110816-2)\n", fontsize=18)
    # Ajustar el layout para evitar solapamientos
    plt.tight_layout()

    plt.savefig(os.path.join(results_subdir,"metrics_test2.pdf"))
    plt.savefig(os.path.join(results_subdir,"metrics_test2.png"))
    

