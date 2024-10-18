import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

RESULTS_PATH = sys.argv[1]
RUNS = int(sys.argv[2])

cap_names_cv = ["10", "11", "12", "15", "15-2", "15-3", "16", "16-3", "18-2", "19"]

dict_losses_m1 = {}
dict_losses_m2 = {}

dict_metrics_t1 = {} 
dict_metrics_t2 = {} 


count = 0
for k in range(len(cap_names_cv)):
    #
    dict_losses_m1[str(cap_names_cv[k])] = pd.DataFrame()
    dict_losses_m2[str(cap_names_cv[k])] = pd.DataFrame()
    #
    dict_metrics_t1[str(cap_names_cv[k])] = pd.DataFrame()
    dict_metrics_t2[str(cap_names_cv[k])] = pd.DataFrame()
    
    for run in range(RUNS):
        df_losses_m1 = pd.read_csv(os.path.join(RESULTS_PATH, f"run_{run:02}", "losses_m1.csv")).iloc[:,1:]
        df_losses_m1 = df_losses_m1.add_prefix(f'run_{run:02}_')
        
        df_losses_m2 = pd.read_csv(os.path.join(RESULTS_PATH, f"run_{run:02}", "losses_m2.csv")).iloc[:,1:]
        df_losses_m2 = df_losses_m2.add_prefix(f'run_{run:02}_')
        
        df_metrics_t1 = pd.read_csv(os.path.join(RESULTS_PATH, f"run_{run:02}", "metrics_t1.csv")).iloc[k:k+1,1:]
        dict_metrics_t1[str(cap_names_cv[k])] = pd.concat([dict_metrics_t1[str(cap_names_cv[k])], df_metrics_t1],axis=0)
        
        df_metrics_t2 = pd.read_csv(os.path.join(RESULTS_PATH, f"run_{run:02}", "metrics_t2.csv")).iloc[k:k+1,1:]
        dict_metrics_t2[str(cap_names_cv[k])] = pd.concat([dict_metrics_t2[str(cap_names_cv[k])], df_metrics_t2],axis=0)
        
        col1 = count*2
        col2 = col1 + 1
        dict_losses_m1[str(cap_names_cv[k])] = pd.concat([dict_losses_m1[str(cap_names_cv[k])], df_losses_m1.iloc[:,[col1,col2]]], axis=1)
        dict_losses_m2[str(cap_names_cv[k])] = pd.concat([dict_losses_m2[str(cap_names_cv[k])], df_losses_m2.iloc[:,[col1,col2]]], axis=1)
    count+=1
    
    
    os.makedirs(os.path.join(RESULTS_PATH, "comparative_runs", "losses_plots", "model1"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_PATH, "comparative_runs", "losses_plots", "model2"), exist_ok=True)
    
    os.makedirs(os.path.join(RESULTS_PATH, "comparative_runs", "metrics_plots", "test1"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_PATH, "comparative_runs", "metrics_plots", "test2"), exist_ok=True)
    
    losses_dir_m1 = os.path.join(RESULTS_PATH, "comparative_runs", "losses_plots", "model1")
    losses_dir_m2 = os.path.join(RESULTS_PATH, "comparative_runs", "losses_plots", "model2")
    
    metrics_dir_t1 = os.path.join(RESULTS_PATH, "comparative_runs", "metrics_plots", "test1")
    metrics_dir_t2 = os.path.join(RESULTS_PATH, "comparative_runs", "metrics_plots", "test2")
    
    ################# LOSSES Model 1 ##################################
    
    # Configurar la grilla de 30 plots (5 filas x 6 columnas)
    fig, axs = plt.subplots(5, 6, figsize=(20, 15), sharey=True)
    run_counter = 0
    # Iterar sobre cada par de columnas (0-1, 2-3, ..., 58-59)
    for i in range(5):
        for j in range(6):
            # Índices de las columnas a graficar
            col1 = (run_counter * 2)
            col2 = col1 + 1
            #
            # Graficar los valores de las columnas col1 y col2 vs range(200)
            _=axs[i, j].plot(range(200), dict_losses_m1[str(cap_names_cv[k])].iloc[:, col1], label='train')
            _=axs[i, j].plot(range(200), dict_losses_m1[str(cap_names_cv[k])].iloc[:, col2], label='val')
            #
            # Añadir título y leyenda
            _=axs[i, j].set_title(f'Run {run_counter}')
            _=axs[i, j].legend()
            #
            # Incrementar el contador de "runs"
            run_counter += 1
    
    _=fig.suptitle(f"val {str(cap_names_cv[k])} - Training and validation loss for Model 1\n", fontsize=18)
    _=fig.text(0.5, 0.04, 'Epochs', ha='center', va='center', fontsize=14)
    _=fig.text(0.08, 0.5, 'Loss', ha='center', va='center', rotation='vertical', fontsize=14)
    # Ajustar el layout para evitar solapamientos
    plt.tight_layout()
    plt.savefig(os.path.join(losses_dir_m1, f"losses_m1_{str(cap_names_cv[k])}.pdf"))
    plt.savefig(os.path.join(losses_dir_m1, f"losses_m1_{str(cap_names_cv[k])}.png"))
    
    plt.close(fig)
    
    ################# LOSSES Model 2 ##################################
    
    # Configurar la grilla de 30 plots (5 filas x 6 columnas)
    fig, axs = plt.subplots(5, 6, figsize=(20, 15), sharey=True)
    run_counter = 0
    # Iterar sobre cada par de columnas (0-1, 2-3, ..., 58-59)
    for i in range(5):
        for j in range(6):
            # Índices de las columnas a graficar
            col1 = (run_counter * 2)
            col2 = col1 + 1
            #
            # Graficar los valores de las columnas col1 y col2 vs range(200)
            _=axs[i, j].plot(range(200), dict_losses_m2[str(cap_names_cv[k])].iloc[:, col1], label='train')
            _=axs[i, j].plot(range(200), dict_losses_m2[str(cap_names_cv[k])].iloc[:, col2], label='val')
            #
            # Añadir título y leyenda
            _=axs[i, j].set_title(f'Run {run_counter}')
            _=axs[i, j].legend()
            #
            # Incrementar el contador de "runs"
            run_counter += 1
    
    _=fig.suptitle(f"val {str(cap_names_cv[k])} - Training and validation loss for Model 2\n", fontsize=18)
    _=fig.text(0.5, 0.04, 'Epochs', ha='center', va='center', fontsize=14)
    _=fig.text(0.08, 0.5, 'Loss', ha='center', va='center', rotation='vertical', fontsize=14)
    # Ajustar el layout para evitar solapamientos
    plt.tight_layout()
    plt.savefig(os.path.join(losses_dir_m2, f"losses_m2_{str(cap_names_cv[k])}.pdf"))
    plt.savefig(os.path.join(losses_dir_m2, f"losses_m2_{str(cap_names_cv[k])}.png"))
    
    plt.close(fig)
    
    ################# METRICS Test 1 ##################################
    
    # Configurar la grilla de 30 plots (5 filas x 6 columnas)
    fig, axs = plt.subplots(5, 6, figsize=(20, 15), sharey=True)

    # Nombres para los xticks (las 4 comparaciones)
    xtick_labels = ['acc', 'prec', 'rec', 'f1']

    # Iterar sobre cada fila del DataFrame
    for idx, row in enumerate(dict_metrics_t1[str(cap_names_cv[k])].iterrows()):
        i, j = divmod(idx, 6)  # Calcular el índice para las subplots (filas y columnas)
        
        # Extraer los valores de cada fila para graficar
        group_1 = row[1].iloc[[1, 2, 3, 4]]  
        group_2 = row[1].iloc[[6, 7, 8, 9]]  
        
        # Crear el gráfico de barras comparando los grupos
        _=axs[i, j].bar(np.arange(4) - 0.2, group_1, width=0.4, label='Model 1')
        _=axs[i, j].bar(np.arange(4) + 0.2, group_2, width=0.4, label='Model 2')
        
        # Añadir título, xticks y leyenda
        _=axs[i, j].set_title(f'Run {idx}')
        _=axs[i, j].set_xticks(np.arange(4))  # Colocar los xticks en las posiciones correctas
        if i==4: 
            _=axs[i, j].set_xticklabels(xtick_labels, rotation=45, ha='right')  # Colocar los nombres de los xticks
        #_=axs[i, j].legend()
    
    _=axs[0,0].legend()
    
    _=fig.suptitle(f"val {str(cap_names_cv[k])} - Metrics for test1 (capture20110818)\n", fontsize=18)
    # Ajustar el layout para evitar solapamientos
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir_t1, f"metrics_t1_{str(cap_names_cv[k])}.pdf"))
    plt.savefig(os.path.join(metrics_dir_t1, f"metrics_t1_{str(cap_names_cv[k])}.png"))
    
    plt.close(fig)
    
    ################# METRICS Test 2 ##################################
    
    # Configurar la grilla de 30 plots (5 filas x 6 columnas)
    fig, axs = plt.subplots(5, 6, figsize=(20, 15), sharey=True)

    # Nombres para los xticks (las 4 comparaciones)
    xtick_labels = ['acc', 'prec', 'rec', 'f1']

    # Iterar sobre cada fila del DataFrame
    for idx, row in enumerate(dict_metrics_t2[str(cap_names_cv[k])].iterrows()):
        i, j = divmod(idx, 6)  # Calcular el índice para las subplots (filas y columnas)
        
        # Extraer los valores de cada fila para graficar
        group_1 = row[1].iloc[[1, 2, 3, 4]]  
        group_2 = row[1].iloc[[6, 7, 8, 9]]  
        
        # Crear el gráfico de barras comparando los grupos
        _=axs[i, j].bar(np.arange(4) - 0.2, group_1, width=0.4, label='m1')
        _=axs[i, j].bar(np.arange(4) + 0.2, group_2, width=0.4, label='m2')
        
        # Añadir título, xticks y leyenda
        _=axs[i, j].set_title(f'Run {idx}')
        _=axs[i, j].set_xticks(np.arange(4))  # Colocar los xticks en las posiciones correctas
        if i==4: 
            _=axs[i, j].set_xticklabels(xtick_labels, rotation=45, ha='right')  # Colocar los nombres de los xticks
        #_=axs[i, j].legend()
    
    _=axs[0,0].legend()
    
    _=fig.suptitle(f"val {str(cap_names_cv[k])} - Metrics for test2 (capture20110816-2)\n", fontsize=18)
    # Ajustar el layout para evitar solapamientos
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir_t2, f"metrics_t2_{str(cap_names_cv[k])}.pdf"))
    plt.savefig(os.path.join(metrics_dir_t2, f"metrics_t2_{str(cap_names_cv[k])}.png"))
    
    plt.close(fig)
    
    ################## Boxplot metrics test1 (capture 18) #################################
    
    data_model1 = dict_metrics_t1[str(cap_names_cv[k])].iloc[:, 1:5]
    data_model2 = dict_metrics_t1[str(cap_names_cv[k])].iloc[:, 6:10]
    
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
    _=ax.set_title(f'val {str(cap_names_cv[k])} - Distribution metrics test1 (capture20110818)')
    #_=ax.set_ylabel('Values')

    # Add a legend to indicate which boxplot represents which model
    legend_elements = [plt.Line2D([0], [0], color='tab:blue', lw=6, label='Model 1'),
                       plt.Line2D([0], [0], color='tab:orange', lw=6, label='Model 2')]
    _=ax.legend(handles=legend_elements, loc='upper right')

    # Show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir_t1, f"boxplot_metrics_test1_{str(cap_names_cv[k])}.pdf"))
    plt.savefig(os.path.join(metrics_dir_t1, f"boxplot_metrics_test1_{str(cap_names_cv[k])}.png"))
        
    plt.close(fig)

    ################## Boxplot metrics test2 (capture 16-2) #################################

    data_model1 = dict_metrics_t2[str(cap_names_cv[k])].iloc[:, 1:5]
    data_model2 = dict_metrics_t2[str(cap_names_cv[k])].iloc[:, 6:10]
    
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
    _=ax.set_title(f'val {str(cap_names_cv[k])} - Distribution metrics test2 (capture20110816-2)')
    #_=ax.set_ylabel('Values')

    # Add a legend to indicate which boxplot represents which model
    legend_elements = [plt.Line2D([0], [0], color='tab:blue', lw=6, label='Model 1'), plt.Line2D([0], [0], color='tab:orange', lw=6, label='Model 2')]
    _=ax.legend(handles=legend_elements, loc='upper right')

    # Show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir_t2, f"boxplot_metrics_test2_{str(cap_names_cv[k])}.pdf"))
    plt.savefig(os.path.join(metrics_dir_t2, f"boxplot_metrics_test2_{str(cap_names_cv[k])}.png"))
    
    plt.close(fig)



########################################## BOXPLOTS COMPARATIVOS #######################
    
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.flatten()  # Flatten the axes array for easy iteration

for k in range(len(cap_names_cv)):
    data_model1 = dict_metrics_t1[str(cap_names_cv[k])].iloc[:, 1:5]
    data_model2 = dict_metrics_t1[str(cap_names_cv[k])].iloc[:, 6:10]
    
    ax = axes[k]
    
    # Plot the boxplot for Model 1 with slightly shifted positions to avoid overlap
    boxplot1 = ax.boxplot(data_model1.values, positions=[1, 4, 7, 10], widths=0.6, patch_artist=True, medianprops={'color': 'black'})
        
    # Plot the boxplot for Model 2 with slightly shifted positions
    boxplot2 = ax.boxplot(data_model2.values, positions=[2, 5, 8, 11], widths=0.6, patch_artist=True, medianprops={'color': 'black'})
    
    # Fill colors for Model 1 and Model 2
    for box in boxplot1['boxes']:
        _=box.set(facecolor='tab:blue', edgecolor='black')
    
    for box in boxplot2['boxes']:
        _=box.set(facecolor='tab:orange', edgecolor='black')
    
    _=ax.set_xticks([0, 1.5, 4.5, 7.5, 10.5])
    
    if k >= len(cap_names_cv) - 5:
        _=ax.set_xticklabels(['', 'Acc', 'Prec', 'Rec', 'F1'])
    else:
        _=ax.set_xticklabels([])
    
    _=ax.set_title(f'val {str(cap_names_cv[k])}')
    
    if k==0:
        legend_elements = [plt.Line2D([0], [0], color='tab:blue', lw=6, label='Model 1'), plt.Line2D([0], [0], color='tab:orange', lw=6, label='Model 2')]
        _=ax.legend(handles=legend_elements, loc='upper right')

# Add a main title for the entire grid
_=fig.suptitle('Distribution Metrics - test1 (capture20110818)\n\n', fontsize=16)

# Adjust layout for better spacing and to make room for the legend
#plt.tight_layout()

plt.savefig(os.path.join(RESULTS_PATH, "comparative_runs/metrics_plots/boxplot_metrics_test1.pdf"))
plt.savefig(os.path.join(RESULTS_PATH, "comparative_runs/metrics_plots/boxplot_metrics_test1.png"))
    
plt.close(fig)


#########################

fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.flatten()  # Flatten the axes array for easy iteration

for k in range(len(cap_names_cv)):
    data_model1 = dict_metrics_t2[str(cap_names_cv[k])].iloc[:, 1:5]
    data_model2 = dict_metrics_t2[str(cap_names_cv[k])].iloc[:, 6:10]
    
    ax = axes[k]
    
    # Plot the boxplot for Model 1 with slightly shifted positions to avoid overlap
    boxplot1 = ax.boxplot(data_model1.values, positions=[1, 4, 7, 10], widths=0.6, patch_artist=True, medianprops={'color': 'black'})
        
    # Plot the boxplot for Model 2 with slightly shifted positions
    boxplot2 = ax.boxplot(data_model2.values, positions=[2, 5, 8, 11], widths=0.6, patch_artist=True, medianprops={'color': 'black'})
    
    # Fill colors for Model 1 and Model 2
    for box in boxplot1['boxes']:
        _=box.set(facecolor='tab:blue', edgecolor='black')
    
    for box in boxplot2['boxes']:
        _=box.set(facecolor='tab:orange', edgecolor='black')
    
    _=ax.set_xticks([0, 1.5, 4.5, 7.5, 10.5])
    if k >= len(cap_names_cv) - 5:
        _=ax.set_xticklabels(['', 'Acc', 'Prec', 'Rec', 'F1'])
    else:
        _=ax.set_xticklabels([])
    
    _=ax.set_title(f'val {str(cap_names_cv[k])}')
    
    if k==0:
        legend_elements = [plt.Line2D([0], [0], color='tab:blue', lw=6, label='Model 1'), plt.Line2D([0], [0], color='tab:orange', lw=6, label='Model 2')]
        _=ax.legend(handles=legend_elements, loc='upper right')

# Add a main title for the entire grid
_=fig.suptitle('Distribution Metrics - test2 (capture20110816-2)\n\n', fontsize=16)

# Adjust layout for better spacing and to make room for the legend
#plt.tight_layout() #rect=[0, 0, 1, 0.96])

plt.savefig(os.path.join(RESULTS_PATH, "comparative_runs/metrics_plots/boxplot_metrics_test2.pdf"))
plt.savefig(os.path.join(RESULTS_PATH, "comparative_runs/metrics_plots/boxplot_metrics_test2.png"))
    
plt.close(fig)


########################################

plt.close('all')

with open(os.path.join(RESULTS_PATH, "comparative_runs", "dict_losses_m1.pkl"), 'wb') as filename:
    pickle.dump(dict_losses_m1, filename)
        
with open(os.path.join(RESULTS_PATH, "comparative_runs", "dict_losses_m2.pkl"), 'wb') as filename:
    pickle.dump(dict_losses_m2, filename)
    
with open(os.path.join(RESULTS_PATH, "comparative_runs", "dict_metrics_t1.pkl"), 'wb') as filename:
    pickle.dump(dict_metrics_t1, filename)
        
with open(os.path.join(RESULTS_PATH, "comparative_runs", "dict_metrics_t2.pkl"), 'wb') as filename:
    pickle.dump(dict_metrics_t2, filename)
