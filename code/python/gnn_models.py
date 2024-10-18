import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, EdgePooling

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# MODEL 1

class EGraphSAGE1(torch.nn.Module):
    def __init__(self, in_channels_node, in_channels_edge, hidden_channels, out_channels):
        super(EGraphSAGE1, self).__init__()

        # Node-wise GraphSAGE convolution layers
        self.conv1 = SAGEConv(in_channels_node, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        # Linear layer for combining node and edge features
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels + in_channels_edge, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        # Apply GraphSAGE convolutions to node features
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # Extract the features for source and destination nodes of each edge
        row, col = edge_index
        src_features = x[row]
        dst_features = x[col]

        # Concatenate source, destination node features, and edge features
        edge_features = torch.cat([src_features, dst_features, edge_attr], dim=1)

        # Pass through the MLP for edge classification
        edge_logits = self.edge_mlp(edge_features)

        return edge_logits.squeeze() # Agrego squeeze() para que la salida pase de [N,1] a [N] (asumiendo que out_channels=1)


###########################################################################


# MODEL 2

# Define the E-GraphSAGE Model
class EGraphSAGE2(torch.nn.Module):
    def __init__(self, in_channels, edge_in_channels, hidden_channels1, hidden_channels2, out_channels):
        super(EGraphSAGE2, self).__init__()
        self.sage1 = SAGEConv(in_channels, hidden_channels1)
        self.sage2 = SAGEConv(hidden_channels1, hidden_channels2)

        # Linear layer for edge classification
        self.edge_fc = torch.nn.Linear(hidden_channels2*2 + edge_in_channels, out_channels)  # Concatenate node embeddings and edge features

    def forward(self, x, edge_index, edge_attr):
        # Step 1: Node embedding via GraphSAGE
        x = F.relu(self.sage1(x, edge_index))
        x = self.sage2(x, edge_index)

        # Step 2: Edge classification - Concatenate the embeddings of the source and target nodes
        edge_src = x[edge_index[0]]  # Embeddings for source nodes
        edge_dst = x[edge_index[1]]  # Embeddings for destination nodes

        # Concatenate source node, destination node embeddings, and edge features
        edge_features = torch.cat([edge_src, edge_dst, edge_attr], dim=1)

        # Step 3: Pass concatenated features through fully connected layer to predict edge class
        return self.edge_fc(edge_features).squeeze() # Agrego squeeze() para que la salida pase de [N,1] a [N] (asumiendo que out_channels=1)



#############################################################################

criterion = torch.nn.BCEWithLogitsLoss()
# NOTA: BCEWithLogitsLoss requiere que la salida del modelo sea de tamaño [N] o a lo sumo [N,1] y aplicar squeeze()
# Ademas para calcular la loss los valores del target deben ser float
# Ademas para calcular las predicciones hay que calcular las probabilidades primero (con crossEntropy haciamos out.argmax(dim=1))


def train(model, optimizer, loader):
    model.train()
    total_loss = 0

    for data in loader:
        optimizer.zero_grad()

        # Forward pass
        out = model(data.x, data.edge_index, data.edge_attr)

        # Compute loss (compare predictions with edge labels)
        loss = criterion(out, data.edge_label.float()) # Convierto a float al usar BCEWithLogitsLoss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


##########################################################


def test(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0

    with torch.no_grad():
        for data in loader:
            # Forward pass
            out = model(data.x, data.edge_index, data.edge_attr)

            loss = criterion(out, data.edge_label.float()) # Convierto a float al usar BCEWithLogitsLoss
            test_loss += loss.item()

            # Get predicted class (as argmax over output). Only if use crossEntropy, as the output is [N,2]
            # pred = out.argmax(dim=1)

            # Get predicted class. Only if use BCEWithLogitsLoss, as the output is [N]
            # Convert logits to probabilities
            probabilities = torch.sigmoid(out)  # Apply sigmoid to get probabilities
            # Convert probabilities to binary predictions using a threshold
            pred = (probabilities.view(-1) > 0.5).long()  # Thresholding at 0.5

            # Store predictions and true labels
            all_preds.append(pred.cpu())
            all_labels.append(data.edge_label.cpu())

    # Concatenate all the predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Calculate average validation loss
    avg_test_loss = test_loss / len(loader)

    # Calculate accuracy, precision, recall, and f1-score
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)  # Use 'binary' for binary classification
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    return avg_test_loss, accuracy, precision, recall, f1, all_labels, all_preds



