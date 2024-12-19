import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool as gep
import torch.nn.functional as F

class DeepCPP(nn.Module):
    def __init__(self, ):
        super(DeepCPP, self).__init__()
        self.feature_size=9 
        self.num_channels=30 
        self.sequence_length=20 
        self.n_output=1 
        self.hidden = 4
        self.heads = 4

        # GATConv with more heads
        self.gat_conv1 = GATConv(self.feature_size, self.hidden * 8, heads=self.heads, dropout=0.3)
        
        # Add a residual connection
        self.gcn_conv2 = GCNConv(self.hidden * 8 * self.heads, 64)
        
        # Additional GCN layers for deeper representations
        self.gcn_conv3 = GCNConv(64, 128)

        # Sequence branch
        # First convolutional layer without max pooling
        self.conv1d_1 = nn.Conv1d(in_channels=self.num_channels, out_channels=64, kernel_size=3, stride=1)
        # Second convolutional layer without max pooling
        self.conv1d_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        transformed_seq_length = self.calculate_transformed_sequence_length(self.sequence_length)      
        
        # Flattened size 
        self.flattened_size = transformed_seq_length * 64  # Directly from CNN output
        self.fc1_seq = nn.Linear(self.flattened_size, 64)  # Reduced size
        # Batch Normalization layers applied after conv1d
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.relu = nn.LeakyReLU()


        
        # Fusion and Classifier
        self.fusion_fc = nn.Linear(64+128, 128)
        self.classifier_fc1 = nn.Linear(128, 64)
        # self.classifier_fc2 = nn.Linear(64, 32)
        self.classifier_fc3 = nn.Linear(64, self.n_output)

        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.LeakyReLU()
        self.sigmoid = torch.sigmoid

    def forward(self, graph_data, seq_data):

        x, edge_index, batch = graph_data.x.float(), graph_data.edge_index, graph_data.batch
       
        # Graph branch
        # Apply GATConv
        x = self.gat_conv1(x, edge_index)
        x = self.relu(x)
        
        # Apply GCN layers with residual connections
        x = self.gcn_conv2(x, edge_index)
        x = self.relu(x)

        # Apply a second GCN layer for deeper representation
        x = self.gcn_conv3(x, edge_index)
        x = self.relu(x)
        
        x = gep(x, batch)  # Global pooling

        # Sequence branch
        # Apply first conv layer and batch normalization
        seq = self.conv1d_1(seq_data)
        seq = self.relu(self.bn1(seq))
        
        # Apply second conv layer and batch normalization
        seq = self.conv1d_2(seq)
        seq = self.relu(self.bn2(seq))

        # Apply dropout
        seq = self.dropout(seq)
        
        # Flattened output 
        seq = seq.reshape(seq.size(0), -1)
        seq = self.fc1_seq(seq)

        # Fusion
        combined = torch.cat((x, seq), dim=1)
        
        combined = self.dropout(self.fusion_fc(combined))
        combined = self.relu(combined)

        # Classifier
        combined = self.classifier_fc1(combined)
        combined = self.relu(combined)
        combined = self.dropout(combined)
        # combined = self.classifier_fc2(combined)
        # combined = self.relu(combined)
        # combined = self.dropout(combined)
        combined = self.classifier_fc3(combined)

        return combined

    def calculate_transformed_sequence_length(self, sequence_length):
        # Calculate length without max-pooling
        return (sequence_length - 2 * (3 - 1))  # Adjusted for convolution kernel size