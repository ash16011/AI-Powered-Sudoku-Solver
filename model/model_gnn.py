
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv

class SudokuGNN(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, output_dim=9, num_layers=4, dropout=0.3):
        super(SudokuGNN, self).__init__()

        # Input Layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Graph Convolutional Layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(GraphConv(hidden_dim, hidden_dim))

        # Residual Connections
        self.residual = nn.Linear(hidden_dim, hidden_dim)  # Residual connection adjusted

        # Output Layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Layer Normalization to stabilize training
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Weight Initialization (Xavier)
        self._initialize_weights()

    def _initialize_weights(self):
        """Apply Xavier Initialization to all layers."""
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x, edge_index):
        """
        Forward pass of the GNN model.
        - x: Node features (batch_size * 81, 9)
        - edge_index: Graph structure
        """
        # Input Embedding
        x = self.input_layer(x)

        # Process node-wise for residual connection
        batch_size = x.size(0) // 81
        x = x.view(batch_size, 81, -1)  # Reshape to (batch_size, 81, hidden_dim)
        
        # Residual Connection (Node-wise)
        residual = self.residual(x)

        # Graph Convolutional Layers with Residual Connections
        for conv in self.conv_layers:
            x = x.view(-1, x.size(-1))  # Flatten for GCN (batch_size * 81, hidden_dim)
            x = conv(x, edge_index)

            # Layer Normalization and LeakyReLU Activation
            x = self.layer_norm(x)
            x = F.leaky_relu(x, negative_slope=0.01)
            x = self.dropout(x)

            # Reshape back for residual connection
            x = x.view(batch_size, 81, -1)
            x = x + residual  # Node-wise Residual Connection

        # Output Layer
        x = x.view(-1, x.size(-1))  # Flatten for output layer
        x = self.output_layer(x)
        x = x.view(batch_size, 9, 9, 9)  # Reshape to (batch_size, 9, 9, 9)
        return x
