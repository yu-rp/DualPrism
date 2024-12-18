import torch
import torch.nn.functional as F
from torch.nn import Linear

from math import ceil

from torch_geometric.nn import GraphConv, DenseGraphConv, dense_mincut_pool,TopKPooling,SAGEConv,GCNConv,DenseSAGEConv,GINConv,dense_diff_pool, global_mean_pool, JumpingKnowledge
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj


class GIN(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=1, num_hidden=32):
        super(GIN, self).__init__()

        # if data.x is None:
        #   data.x = torch.ones([data.num_nodes, 1], dtype=torch.float)
        # dataset.data.edge_attr = None

        # num_features = dataset.num_features
        dim = num_hidden

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        # x = global_add_pool(x, batch)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

class GCN(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=1, num_hidden=32):
        super(GCN, self).__init__()
        
        self.conv1 = GCNConv(num_features, num_hidden)
        self.conv2 = GCNConv(num_hidden, num_hidden)
        self.conv3 = GCNConv(num_hidden, num_hidden)
        self.conv4 = GCNConv(num_hidden, num_hidden)
        
        # Output layer
        self.fc1 = Linear(num_hidden, num_hidden)
        self.fc2 = Linear(num_hidden, num_classes)

    def forward(self, x, edge_index, batch):
        
        # Apply the four GCN layers with ReLU activation
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        
        # Perform global mean pooling to aggregate node features
        x = global_mean_pool(x, batch)
        
        # Apply the output layer for graph-level classification
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=-1)

class GCN_with_BN(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=1, num_hidden=32):
        super(GCN_with_BN, self).__init__()
        
        self.conv1 = GCNConv(num_features, num_hidden)
        self.conv2 = GCNConv(num_hidden, num_hidden)
        self.conv3 = GCNConv(num_hidden, num_hidden)
        self.conv4 = GCNConv(num_hidden, num_hidden)

        # BN layer
        self.bn1 = torch.nn.BatchNorm1d(num_hidden)
        self.bn2 = torch.nn.BatchNorm1d(num_hidden)
        self.bn3 = torch.nn.BatchNorm1d(num_hidden)
        self.bn4 = torch.nn.BatchNorm1d(num_hidden)
        
        # Output layer
        self.fc1 = Linear(num_hidden, num_hidden)
        self.fc2 = Linear(num_hidden, num_classes)

    def forward(self, x, edge_index, batch):
        
        # Apply the four GCN layers with ReLU activation
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        
        # Perform global mean pooling to aggregate node features
        x = global_mean_pool(x, batch)
        
        # Apply the output layer for graph-level classification
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=-1)