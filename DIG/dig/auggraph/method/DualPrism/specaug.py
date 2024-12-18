import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

from dig.auggraph.method.SpecAug.model.GCN import GCN
from dig.auggraph.method.SpecAug.model.GIN import GIN
from dig.auggraph.method.SpecAug.utils.utils import NormalizedDegree, triplet_loss
from dig.auggraph.dataset.aug_dataset import TripleSet

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj, dense_to_sparse, degree
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from torch.nn.functional import softmax

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score

from utils.augmentation import spectral_noise

class specaug():
    r"""
    The S-Mixup from the `"Graph Mixup with Soft Alignments" <https://icml.cc/virtual/2023/poster/24930>`_ paper.
    
    Args:
        data_root_path (string): Directory where datasets are saved. 
        dataset (string): Dataset Name.
        conf (dict): Hyperparameters of the graph matching network which is used to compute the soft alignments.
    """
    
    def __init__(self, data_root_path, dataset):
        self._get_dataset(data_root_path, dataset)
        
        
    def _get_dataset(self, data_root_path, dataset):
        if (dataset == 'IMDBB'):
            dataset = TUDataset(data_root_path, name='IMDB-BINARY', use_node_attr=True)
            num_cls = 2
        elif dataset == 'MUTAG':
            dataset = TUDataset(data_root_path, name="MUTAG", use_node_attr=True)
            num_cls = 2
        elif dataset == 'PROTEINS':
            dataset = TUDataset(data_root_path, name=dataset, use_node_attr=True)
            num_cls = 2
        elif dataset == 'REDDITB':
            dataset = TUDataset(data_root_path, name="REDDIT-BINARY", use_node_attr=True)
            num_cls = 2
        elif dataset == 'IMDBM':
            dataset = TUDataset(data_root_path, name='IMDB-MULTI', use_node_attr=True)
            num_cls = 3
        elif dataset == 'REDDITM5':
            dataset = TUDataset(data_root_path, name='REDDIT-MULTI-5K', use_node_attr=True)
            num_cls = 5
        elif dataset == 'REDDITM12':
            dataset = TUDataset(data_root_path, name='REDDIT-MULTI-12K', use_node_attr=True)
            num_cls = 11
        elif dataset == 'NCI1':
            dataset = TUDataset(data_root_path, name='NCI1', use_node_attr=True)
            num_cls = 2
            
        if dataset.data.x is None:
            max_degree = 0
            degs = []
            for data in dataset:
                degs += [degree(data.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())

            if max_degree < 1000:
                dataset.transform = T.OneHotDegree(max_degree)
            else:
                deg = torch.cat(degs, dim=0).to(torch.float)
                mean, std = deg.mean().item(), deg.std().item()
                dataset.transform = NormalizedDegree(mean, std)
                
        self.dataset = dataset
        self.num_cls = num_cls
        
    
    def train_test(
        self, batch_size, cls_model, cls_nlayers, cls_hidden, cls_dropout, cls_lr, cls_epochs, ckpt_path, 
        aug_ratio, std_dev,   aug_prob, aug_freq, aug_freq_ratio 
    ):
        r"""
        This method first train a GMNET and then use the GMNET to perform S-Mixup. 
        
        Args:
            batch_size (int): Batch size of training the classifier.
            cls_model (string): Use GCN or GIN as the backbone of the classifier. 
            cls_nlayers (int): Number of GNN layers of the classifier.
            cls_hidden (int): Number of hidden units of the classifier.
            cls_dropout (float): Dropout ratio of the classifier.
            cls_lr (float): Initial learning rate of training the classifier. 
            cls_epochs (int): Training epochs of the classifier.
            ckpt_path (string): Location for saving checkpoints. 
        """
        criterion = nn.CrossEntropyLoss()
        test_accs = []
        kf = KFold(n_splits=10, shuffle=True)
        for i, (train_idx, test_idx) in enumerate(kf.split(list(range(len(self.dataset))))):
            train_idx, val_idx = train_test_split(train_idx, test_size=0.1)
            train_set, val_set, test_set = self.dataset[train_idx.tolist()], self.dataset[val_idx.tolist()], self.dataset[test_idx.tolist()]
            
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = 8)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers = 8)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers = 8)

            if (cls_model == 'GCN'):
                model = GCN(self.dataset[0].x.shape[1], self.num_cls, cls_nlayers, cls_hidden, cls_dropout)
            elif (cls_model == 'GIN'):
                model = GIN(self.dataset[0].x.shape[1], self.num_cls, cls_nlayers, cls_hidden, cls_dropout)
            model.cuda()
            
            optimizer = optim.Adam(model.parameters(),
                                lr=cls_lr, weight_decay=1e-5)
            
            best_acc = 0.0         
            for epoch in range(1, cls_epochs + 1):
                loss_accum = 0.0
                for step, batch in enumerate(train_loader):
                    batch = self.spectral_noise(
                        batch, aug_ratio = aug_ratio,
                        std_dev = std_dev, aug_prob = aug_prob, aug_freq = aug_freq, aug_freq_ratio = aug_freq_ratio,
                        )

                    model.train()
                    optimizer.zero_grad()

                    _, output = model(batch)
                    
                    loss = batch.lam * criterion(output, batch.y1.long()) + (1-batch.lam) * criterion(output, batch.y2.long())
                    loss.backward()
                    optimizer.step()

                    loss_accum += loss.item()

                train_loss = loss_accum / (step + 1)
                print("Epoch [{}] Train_loss {}".format(epoch, train_loss))

                y_label = []
                y_pred = []
                
                for step, batch in enumerate(val_loader):
                    batch = batch.cuda()
                    model.eval()
                    _, output = model(batch)                   
                    pred = torch.argmax(output, dim = 1).long()
                    y_pred = y_pred + pred.cpu().detach().numpy().flatten().tolist()
                    y = batch.y.long()
                    y_label = y_label + y.cpu().detach().numpy().flatten().tolist()

                acc_val = accuracy_score(y_pred, y_label)
                print("Epoch [{}] Test results:".format(epoch),
                        "acc_val: {:.4f}".format(acc_val),)
                
                if acc_val >= best_acc:
                    best_acc = acc_val
                    torch.save(model.state_dict(), ckpt_path + "/best_val.pth")
            
            model.load_state_dict(torch.load(ckpt_path + "/best_val.pth"))
            
            y_label = []
            y_pred = []
            for step, batch in enumerate(test_loader):
                batch = batch.cuda()
                model.eval()
                _, output = model(batch)
                pred = torch.argmax(output, dim = 1).long()
                y_pred = y_pred + pred.cpu().detach().numpy().flatten().tolist()
                y = batch.y.long()
                y_label = y_label + y.cpu().detach().numpy().flatten().tolist()
                
            acc_test = accuracy_score(y_pred, y_label)
            
            print("Split {}: acc_test: {:.4f}".format(i, acc_test),)

            test_accs.append(acc_test)
            
        print("Final result: acc_test: {:.4f}+-{:.4f}".format(np.mean(test_accs), np.std(test_accs)))

    

