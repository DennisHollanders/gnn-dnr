import wandb
import os 
import json
import yaml
import numpy as np  
import networkx as nx
import torchmetrics
import matplotlib.pyplot as plt
import pickle as pkl
import seaborn as sns
import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
import pytorch_lightning as pl
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.utils import to_networkx

from typing import Optional, Dict, Any, List, Union

wandb_logger = WandbLogger(log_model="all")
trainer = Trainer(logger=wandb_logger)

def load_graph_data(folder_path):
    graph_features_path = os.path.join(folder_path, "graph_features.pkl")
    nx_graphs_path = os.path.join(folder_path, "networkx_graphs.pkl")
    pp_networks_path = os.path.join(folder_path, "pandapower_networks.json")
    with open(graph_features_path, "rb") as f:
        graph_features = pkl.load(f)
    with open(nx_graphs_path, "rb") as f:
        nx_graphs = pkl.load(f)
    with open(pp_networks_path, "r") as f:
        pp_networks = json.load(f)
    return nx_graphs, graph_features, pp_networks

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    return torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

def unify_node_attributes(nx_graph, required_node_attrs):
    for node in nx_graph.nodes():
        for attr in list(nx_graph.nodes[node].keys()):
            if attr not in required_node_attrs:
                del nx_graph.nodes[node][attr]
        for attr in required_node_attrs:
            if attr not in nx_graph.nodes[node]:
                print(f"Missing attribute {attr} for node {node}. Adding it as 0.0.")
                nx_graph.nodes[node][attr] = 0.0

def unify_edge_attributes(nx_graph, required_edge_attrs):
    for u, v in nx_graph.edges():
        for attr in list(nx_graph[u][v].keys()):
            if attr not in required_edge_attrs:
                del nx_graph[u][v][attr]
        for attr in required_edge_attrs:
            if attr not in nx_graph[u][v]:
                print(f"Missing attribute {attr} for edge ({u}, {v}). Adding it as 0.0.")
                nx_graph[u][v][attr] = 0.0

def merge_features_into_nx(nx_graph, node_feats, edge_feats):
    if node_feats is None or edge_feats is None:
        raise ValueError("Missing node or edge features")
    for node, feats in node_feats.items():
        if node not in nx_graph.nodes():
            continue
        for feat_key, feat_value in feats.items():
            nx_graph.nodes[node][feat_key] = feat_value
    for (u, v), e_feats in edge_feats.items():
        if not nx_graph.has_edge(u, v):
            continue
        for feat_key, feat_value in e_feats.items():
            nx_graph[u][v][feat_key] = feat_value

def create_pyg_data(nx_graph):
    unify_node_attributes(nx_graph, ["p", "q", "v", "theta"])
    unify_edge_attributes(nx_graph, ["R", "X", "switch_state"])
    data = from_networkx(
        nx_graph,
        group_node_attrs=["p", "q", "v", "theta"],
        group_edge_attrs=["R", "X", "switch_state"]
    )
    if hasattr(data, "p"):
        data.x = torch.stack([data.p, data.q, data.v, data.theta], dim=-1).float()
        del data.p, data.q, data.v, data.theta
    if hasattr(data, "R"):
        data.edge_attr = torch.stack([data.R, data.X], dim=-1).float()
        del data.R, data.X
    if hasattr(data, "switch_state"):
        data.edge_y = data.switch_state.view(-1).float()
        del data.switch_state
    return data

def create_pyg_dataset(folder_path):
    nx_graphs, graph_features, _ = load_graph_data(folder_path)
    data_list = []
    skipped = 0
    total = len(nx_graphs)
    for graph_name, nx_graph in nx_graphs.items():
        if nx_graph is None:
            print(f" Skipping: Graph {graph_name} is None (non-converged).")
            skipped += 1
            continue
        if graph_name not in graph_features:
            print(f" Skipping: graph {graph_name} not found in graph features")
            skipped += 1
            continue
        features_dict = graph_features[graph_name]
        node_feats = features_dict.get("node_features")
        edge_feats = features_dict.get("edge_features")
        if node_feats is None or edge_feats is None:
            print(f"Skipping: raph {graph_name} has missing features")
            skipped += 1
            continue
        try:
            merge_features_into_nx(nx_graph, node_feats, edge_feats)
        except Exception as e:
            print(f"Skipping: error merging features for graph {graph_name}: {e}. ")
            skipped += 1
            continue
        try:
            data = create_pyg_data(nx_graph)
        except Exception as e:
            print(f"Skipping: error converting graph {graph_name} to pytorch geoemtric: {e}. ")
            skipped += 1
            continue
        data_list.append(data)
    print(f"total graphs: {total}; Skipped: {skipped}; Processed: {len(data_list)}")
    return data_list

def get_pyg_loader(folder_path, batch_size=1, shuffle=True):
    data_list = create_pyg_dataset(folder_path)
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)
    return loader

class GraphAutoencoder(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16],
                 latent_dim: int = 8, learning_rate: float = 1e-3, weight_decay: float = 1e-5, 
                 loss_fn: str = 'mse', optimizer: str = 'adam', scheduler: str = None,
                 activation: str = 'prelu', dropout_rate: float = 0.0):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.learning_rate = float(self.hparams.learning_rate)
        self.hparams.weight_decay = float(self.hparams.weight_decay)
        
        activation_map = {'relu': F.relu,
                          'leaky_relu': F.leaky_relu,
                          'elu': F.elu,
                          'selu': F.selu,
                          'prelu': F.prelu} # should add weight
        self.activation = activation_map.get(activation, F.relu)
        
        # Encoder layers 
        self.encoder_layers = torch.nn.ModuleList()
        layer_sizes = [input_dim] + hidden_dims
        for i in range(len(layer_sizes) - 1):
            self.encoder_layers.append(
                pyg_nn.GCNConv(layer_sizes[i], layer_sizes[i+1])
            )
        
        # Latent representation
        self.fc_mu = torch.nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = torch.nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder layersrs
        self.decoder_layers = torch.nn.ModuleList()
        reversed_dims = list(reversed(hidden_dims))
        layer_sizes = [latent_dim] + reversed_dims
        for i in range(len(layer_sizes) - 1):
            self.decoder_layers.append(
                torch.nn.Linear(layer_sizes[i], layer_sizes[i+1])
            )
        self.decoder_out = torch.nn.Linear(reversed_dims[-1], input_dim)
        
        # Metrics
        self.train_loss_metric = torchmetrics.MeanMetric()
        self.val_loss_metric = torchmetrics.MeanMetric()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.encoder_batchnorms = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(dim) for dim in hidden_dims
        ])
        reversed_dims = list(reversed(hidden_dims))
        self.decoder_batchnorms = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(dim) for dim in reversed_dims
        ])
        
    def encode(self, data):
        x, edge_index = data.x.float(), data.edge_index
        for layer, bn in zip(self.encoder_layers, self.encoder_batchnorms):
            x = layer(x, edge_index)
            x = self.activation(x)
            x = bn(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, data):
        for layer, bn in zip(self.decoder_layers, self.decoder_batchnorms):
            z = layer(z)
            z = self.activation(z)
            z = bn(z)
        return self.decoder_out(z)
        
    def training_step(self, batch, batch_idx):
        batch.x = batch.x.float()  
        mu, logvar = self.encode(batch)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, batch)
        if self.hparams.loss_fn == 'mse':
            recon_loss = F.mse_loss(x_hat, batch.x)
        elif self.hparams.loss_fn == 'mae':
            recon_loss = F.l1_loss(x_hat, batch.x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        self.log("train_loss", loss, prog_bar=True)
        self.train_loss_metric(loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch.x = batch.x.float()
        mu, logvar = self.encode(batch)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, batch)
        if self.hparams.loss_fn == 'mse':
            val_loss = F.mse_loss(x_hat, batch.x)
        elif self.hparams.loss_fn == 'mae':
            val_loss = F.l1_loss(x_hat, batch.x)
        self.log("val_loss", val_loss, prog_bar=True)
        self.val_loss_metric(val_loss)
        #if batch_idx == 0:
        #    self._log_reconstructions(batch.x, x_hat)
        return val_loss

    def test_step(self, batch, batch_idx):
        mu, logvar = self.encode(batch)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, batch)
        if self.hparams.loss_fn == 'mse':
            test_loss = F.mse_loss(x_hat, batch.x)
        elif self.hparams.loss_fn == 'mae':
            test_loss = F.l1_loss(x_hat, batch.x)
        self.log("test_loss", test_loss, prog_bar=True)
        return test_loss
    
    def _log_reconstructions(self, original, reconstructed):
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.title("Original")
        sns.heatmap(original.cpu().detach().numpy())
        plt.subplot(122)
        plt.title("Reconstructed")
        sns.heatmap(reconstructed.cpu().detach().numpy())
        #self.logger.experiment.log({"reconstructions": self.logger.experiment.Image(plt)})
        plt.close()
    
    def configure_optimizers(self):
        lr = float(self.hparams.learning_rate)
        wd = float(self.hparams.weight_decay)
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=wd)
        if self.hparams.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1}}
        return optimizer

def is_radial(data):
    # Check if the reconstructed graph is radial (i.e. a tree)
    G = to_networkx(data, to_undirected=True)
    return nx.is_tree(G)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_trainer(config):
    logger = WandbLogger(
        project=config.get("project", "graph-autoencoder"),
        save_dir="models/"
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=3, filename="{epoch}-{val_loss:.2f}"
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, verbose=True, mode="min"
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        max_epochs=config.get("max_epochs", 100),
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor]
    )
    return trainer, logger

if __name__ == "__main__":

    config =load_config(r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\notebooks\config_files\config.yaml")
    dataset = create_pyg_dataset(config["data_folder"])
    train_set, val_set, test_set = split_dataset(dataset, train_ratio=config.get("train_ratio", 0.7), val_ratio=config.get("val_ratio", 0.15),
    test_ratio=config.get("test_ratio", 0.15))
    train_loader = DataLoader(train_set, batch_size=config.get("batch_size", 1), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.get("batch_size", 1), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config.get("batch_size", 1), shuffle=False)

    model = GraphAutoencoder(
    input_dim=config["input_dim"],
    hidden_dims=config.get("hidden_dims", [64, 32, 16]),
    latent_dim=config.get("latent_dim", 8),
    learning_rate=config.get("learning_rate", 1e-3),
    weight_decay=config.get("weight_decay", 1e-5),
    loss_fn=config.get("loss_fn", "mse"),
    optimizer=config.get("optimizer", "adam"),
    scheduler=config.get("scheduler", None),
    activation=config.get("activation", "prelu"),
    dropout_rate=config.get("dropout_rate", 0.0)
    )
    # Setup trainer and logger
    trainer, logger = setup_trainer(config)
    logger.experiment.config.update(config)

    # Train and validate
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint("models/last.ckpt")