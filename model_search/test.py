import os
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import wandb

# Load environment variables
load_dotenv('WANDB')

# ----------------------------
# 1. Data Module with Auto-Download
# ----------------------------
class CoraDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = T.Compose([
            T.NormalizeFeatures(),
            T.RandomNodeSplit(split='train_rest', num_val=500, num_test=500)
        ])

    def prepare_data(self):
        # Auto-download if not exists
        Planetoid(root=self.data_dir, name='Cora', transform=self.transform)

    def setup(self, stage=None):
        dataset = Planetoid(root=self.data_dir, name='Cora', transform=self.transform)
        self.data = dataset[0]

    def train_dataloader(self):
        return DataLoader([self.data], batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader([self.data], batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader([self.data], batch_size=self.batch_size)

# ----------------------------
# 2. Model Definition
# ----------------------------
class GNNModel(pl.LightningModule):
    def __init__(self, in_channels=1433, hidden_channels=64, out_channels=7, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.lr = lr

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = torch.dropout(x, p=0.5, train=self.training)
        return self.conv2(x, edge_index)

    def training_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index)
        loss = torch.nn.functional.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index)
        loss = torch.nn.functional.cross_entropy(out[batch.val_mask], batch.y[batch.val_mask])
        pred = out.argmax(dim=1)
        acc = (pred[batch.val_mask] == batch.y[batch.val_mask]).sum() / batch.val_mask.sum()
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index)
        pred = out.argmax(dim=1)
        acc = (pred[batch.test_mask] == batch.y[batch.test_mask]).sum() / batch.test_mask.sum()
        self.log('test_acc', acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# ----------------------------
# 3. Main Execution
# ----------------------------
if __name__ == '__main__':
    pl.seed_everything(42)
    
    # Initialize WandB
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    wandb_logger = WandbLogger(project='gnn-cora', log_model=True)

    # Ensure data directory exists
    os.makedirs('./data', exist_ok=True)

    # Initialize components
    dm = CoraDataModule()
    model = GNNModel()
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    checkpoint = ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=1)

    # Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=100,
        logger=wandb_logger,
        callbacks=[early_stop, checkpoint],
        deterministic=True
    )

    # Training
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
    wandb.finish()