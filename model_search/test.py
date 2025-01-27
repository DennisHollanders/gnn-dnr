import os
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb
import os
from dotenv import load_dotenv


# ----------------------------
# 1. Define LightningDataModule
# ----------------------------
class CoraDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.transform = T.NormalizeFeatures()

    def prepare_data(self):
        Planetoid(root='./data', name='Cora')

    def setup(self, stage=None):
        dataset = Planetoid(root='./data', name='Cora', transform=self.transform)
        self.data = dataset[0]

    def train_dataloader(self):
        return DataLoader([self.data], batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader([self.data], batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader([self.data], batch_size=self.batch_size)

# ----------------------------
# 2. Define LightningModule
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
# 3. Main Training Routine
# ----------------------------
if __name__ == '__main__':


    # Load API key directly
    load_dotenv('WANDB')
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    pl.seed_everything(42)

    # Initialize WandB logger
    wandb_logger = WandbLogger(
        project='gnn-cora',
        log_model='all',  # Log model checkpoints
        save_dir='./logs/',
        tags=['gcn', 'cora'],
        entity='your-wandb-team'  # Optional: if using team account
    )

    dm = CoraDataModule()
    model = GNNModel()

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    checkpoint = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        dirpath=wandb_logger.experiment.dir  # Save checkpoints to WandB directory
    )

    # Trainer configuration
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=100,
        logger=wandb_logger,
        callbacks=[early_stop, checkpoint],
        deterministic=True,
    )

    # Log model architecture
    wandb_logger.watch(model, log='all', log_freq=100)

    # Train and test
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)

    # Finish WandB run
    wandb.finish()