import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
class Module(L.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze()
        z = self.model_name(x)
        loss = F.mse_loss(z.squeeze(), y.float())
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze()
        z = self.model_name(x)
        loss = F.mse_loss(z.squeeze(), y.float())
        self.log('test_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze()
        z = self.model_name(x)
        loss = F.mse_loss(z.squeeze(), y.float())
        self.log('val_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y = y.squeeze()
        z = self.model_name(x)
        loss = F.mse_loss(z.squeeze(), y.float())
        return z

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
def train(model,epochs, accelerator,train_data,valid_data,callbacks:None):
    model = model
    trainer_lstm = L.Trainer(max_epochs=epochs, accelerator=accelerator,callbacks=callbacks)
    trainer_lstm.fit(model=model, train_dataloaders=train_data, val_dataloaders=valid_data)
    return trainer_lstm