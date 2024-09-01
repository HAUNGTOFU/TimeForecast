import pandas as pd
import torch
from torch.utils.data import  DataLoader, TensorDataset, Subset
import lightning as L
from forecast.Scale.scale_data import Scale
from forecast.dataload.sequence import create_sequences_train
class CSVDataModule1(L.LightningDataModule):
    def __init__(self, file_path='./data.csv', seq_length=10, pred_length=2,train_size=0.8,valid_size=0.1,test_size=0.1, batch_size=32,feature_range=(0, 1)):
        super().__init__()
        self.file_path = file_path
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.batch_size = batch_size
        self.feature_range = feature_range
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
    def prepare_data(self):
        pass
    def setup(self, stage=None):
        df = pd.read_csv(self.file_path)
        data = df['number']
        scale1 = Scale(data, feature_range=self.feature_range)
        data1=scale1.min_max_scaler()
        X, y = create_sequences_train(data1, self.seq_length, self.pred_length)
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32)
        full_dataset = TensorDataset(X, y)
        train_size = int(self.train_size * len(full_dataset))
        val_size = int(self.valid_size * len(full_dataset))
        test_size = len(full_dataset) - int(self.train_size * len(full_dataset)) - int(self.valid_size * len(full_dataset))
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, len(full_dataset)))
        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)
        self.test_dataset = Subset(full_dataset, test_indices)
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
def train_val_test(path='./data.csv',seq_length=10, pred_length=2, batch_size=32,feature_range=(0, 1)):
    data_module_dli = CSVDataModule1(file_path=path
                                     ,seq_length=seq_length
                                     ,pred_length=pred_length
                                     ,batch_size=batch_size
                                     ,feature_range=feature_range)
    data_module_dli.prepare_data()
    data_module_dli.setup()
    train_loader_dli = data_module_dli.train_dataloader()
    val_loader_dli = data_module_dli.val_dataloader()
    test_loader_dli = data_module_dli.test_dataloader()
    return train_loader_dli, val_loader_dli, test_loader_dli