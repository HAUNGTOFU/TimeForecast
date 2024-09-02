import pandas as pd
import torch
from forecast.Scale.scale_data import Scale
from forecast.dataload.sequence import create_sequences_pre
from torch.utils.data import  TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from forecast.Scale.scale_data import inverse_min_max_scaler
class test_model:
    def __init__(self,model,model_train_pl,path,seq_len,pre_len,start):
        self.path = path
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.start = start
        self.model = model
        self.model_train_pl = model_train_pl
    def test_load(self):
        self.df = pd.read_csv(self.path)
        self.data = self.df['number'].to_numpy()
        self.min = self.df['number'].min()
        self.max = self.df['number'].max()
        self.scale = Scale(self.data, feature_range=(0, 1))
        data = self.scale.min_max_scaler()
        X, y = create_sequences_pre(data=data[self.start:], seq_length=self.seq_len, pred_length=self.pre_len)
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X, y)
        dataloder = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        return dataloder
    def predict(self):
        model = self.model
        model_train_pl = self.model_train_pl
        model.eval()
        prediction=model_train_pl.predict(model,self.test_load())
        pred_cat = torch.cat(prediction, dim=0)
        pred_re = pred_cat.reshape(pred_cat.size(0) * 2, )
        return pred_re
    def plot(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data[self.start+self.seq_len:], color="black", label='actual')
        plt.plot(inverse_min_max_scaler(self.predict().numpy(),self.min,self.max), color="red", label='predictions')
        plt.title("LSTM")
        plt.legend()
        plt.show()






