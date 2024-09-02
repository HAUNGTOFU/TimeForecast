import torch
from torch import nn
class ResidualBlock(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,dropout=0.1):
        super(ResidualBlock,self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.drop=nn.Dropout(0.1)
        self.fc1=nn.Linear(input_dim,output_dim)
        self.fc2=nn.Linear(input_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,output_dim)
        self.relu=nn.ReLU()
        self.layer=nn.LayerNorm(output_dim)
    def forward(self,x):
        x1=self.fc2(x)
        x1=self.relu(x1)
        x1=self.fc3(x1)
        x1=self.drop(x1)
        x2=self.fc1(x)+x1
        x2=self.layer(x2)
        return x2
#%%
class Encoder(nn.Module):
    def __init__(self,back_dim,r,hidden_dim,r_hat,fore_dim,encoder_num):
        super(Encoder,self).__init__()
        self.back_dim=back_dim
        self.r=r
        self.r_hat=r_hat
        self.hidden_dim=hidden_dim
        self.fore_dim=fore_dim
        self.encoder_num=encoder_num
        self.feature_Pro=ResidualBlock(r,hidden_dim,r_hat)
        self.cat_encoder=ResidualBlock((back_dim+fore_dim)*r_hat+back_dim,hidden_dim,hidden_dim,dropout=0.1)
        self.other_encoder=nn.ModuleList()
        for i in range(encoder_num):
            self.other_encoder.append(ResidualBlock(hidden_dim,hidden_dim,hidden_dim))
    def forward(self,back_x,dynamic_x):
        dynamic_x=self.feature_Pro(dynamic_x)
        dynamic_future=dynamic_x[:,-self.fore_dim:,:]
        dynamic_flatten=torch.reshape(dynamic_x,(dynamic_x.shape[0],dynamic_x.shape[1]*dynamic_x.shape[2]))
        x=torch.concat((back_x.squeeze(-1),dynamic_flatten),dim=1)
        x1=self.cat_encoder(x)
        for i in range(self.encoder_num-1):
            x1=self.other_encoder[i](x1)
        return x1,dynamic_future
#%%
class Decoder(nn.Module):
    def __init__(self,decoder_num,p,hidden_dim,fore_dim,hidden_tem,r_hat):
        super(Decoder,self).__init__()
        self.decoder_num=decoder_num
        self.p=p
        self.hidden_dim=hidden_dim
        self.fore_dim=fore_dim
        self.r_hat=r_hat
        self.hidden_tem=self,hidden_tem
        self.decoder1=nn.ModuleList()
        for i in range(decoder_num):
            self.decoder1.append(ResidualBlock(hidden_dim,hidden_dim,hidden_dim))
        self.decoder_last=ResidualBlock(hidden_dim,hidden_dim,p*fore_dim)
        self.temporal=ResidualBlock(p+r_hat,hidden_tem,1)
    def forward(self,x1,dynamic_future):
        for i in range(self.decoder_num):
            x1=self.decoder1[i](x1)
        x2=self.decoder_last(x1)
        x3=torch.reshape(x2,(x2.shape[0],self.fore_dim,self.p))
        x4=torch.concat((x3,dynamic_future),dim=-1)
        x5=self.temporal(x4)
        return x5
#%%
class TiDEModel(nn.Module):
    def __init__(self,back_dim,r,hidden_dim,r_hat,fore_dim,encoder_num,decoder_num,p,hidden_tem):
        super(TiDEModel,self).__init__()
        self.back_dim=back_dim
        self.r=r
        self.r_hat=r_hat
        self.hidden_dim=hidden_dim
        self.fore_dim=fore_dim
        self.encoder_num=encoder_num
        self.decoder_num=decoder_num
        self.p=p
        self.hidden_tem=self,hidden_tem
        self.Encoder=Encoder(back_dim=back_dim,r=r,hidden_dim=hidden_dim,r_hat=r_hat,fore_dim=fore_dim,encoder_num=encoder_num)
        self.Decoder=Decoder(decoder_num=decoder_num,p=p,hidden_dim=hidden_dim,fore_dim=fore_dim,hidden_tem=hidden_tem,r_hat=r_hat)
        self.fc1=nn.Linear(back_dim,fore_dim)
        self.fc2=ResidualBlock(back_dim,hidden_dim,fore_dim)
    def forward(self,x,dynamic):
        batch_size=x.shape[0]
        x=torch.reshape(x,(x.shape[0]*x.shape[-1],x.shape[1]))
        dynamic=torch.reshape(dynamic,(dynamic.shape[0]*dynamic.shape[2],dynamic.shape[1],dynamic.shape[-1]))
        x1,dynamic1=self.Encoder(x,dynamic)
        x2=self.Decoder(x1,dynamic1)
        prediction=x2.squeeze(-1)+self.fc1(x)
        prediction=torch.reshape(prediction,(batch_size,prediction.shape[1],int(prediction.shape[0]/batch_size)))
        return prediction