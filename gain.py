# Packages
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
# from tqdm.notebook import tqdm_notebook as tqdm
import torch.nn.functional as F
import pandas as pd

# System Parameters
# 1. Mini batch size
mb_size = 256
# 2. Missing rate
p_miss = 0.2
# 3. Hint rate
p_hint = 0.9
# 4. Loss Hyperparameters
alpha = 10
# 5. Train Rate
train_rate = 0.8

    
# Hint Vector Generation
def sample_M(m, n, p):
    A = np.random.uniform(0., 1., size = [m, n])
    B = A > p
    C = 1.*B
    return C

def sample_Z(m, n):
    pass

def fill(ts):
    tsc = ts.copy()
    x_min = ts.min()
    x_max = ts.max()
    tsc = (tsc - x_min) / (x_max - x_min)
    ds = MyDataset(tsc)
    dataloader = torch.utils.data.DataLoader(dataset=ds,batch_size = 256, shuffle=True, drop_last=True)
    model = train(tsc, dataloader)
    complete(tsc, model)
    tsc = tsc * (x_max - x_min) + x_min
    return tsc

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.gap_idx = self.data.isna().any()
        self.idxs = data.index[data.isna().T.any()==False]
        # self.tsg = self.data.loc[:, self.gap_idx == True] # na ts
        # self.tsd = self.data.loc[:, self.gap_idx == False] # no na ts
        
    def __len__(self):
        return len(self.idxs)
        
    def __getitem__(self, index):
        idx = self.idxs[index]
        return self.data.loc[idx].to_numpy()
        # return self.tsd.loc[idx-pd.Timedelta(days=self.range_):idx+pd.Timedelta(days=self.range_),:].to_numpy().flatten(), self.tsg.loc[idx, :].to_numpy()

def train(tsc, dataloader):
    # make generator discriminator
    # train all
    
    ### make generator and discriminator
    generator = nn.Sequential()
    generator.add_module('l1', nn.Linear(18,9))
    generator.add_module('ac1', nn.ReLU())
    generator.add_module('l2', nn.Linear(9,9))
    generator.add_module('ac2', nn.ReLU())
    generator.add_module('l3', nn.Linear(9,9))
    generator.add_module('ac3', nn.Sigmoid())

    discriminator = nn.Sequential()
    discriminator.add_module('l1', nn.Linear(18,9))
    discriminator.add_module('ac1', nn.ReLU())
    discriminator.add_module('l2', nn.Linear(9,9))
    discriminator.add_module('ac2', nn.ReLU())
    discriminator.add_module('l3', nn.Linear(9,9))
    discriminator.add_module('ac3', nn.Sigmoid())

    ### train
    epochs = 5000
    optG = torch.optim.Adam(generator.parameters(), lr=0.001)
    optD = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    for epoch in tqdm(range(epochs)):
        for i, x in enumerate(dataloader):
            x = x.float() 
            m = (torch.rand(x.size()) > p_miss).float()
            z = torch.rand(256,9) * 0.01
            h_1 = (torch.rand(x.size()) < p_hint).float()
            h = m * h_1
            new_x = m * x + (1-m) * z
            
            ### d step
            optD.zero_grad()
            inputs = torch.cat(dim=1,tensors=[new_x, m])
            g_sample = generator(inputs)
            hat_new_x = new_x * m + g_sample * (1-m)
            inputs = torch.cat(dim=1,tensors=[hat_new_x, h])
            d_prob =  discriminator(inputs)
            d_loss = -torch.mean(m*torch.log(d_prob + 1e-8) + (1-m) * torch.log(1. - d_prob + 1e-8))
            d_loss.backward()

            ### g step
            optG.zero_grad()
            inputs = torch.cat(dim=1,tensors=[new_x, m])
            g_sample = generator(inputs)
            hat_new_x = new_x * m + g_sample * (1-m)
            inputs = torch.cat(dim=1,tensors=[hat_new_x, h])
            d_prob =  discriminator(inputs)
            g_loss1 = -torch.mean((1-m) * torch.log(d_prob + 1e-8))
            mse_train_loss = torch.mean((m * new_x - m * g_sample)**2) / torch.mean(m)
            g_loss = g_loss1 + mse_train_loss * alpha
            g_loss.backward()
        # if epoch % 10 == 0:
        #     print(f"d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")


    return generator


def complete(tsc, model):
    for i in range(len(tsc)):
        if tsc.iloc[i,:].isna().any() == False: 
            continue        
        x = tsc.iloc[i,:]
        gidx = tsc.iloc[i,:].isna()
        x[gidx] = 0
        x = torch.from_numpy(x.to_numpy()).float().unsqueeze(0)
        m = 1 - tsc.iloc[i,:].isna()
        m = torch.from_numpy(m.to_numpy()).float().unsqueeze(0)
        z = torch.rand(1,9) * 0.01
        new_x = m * x + (1-m) * z
        t = torch.cat(dim=1,tensors=[new_x, m])
        # x = torch.from_numpy(x.to_numpy().transpose()).float().unsqueeze(0).unsqueeze(0)
        model.eval()
        x_hat = model(t)
        out = x*m + (1-m) * x_hat
        tsc.iloc[i,:] = out.detach().numpy().squeeze(0)
        