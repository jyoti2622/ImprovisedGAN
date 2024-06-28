# Implement Isoltaed Forest Algorithm in Python in for NAB Dataset. The implementation pattern can be similar to LSTM AutoEncoder. Create belows files with implemntation IsolatedForest.py under algorithm folder,modelIsolatedForest.py under model folder and IsolatedForest Nab.ipynb under root path.
import torch
import torch.nn as nn 
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import datetime


from src.model.modelIsolatedForestv1 import IsolatedForestModel
from src.utils.losses import Wasserstein
import torch.nn.init as init
from src.utils.util import *
from collections import OrderedDict
from src.utils.tf_dtw import SoftDTW
from sklearn.ensemble import IsolationForest

class IsolatedForestAlgo:
    
    def __init__(self,device=None,opt_trn=None,windows_length=60,n_features=1,embedding_dim=16):
        self.embedding_dim=embedding_dim
        self.device=device
        self.lr=opt_trn.lr
        self.windows_length=windows_length
        self.in_dim=n_features
        self.epochs=opt_trn.epochs
        self.criterion = torch.nn.MSELoss()
        self.model=IsolationForest(n_estimators=100, contamination=0.1)
        
    def load_model(self, state_dict, model):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.'+k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k]=v
        model.load_state_dict(new_state_dict)
        return model
    
    def intialize_lstmautoencoder(self,autoencoder):
        self.autoencoder=autoencoder
        
    def predict_loss(self,sequences):
        losses = []
        for x in sequences:
            x=x.float().to(self.device)
            output=self.autoencoder.forward(x)
            err=self.criterion(x,output.to(self.device))
            losses.append(err.item())
        return losses
    
    def train_autoencoder(self,sequences):
        history = dict()
        history['loss'] = []
        history['val_loss'] = []
        history['epoch'] = []  

# Generated code is not full, implement a running code
        for epoch in range(0,self.epochs):
            train_losses = []
            for x in sequences:
                x=x.float().to(self.device)
                # reshape x to 2D array for IsolationForest
                x_2d = x.reshape(-1, self.in_dim)
                # fit the model
                self.model.fit(x_2d)
                # predict anomaly scores
                anomaly_scores = self.model.decision_function(x_2d)
                # calculate loss
                loss = np.mean(anomaly_scores)
                train_losses.append(loss)
            train_loss = np.mean(train_losses)
            history['loss'].append(train_loss)
            history['epoch'].append(epoch)
            print(f'Epoch {epoch}: train loss {train_loss}')
        return self.model