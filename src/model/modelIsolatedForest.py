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
from src.utils.losses import Wasserstein
import torch.nn.init as init
from src.utils.util import *
from collections import OrderedDict
from src.utils.tf_dtw import SoftDTW
from sklearn.ensemble import IsolationForest

class IsolatedForest(nn.Module):
    
    def __init__(self,n_estimators=100, contamination=0.1):
        super(IsolatedForest, self).__init__()
        self.model = IsolationForest(n_estimators=n_estimators, contamination=contamination)
        
    def forward(self, x):
        return self.model.decision_function(x)
    
    def predict(self, x):
        return self.model.predict(x)
    
    def fit(self, x):
        return self.model.fit(x)
    
