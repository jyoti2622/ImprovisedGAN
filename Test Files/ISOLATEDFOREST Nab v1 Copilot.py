import torch
import pandas as pd
import numpy as np
import time
import datetime
from pathlib import Path
from src.algorithms.IsolatedForest import IsolatedForestAlgo
from src.utils.util import *
from src.dataset.Nabdataset import NabDataset
from src.model.modelIsolatedForest import IsolatedForest
from src.utils.timeseries_anomalies import _fixed_threshold,_find_threshold
from src.utils.metrics import *
import seaborn as sns
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

import torch.nn as nn 
import torch.backends.cudnn as cudnn


#This is the beginning of programm
t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)

class ArgsTrn:
    workers=4
    batch_size=32
    epochs=10
    lr=0.0002
    cuda = True
    manualSeed=2
    mean=0
    std=0.1
    
opt_trn=ArgsTrn()
    
opt_trn=ArgsTrn()
torch.manual_seed(opt_trn.manualSeed)



class Datasettings:
    
    def __init__(self):
        end_name='Twitter_volume_AAPL.csv'
        self.data_folder_path = Path.cwd().joinpath("data", "nab")
        key='realTweets/'+end_name 
        self.label_file = './lables/combined_windows.json'
        self.key=key
        self.train=True
        self.window_length=60

data_settings = Datasettings()
dataset = NabDataset(data_settings=data_settings)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt_trn.batch_size,
                                         shuffle=True, num_workers=int(opt_trn.workers))
seq_len = dataset.window_length # sequence length is equal to the window length
in_dim = dataset.n_feature # input dimension is same as number of feature
n_features=dataset.x.shape[2]
sequences=[x for i, (x,y) in enumerate(dataloader, 0)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset.x.shape
embedding_dim=16

isolatedForestAlgo=IsolatedForestAlgo(device=device,opt_trn=opt_trn,windows_length=seq_len,n_features=n_features,embedding_dim=embedding_dim)

trained_isolated_forest=isolatedForestAlgo.train_autoencoder(sequences)

PATH = Path.cwd().joinpath("src","saved_models","IsolatedForest","nab", "isolated_forest.pkl")

torch.save(trained_isolated_forest.module.state_dict(), PATH)

state_dict = torch.load(PATH)
trained_isolated_forest=IsolatedForest(embedding_dim, n_features,device=device)
trained_isolated_forest=nn.DataParallel(trained_isolated_forest)
trained_isolated_forest.to(device)
trained_isolated_forest=isolatedForestAlgo.load_model(state_dict,trained_isolated_forest)


isolatedForestAlgo.intialize_lstmautoencoder(trained_isolated_forest)

#Test Data
class ArgsTest:
    workers = 1
    batch_size = 1
    
opt_test=ArgsTest()


class TestDataSettings:
    
    def __init__(self):
        end_name='Twitter_volume_AAPL.csv'
        self.data_folder_path=Path.cwd().joinpath("data", "nab")
        key='realTweets/'+end_name  
        self.label_file = './lables/combined_windows.json'
        self.key=key
        self.train=False
        self.window_length=60        
        
test_data_settings = TestDataSettings()

# define dataset object and data loader object in evaluation mood for NAB dataset
test_dataset = NabDataset(test_data_settings)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt_test.batch_size, 
                                         shuffle=False, num_workers=int(opt_test.workers))

test_dataset.x.shape, test_dataset.y.shape, test_dataset.data_len # check the dataset shape


test_sequences=[x for i, (x,y) in enumerate(test_dataloader, 0)]

losses=isolatedForestAlgo.predict_loss(test_sequences)

import matplotlib.pyplot as plt

plt.figure(figsize=(16,9), dpi=80)
plt.title('Loss Distribution', fontsize=16)
sns.histplot(losses, bins = 20, kde= False, color = 'blue');
#sns.distplot(losses, bins = 20, kde= True, color = 'blue');


THRESHOLD =1.7

test_score_df = pd.DataFrame(index=range(len(losses)))
test_score_df['loss'] = [loss for loss in losses]
test_score_df['y'] = test_dataset.y
test_score_df['threshold'] = THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
test_score_df['t'] = [x[59].item() for x in test_dataset.x]

plt.plot( test_score_df.loss, label='loss')
plt.plot( test_score_df.threshold, label='threshold')
#plt.plot( test_score_df.y, label='y')
plt.xticks(rotation=25)
plt.legend();

actual,predicted=improve_detection(test_score_df)



predicted = np.array(predicted)
actual = np.array(actual)

print_scores(predicted,actual)
