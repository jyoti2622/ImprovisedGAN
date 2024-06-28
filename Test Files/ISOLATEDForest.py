import torch
import torch.nn as nn 
import pandas as pd
import numpy as np
import time
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import datetime

from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataset.Nabdataset import NabDataset # Assuming this can load data as required
from src.utils.util import *

#This is the beginning of programm
t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)

class ArgsTrn:
    workers=4
    batch_size=32
    epochs=5
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
        '''
        end_name='ambient_temperature_system_failure.csv'
        self.dataset_name="NAB_Ambient_Temperature"
        key='realKnownCause/'+end_name 
        self.label_file = './lables/combined_windows.json'
        '''
        '''
        end_name='exchange-2_cpc_results.csv'
        self.dataset_name="exchange-2_cpc_results"
        key='realAdExchange/'+end_name 
        self.label_file = './lables/combined_windows.json'
        '''
        end_name='Twitter_volume_AAPL.csv'
        self.data_folder_path="/home/jupyter/GRANOGAN-IISC/data/nab/"
        key='realTweets/'+end_name 
        self.label_file = './lables/combined_windows.json'
        self.key=key
        self.train=True
        self.window_length=60

# Load and prepare data
data_settings = Datasettings()
dataset = NabDataset(data_settings=data_settings)
X_train = [x.numpy() for x, _ in dataset] # Convert to numpy array if not already
X_train = np.array(X_train).reshape(len(X_train), -1) # Reshape for Isolation Forest

test_data_settings = TestDataSettings()
test_dataset = NabDataset(test_data_settings)
X_test = [x.numpy() for x, _ in test_dataset]
X_test = np.array(X_test).reshape(len(X_test), -1) # Reshape for Isolation Forest
y_test = np.array([y for _, y in test_dataset])

# Train Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
iso_forest.fit(X_train)

# Predict anomalies
scores_pred = iso_forest.decision_function(X_test)
y_pred = iso_forest.predict(X_test)
y_pred = [1 if x == -1 else 0 for x in y_pred] # Convert -1 for outliers and 1 for inliers to 0 and 1

# Evaluation
print_scores(y_test, y_pred)

# Visualization
plt.figure(figsize=(16,9))
plt.plot(scores_pred, label='Anomaly Score')
plt.hlines(-0.2, xmin=0, xmax=len(scores_pred)-1, colors='r', label='Threshold')
plt.legend()
plt.title('Isolation Forest Anomaly Scores')
plt.show()

# Further visualization and evaluation as needed
