import torch
import pandas as pd
import numpy as np
import time
import datetime
from pathlib import Path
from sklearn.ensemble import IsolationForest
from src.utils.util import *
from src.dataset.Nabdataset import NabDataset
from src.utils.timeseries_anomalies import _fixed_threshold, _find_threshold
from src.utils.metrics import *
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

# This is the beginning of the program
t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)

class ArgsTrn:
    workers = 4
    batch_size = 32
    epochs = 10
    lr = 0.0002
    cuda = True
    manualSeed = 2
    mean = 0
    std = 0.1
    
opt_trn = ArgsTrn()
torch.manual_seed(opt_trn.manualSeed)

class Datasettings:
    
    def __init__(self):
        end_name = 'Twitter_volume_AAPL.csv'
        self.data_folder_path = Path.cwd().joinpath("data", "nab")
        key = 'realTweets/' + end_name 
        self.label_file = './lables/combined_windows.json'
        self.key = key
        self.train = True
        self.window_length = 60

data_settings = Datasettings()
dataset = NabDataset(data_settings=data_settings)
seq_len = dataset.window_length # sequence length is equal to the window length
n_features = dataset.x.shape[2]
sequences = [x for i, (x, y) in enumerate(dataset)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset.x.shape
embedding_dim=16

# Flatten the sequences for training
sequences_flat = np.array([x.flatten() for x in sequences])

# Train the Isolation Forest model with flattened data
model = IsolationForest()
model.fit(sequences_flat)

# Test Data
class ArgsTest:
    workers = 1
    batch_size = 1
    
opt_test = ArgsTest()

class TestDataSettings:
    
    def __init__(self):
        end_name = 'Twitter_volume_AAPL.csv'
        self.data_folder_path = Path.cwd().joinpath("data", "nab")
        key = 'realTweets/' + end_name  
        self.label_file = './lables/combined_windows.json'
        self.key = key
        self.train = False
        self.window_length = 60        
        
test_data_settings = TestDataSettings()

# Define dataset object and data loader object in evaluation mode for NAB dataset
test_dataset = NabDataset(test_data_settings)
test_sequences = [x for i, (x, y) in enumerate(test_dataset)]

# Predict the anomaly scores using the Isolation Forest model
anomaly_scores = model.decision_function(test_sequences)

import matplotlib.pyplot as plt

plt.figure(figsize=(16, 9), dpi=80)
plt.title('Anomaly Score Distribution', fontsize=16)
sns.histplot(anomaly_scores, bins=20, kde=False, color='blue');


THRESHOLD = 0.0

test_score_df = pd.DataFrame(index=range(len(anomaly_scores)))
test_score_df['anomaly_score'] = [score for score in anomaly_scores]
test_score_df['anomaly'] = test_score_df.anomaly_score < THRESHOLD

plt.plot(test_score_df.anomaly_score, label='anomaly score')
plt.axhline(y=THRESHOLD, color='r', linestyle='--', label='threshold')
plt.xticks(rotation=25)
plt.legend();

actual, predicted = improve_detection(test_score_df)


predicted = np.array(predicted)
actual = np.array(actual)

print_scores(predicted, actual)
