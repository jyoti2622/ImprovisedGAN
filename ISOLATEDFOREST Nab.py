import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, roc_curve, auc
import numpy as np
from pathlib import Path


# Step 1: Load the dataset
file_path = Path.cwd().joinpath("data","nab", "Twitter_volume_AAPL.csv")
data = pd.read_csv(file_path)

# Step 2: Preprocess the data
# Convert the timestamp to a datetime object
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# Check for missing values
print(data.isnull().sum())

# Step 3: Implement the Isolation Forest algorithm
# Using 'value' column for anomaly detection
X = data[['value']]

# Initialize the Isolation Forest
iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X)

# Predict anomalies
data['anomaly'] = iso_forest.predict(X)
data['anomaly'] = data['anomaly'].map({1: 0, -1: 1})

# Step 4: Evaluate the model
# Assuming ground truth is available in 'true_label' column
if 'true_label' in data.columns:
    y_true = data['true_label']
    y_pred = data['anomaly']

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    fmeasure = f1_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_val = auc(fpr, tpr)
    roc_auc_val = roc_auc_score(y_true, y_pred)

    print('Accuracy\t', accuracy)
    print('Precision\t', precision)
    print('Recall\t', recall)
    print('f-measure\t', fmeasure)
    print('cohen_kappa_score\t', kappa)
    print('auc\t', auc_val)
    print('roc_auc\t', roc_auc_val)

# Step 4: Visualize the results
plt.figure(figsize=(14, 6))
plt.plot(data.index, data['value'], label='Value')
plt.scatter(data[data['anomaly'] == 1].index, data[data['anomaly'] == 1]['value'], color='red', label='Anomaly')
plt.title('Anomaly Detection in Twitter Volume Data (AAPL)')
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.legend()
plt.show()
