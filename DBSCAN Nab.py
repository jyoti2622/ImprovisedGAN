import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from pathlib import Path

# Load the dataset
file_path = Path.cwd().joinpath("data","nab", "Twitter_volume_AAPL.csv")
data = pd.read_csv(file_path)

# Preview the data
print(data.head())

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Feature extraction
# Using only the 'value' column for DBScan, more features can be added if available
X = data[['value']].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBScan algorithm
db = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)

# Add the cluster labels to the data
data['cluster'] = db.labels_

# Identifying anomalies (noise points are labeled as -1 by DBScan)
data['anomaly'] = data['cluster'] == -1

# Assume anomalies based on a threshold value
threshold = data['value'].quantile(0.95)
data['assumed_anomaly'] = data['value'] > threshold

# Define ground truth and predicted labels
y_true = data['assumed_anomaly'].astype(int)
y_pred = data['anomaly'].astype(int)

# Calculate performance metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
fmeasure = f1_score(y_true, y_pred)
cohen_kappa = cohen_kappa_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc_val = auc(fpr, tpr)

# Print performance metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('f-measure:', fmeasure)
print('cohen_kappa_score:', cohen_kappa)
print('auc:', roc_auc)
print('roc_auc:', roc_auc_val)

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(data['timestamp'], data['value'], label='Value')
plt.scatter(data[data['anomaly']]['timestamp'], data[data['anomaly']]['value'], color='red', label='Anomaly')
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.title('Anomaly Detection using DBScan')
plt.legend()
plt.show()

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_val)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
