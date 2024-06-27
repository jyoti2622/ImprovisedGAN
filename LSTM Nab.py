import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, roc_curve, auc
from pathlib import Path


# Load the data
data_path = Path.cwd().joinpath("data","nab", "Twitter_volume_AAPL.csv")
df = pd.read_csv(data_path)
print("Data loaded successfully")

# Parse the timestamp column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
print("Timestamp parsed successfully")

# Set the timestamp column as the index
df.set_index('timestamp', inplace=True)
print("Timestamp set as index")

# Normalize the data
scaler = MinMaxScaler()
df['value'] = scaler.fit_transform(df[['value']])
print("Data normalization completed")
print(df.head())

# Prepare the data for LSTM
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append((seq, label))
    return sequences

seq_length = 50
sequences = create_sequences(df['value'].values, seq_length)
print(f"Sequences created, total sequences: {len(sequences)}")

X = np.array([seq for seq, _ in sequences])
y = np.array([label for _, label in sequences])

# Reshape X to fit LSTM input requirements (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))
print(f"Input data reshaped to: {X.shape}")

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
print("LSTM model built")

model.compile(optimizer='adam', loss='mean_squared_error')
print("Model compiled")
model.summary()

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"Data split into training and testing sets: {X_train.shape}, {X_test.shape}")

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))
print("Model training completed")

# Make predictions
y_pred = model.predict(X_test)
print("Predictions made on test set")

# Calculate the loss
loss = np.mean(np.abs(y_pred - y_test.reshape(-1, 1)), axis=1)
print("Loss calculated for test set")

# Set a threshold for anomalies
threshold = np.percentile(loss, 95)
print(f"Anomaly threshold set at: {threshold}")

# Identify anomalies
anomalies = loss > threshold
print(f"Number of anomalies detected: {np.sum(anomalies)}")

# Generate binary labels for anomalies (1 for anomaly, 0 for normal)
y_true = np.zeros_like(loss)
y_true[anomalies] = 1
y_pred_binary = (loss > threshold).astype(int)
print("Binary labels for anomalies generated")

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred_binary)
precision = precision_score(y_true, y_pred_binary)
recall = recall_score(y_true, y_pred_binary)
fmeasure = f1_score(y_true, y_pred_binary)
cohen_kappa = cohen_kappa_score(y_true, y_pred_binary)
roc_auc_val = roc_auc_score(y_true, loss)

# Calculate AUC for the ROC curve
fpr, tpr, _ = roc_curve(y_true, loss)
roc_auc = auc(fpr, tpr)

print('Accuracy\t', accuracy)
print('Precision\t', precision)
print('Recall\t', recall)
print('f-measure\t', fmeasure)
print('cohen_kappa_score\t', cohen_kappa)
print('auc\t', roc_auc)
print('roc_auc\t', roc_auc_val)