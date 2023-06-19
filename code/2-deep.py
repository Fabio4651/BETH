import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.optimizers import AdamW
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve

# Hyper - Parameters and dataset size definition 
dataset_size = 12000
split_percent = int(dataset_size * 0.2)
defined_hidden_size = 256 #64, 128, 256 -> from paper
defined_learning_rate = 0.003 #0.003, 0.0003, 0.00003 -> from paper
defined_weight_decay = 0.1 #0, 0.01, 0.1 -> from paper
defined_consecute_observations = 50

# Load the datasets
columns_to_use = ["processId", "parentProcessId", "userId", "mountNamespace", "eventId", "argsNum", "returnValue", "sus"] #, "evil"
train_data = pd.read_csv('../labelled_training_data.csv', usecols=columns_to_use, nrows=dataset_size)
val_data = pd.read_csv('../labelled_validation_data.csv', usecols=columns_to_use, nrows=split_percent)
test_data = pd.read_csv('../labelled_testing_data.csv', usecols=columns_to_use, nrows=split_percent)

# Define a function for preprocessing
def preprocess(data):
    data["processId"] = data["processId"].map(lambda x: 1 if x in [0, 1, 2] else 0)
    data["parentProcessId"] = data["parentProcessId"].map(lambda x: 1 if x in [0, 1, 2] else 0)
    data["userId"] = data["userId"].map(lambda x: 0 if x < 1000 else 1)
    data["mountNamespace"] = data["mountNamespace"].map(lambda x: 0 if x == 4026531840 else 1)
    data["eventId"] = data["eventId"]
    data['returnValue'] = data['returnValue'].map(lambda x: -1 if x < 0 else (0 if x == 0 else 1))

    # Apply label encoding for specific columns that are not numerical
    le = LabelEncoder()
    columns_to_encode = ['sus', 'eventId', 'argsNum'] #, 'evil'
    for column in columns_to_encode:
        data[column] = le.fit_transform(data[column])

    # Normalize 'eventId' and 'argsNum'
    #scaler = MinMaxScaler()
    #data[['eventId', 'argsNum']] = scaler.fit_transform(data[['eventId', 'argsNum']])
    
    return data

# Function to convert DataFrames to sequences
def df_to_sequences(data):
    sequences = [data[i-defined_consecute_observations:i].values for i in range(defined_consecute_observations, len(data))]
    X = pad_sequences(sequences)
    y = data['sus'][defined_consecute_observations:]
    return X, y

# Preprocess the datasets
train_data = preprocess(train_data)
val_data = preprocess(val_data)
test_data = preprocess(test_data)

# Convert DataFrames to sequences
X_train, y_train = df_to_sequences(train_data)
X_val, y_val = df_to_sequences(val_data)
X_test, y_test = df_to_sequences(test_data)


# Define the GRU model
model_gru = Sequential()
model_gru.add(GRU(defined_hidden_size, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=False))
model_gru.add(Dense(defined_hidden_size, activation='relu'))
model_gru.add(Dense(1, activation='sigmoid'))

# Compile the GRU model
optimizer = AdamW(learning_rate=defined_learning_rate, weight_decay=defined_weight_decay)
model_gru.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the GRU model
history_gru = model_gru.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Predict probabilities for the testing data
y_pred_proba = model_gru.predict(X_test).flatten()

# Calculate AUROC
gru_test_auc = roc_auc_score(y_test, y_pred_proba)
print(f"GRU Test AUROC: {gru_test_auc}")

# Calculate ROC curve points
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Plot the ROC curve
plt.figure(figsize=(10,6))
plt.plot(fpr, tpr, label=f'GRU Test (AUROC = {gru_test_auc:.3f})')
plt.plot([0,1], [0,1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

'''
# Plot GRU accuracy
plt.figure(figsize=(12,6))
plt.plot(history_gru.history['accuracy'])
plt.plot(history_gru.history['val_accuracy'])
plt.title('GRU Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

# Evaluate GRU model
gru_results = model_gru.evaluate(X_test, y_test)
print(f"GRU Test Loss: {gru_results[0]}, GRU Test Accuracy: {gru_results[1]}")
'''