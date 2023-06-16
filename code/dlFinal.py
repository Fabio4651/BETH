import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from tensorflow.keras.optimizers import AdamW
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Hyper - Parameters and dataset size definition 
dataset_size = 8000
split_percent = int(dataset_size * 0.2)
defined_hidden_size = 256 #64, 128, 256 -> from paper
defined_learning_rate = 0.00003 #0.003, 0.0003, 0.00003 -> from paper
defined_weight_decay = 0.1 #0, 0.01, 0.1 -> from paper

# Load the datasets
columns_to_use = ["processId", "parentProcessId", "userId", "mountNamespace", "eventId", "argsNum", "returnValue", "sus", "evil"]
train_data = pd.read_csv('../labelled_training_data.csv', usecols=columns_to_use) #, nrows=8000)
val_data = pd.read_csv('../labelled_validation_data.csv', usecols=columns_to_use) #, nrows=2000)
test_data = pd.read_csv('../labelled_testing_data.csv', usecols=columns_to_use) #, nrows=2000)

train_data = train_data.sample(frac=1, random_state=42)  # Shuffle the rows randomly
train_data = train_data.head(dataset_size)

val_data = val_data.sample(frac=1, random_state=42)  # Shuffle the rows randomly
val_data = val_data.head(split_percent)

test_data = test_data.sample(frac=1, random_state=42)  # Shuffle the rows randomly
test_data = test_data.head(split_percent)


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
    columns_to_encode = ['sus', 'evil', 'eventId', 'argsNum']
    for column in columns_to_encode:
        data[column] = le.fit_transform(data[column])

    # Normalize 'eventId' and 'argsNum'
    scaler = MinMaxScaler()
    data[['eventId', 'argsNum']] = scaler.fit_transform(data[['eventId', 'argsNum']])
    
    return data

# Function to convert DataFrames to sequences
def df_to_sequences(data):
    sequences = [data[i-50:i].values for i in range(50, len(data))]
    X = pad_sequences(sequences)
    y = data['sus'][50:]
    return X, y

# Preprocess the datasets
train_data = preprocess(train_data)
val_data = preprocess(val_data)
test_data = preprocess(test_data)

# Convert DataFrames to sequences
X_train, y_train = df_to_sequences(train_data)
X_val, y_val = df_to_sequences(val_data)
X_test, y_test = df_to_sequences(test_data)

# Define the LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(defined_hidden_size, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=False))
model_lstm.add(Dense(defined_hidden_size, activation='relu'))
model_lstm.add(Dense(1, activation='sigmoid'))

# Compile the LSTM model
optimizer = AdamW(learning_rate=defined_learning_rate, weight_decay=defined_weight_decay)
model_lstm.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the LSTM model
history_lstm = model_lstm.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))



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


# Plot LSTM accuracy
plt.figure(figsize=(12,6))
plt.plot(history_lstm.history['accuracy'])
plt.plot(history_lstm.history['val_accuracy'])
plt.title('LSTM Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

# Plot GRU accuracy
plt.figure(figsize=(12,6))
plt.plot(history_gru.history['accuracy'])
plt.plot(history_gru.history['val_accuracy'])
plt.title('GRU Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

# Evaluate LSTM model
lstm_results = model_lstm.evaluate(X_test, y_test)
print(f"LSTM Test Loss: {lstm_results[0]}, LSTM Test Accuracy: {lstm_results[1]}")

# Evaluate GRU model
gru_results = model_gru.evaluate(X_test, y_test)
print(f"GRU Test Loss: {gru_results[0]}, GRU Test Accuracy: {gru_results[1]}")
