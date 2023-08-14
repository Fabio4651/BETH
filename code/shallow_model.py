import pandas as pd
#import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasRegressor
from scikeras.wrappers import KerasRegressor
import seaborn as sns

# Function to preprocess data
def preprocess_data(data):
    columns_to_drop = ['processName', 'hostName', 'eventName', 'stackAddresses', 'args']
    data = data.drop(columns=columns_to_drop)
    data['processId'] = data['processId'].apply(lambda x: 1 if x in [0, 1, 2] else 0)
    data['parentProcessId'] = data['parentProcessId'].apply(lambda x: 1 if x in [0, 1, 2] else 0)
    data['userId'] = data['userId'].apply(lambda x: 1 if x < 1000 else 0)
    data['mountNamespace'] = data['mountNamespace'].apply(lambda x: 1 if x == 4026531840 else 0)
    data['returnValue'] = data['returnValue'].apply(lambda x: -1 if x < 0 else 0 if x == 0 else 1)
    return data

def encode_categorical(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == object:
            encoders[column] = LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders

# Preprocess the data
train_data = preprocess_data(pd.read_csv('../labelled_training_data.csv'))
valid_data = preprocess_data(pd.read_csv('../labelled_validation_data.csv'))
test_data = preprocess_data(pd.read_csv('../labelled_testing_data.csv'))
# print(test_data.head(10))

# Convert categorical variables to numeric
train_data = pd.get_dummies(train_data)
valid_data = pd.get_dummies(valid_data)
test_data = pd.get_dummies(test_data)

#train_data, _ = encode_categorical(train_data)
#valid_data, _ = encode_categorical(valid_data)
#test_data, _ = encode_categorical(test_data)

# Split the data into input features (X) and the target variable (y)
X_train = train_data.drop('eventId', axis=1)
y_train = train_data['eventId']
X_valid = valid_data.drop('eventId', axis=1)
y_valid = valid_data['eventId']
X_test = test_data.drop('eventId', axis=1)
y_test = test_data['eventId']

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Function to create a shallow model
def shallow_model():
    model = Sequential()
    model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(1, activation='linear'))
    #model.compile(loss='mean_squared_error', optimizer='adam'), updated version is on the arguments of KerasRegressor
    return model

# Create the shallow model
# model = KerasRegressor(build_fn=shallow_model, epochs=100, batch_size=10, verbose=0)
model = KerasRegressor(model=shallow_model, loss='mean_squared_error', optimizer='adam', epochs=10, batch_size=100, verbose=1)

# Fit the model
model.fit(X_train, y_train)

# Predict the validation set results
y_pred = model.predict(X_valid)

# Calculate the Mean Squared Error (MSE)
#mse = mean_squared_error(y_valid, y_pred)
#print(f'MSE: {mse}')

# Calculate MSE for training data
y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)

# Calculate MSE for validation data
y_valid_pred = model.predict(X_valid)
valid_mse = mean_squared_error(y_valid, y_valid_pred)
print(f'MSE: {valid_mse}')

# Calculate MSE for test data
y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)

# Plot same as figure 2.2 form papper
# Assuming 'timestamp' is the column representing seconds from boot
timestamps = test_data['timestamp']
# Set the number of bins for your histogram
bins = int(max(timestamps) - min(timestamps))
plt.figure(figsize=(10, 6))
plt.hist(timestamps, bins=bins, edgecolor='black')
plt.title('Timeline of Events')
plt.xlabel('Timestamp (Seconds from Boot)')
plt.ylabel('Number of Events')
plt.show()

# Plot MSE for each dataset
#plt.figure(figsize=(10, 6))
#plt.bar(['Train', 'Validation', 'Test'], [train_mse, valid_mse, test_mse])
#plt.title('Mean Squared Error Comparison')
#plt.xlabel('Dataset')
#plt.ylabel('MSE')
#plt.show()

'''
sns.kdeplot(train_data['evil'], label='Train')
sns.kdeplot(valid_data['evil'], label='Validation')
sns.kdeplot(test_data['evil'], label='Test')
plt.legend(title='Dataset', labels=['Train', 'Validation', 'Test'])
plt.title('Density Plot of evil')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()
'''