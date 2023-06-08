'''
Solucao invalida.. era bom mas não fucniona porque só temos os dados de malicious no ultimo dataset.. pensar em como fazer isto..
'''


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Function to preprocess data
def normalize(data):
    columns_to_drop = ['processName', 'hostName', 'eventName', 'stackAddresses', 'args']
    data = data.drop(columns=columns_to_drop)
    data['processId'] = data['processId'].apply(lambda x: 1 if x in [0, 1, 2] else 0)
    data['parentProcessId'] = data['parentProcessId'].apply(lambda x: 1 if x in [0, 1, 2] else 0)
    data['userId'] = data['userId'].apply(lambda x: 1 if x < 1000 else 0)
    data['mountNamespace'] = data['mountNamespace'].apply(lambda x: 1 if x == 4026531840 else 0)
    data['returnValue'] = data['returnValue'].apply(lambda x: -1 if x < 0 else 0 if x == 0 else 1)
    return data

# Define a function to convert 'sus' and 'evil' to 'normal', 'suspicious', and 'malicious'
def convert_columns(df):
    conditions = [
        (df['sus'] == 0) & (df['evil'] == 0),
        (df['sus'] == 1) & (df['evil'] == 0),
        (df['evil'] == 1)]
    choices = ['normal', 'suspicious', 'malicious']
    df['label'] = np.select(conditions, choices)
    return df

# Load data
# Preprocess the data
train_df = normalize(pd.read_csv('../labelled_training_data.csv'))
validation_df = normalize(pd.read_csv('../labelled_validation_data.csv'))
test_df = normalize(pd.read_csv('../labelled_testing_data.csv'))

# Apply the function to your dataframes
train_df = convert_columns(train_df)
validation_df = convert_columns(validation_df)
test_df = convert_columns(test_df)

# Split data and labels
X = df.drop(['label', 'sus', 'evil'], axis=1)
y = df['label']

# Now split data and labels
X_train = train_df.drop(['label', 'sus', 'evil'], axis=1)
y_train = train_df['label']

X_val = validation_df.drop(['label', 'sus', 'evil'], axis=1)
y_val = validation_df['label']

X_test = test_df.drop(['label', 'sus', 'evil'], axis=1)
y_test = test_df['label']



print(y_train.unique())
print(y_val.unique())
print(y_test.unique())



# Encode labels
encoder = LabelEncoder()

encoder.fit(y_train)
encoded_Y_train = encoder.transform(y_train)
dummy_y_train = np_utils.to_categorical(encoded_Y_train)

encoder.fit(y_val)
encoded_Y_val = encoder.transform(y_val)
dummy_y_val = np_utils.to_categorical(encoded_Y_val)

encoder.fit(y_test)
encoded_Y_test = encoder.transform(y_test)
dummy_y_test = np_utils.to_categorical(encoded_Y_test)


# Convert integers to dummy variables (i.e. one hot encoded)
dummy_y_train = np_utils.to_categorical(encoded_Y_train)
dummy_y_val = np_utils.to_categorical(encoded_Y_val)
dummy_y_test = np_utils.to_categorical(encoded_Y_test)

# Define and compile model
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 is the number of classes
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(X_train, dummy_y_train, validation_data=(X_val, dummy_y_val), epochs=10, batch_size=64)

# Evaluate model
_, accuracy = model.evaluate(X_test, dummy_y_test)
print('Accuracy: %.2f' % (accuracy*100))

# Plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Plot training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

# Plot confusion matrix
predictions = model.predict_classes(X_test)
cm = confusion_matrix(encoded_Y_test, predictions)

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), ylabel='True label', xlabel='Predicted label')
plt.show()
