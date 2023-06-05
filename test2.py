import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Load the training dataset
train_df = pd.read_csv('labelled_training_data.csv')

# Split the training data into features (X_train) and labels (y_train)
X_train = train_df.drop(['sus', 'evil'], axis=1)
y_train = train_df[['sus', 'evil']]

# Load the testing dataset
test_df = pd.read_csv('labelled_testing_data.csv')

# Split the testing data into features (X_test) and labels (y_test)
X_test = test_df.drop(['sus', 'evil'], axis=1)
y_test = test_df[['sus', 'evil']]

# Preprocessing
# Handling missing values in training data
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)

# Normalizing numerical features in training data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Preprocessing testing data
# Handling missing values in testing data
X_test = imputer.transform(X_test)

# Normalizing numerical features in testing data
X_test = scaler.transform(X_test)

# Define the neural network architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model on the training data
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=10,
                    validation_split=0.2)

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(X_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)
