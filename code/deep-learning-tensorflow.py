import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# Load datasets
train_data = pd.read_csv('../labelled_training_data.csv')
test_data = pd.read_csv('../labelled_testing_data.csv')
valid_data = pd.read_csv('../labelled_validation_data.csv')

# Let's assume the last column is what we want to predict
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

X_valid = valid_data.iloc[:, :-1]
y_valid = valid_data.iloc[:, -1]

# Preprocess data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)

# Define model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(y_train.unique()), activation='softmax'), # assuming your target is categorical
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# Train model
model.fit(
    X_train,
    y_train,
    validation_data=(X_valid, y_valid),
    epochs=10,
)

# Validate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')
