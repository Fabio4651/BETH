import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Hyperparameters and dataset size definition
dataset_size = 20000
split_percent = int(dataset_size * 0.2)

#SVM
#C: Regularization parameter. The strength of the regularization is inversely proportional to C.
#kernel: Specifies the kernel type to be used in the algorithm. It could be 'linear', 'poly', 'rbf', 'sigmoid', etc.
C = 1.0
kernel = 'rbf'

#MLP
#hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden layer.
#learning_rate_init: The initial learning rate used. It controls the step-size in updating the weights.
#max_iter: Maximum number of iterations. The solver iterates until convergence (determined by 'tol') or this number of iterations.
#alpha: L2 penalty (regularization term) parameter.
#solver: The solver for weight optimization. Can be lbfgs, sgd, or adam.
hidden_layer_sizes = 64
learning_rate_init = 0.003
max_iter = 100
alpha = 0.0001
solver = 'adam'

#print("Debug: Initialize CSV reading")

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

#print("Debug: CSV in memory")

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
    X = np.array([seq.reshape(-1) for seq in sequences])
    y = data['sus'][50:].values
    return X, y

# Preprocess the datasets
train_data = preprocess(train_data)
val_data = preprocess(val_data)
test_data = preprocess(test_data)

# Convert DataFrames to sequences
X_train, y_train = df_to_sequences(train_data)
X_val, y_val = df_to_sequences(val_data)
X_test, y_test = df_to_sequences(test_data)



# Define the SVM model
model_svm = SVC(kernel=kernel, C=C, random_state=42, verbose=True)

# Train the SVM model
model_svm.fit(X_train, y_train)

# Evaluate SVM model
svm_train_pred = model_svm.predict(X_train)
svm_val_pred = model_svm.predict(X_val)
svm_test_pred = model_svm.predict(X_test)
print(f"SVM Train Accuracy: {accuracy_score(y_train, svm_train_pred)}")
print(f"SVM Validation Accuracy: {accuracy_score(y_val, svm_val_pred)}")
print(f"SVM Test Accuracy: {accuracy_score(y_test, svm_test_pred)}")



# Define the MLP model
model_mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), max_iter=max_iter, alpha=alpha,
                    solver=solver, verbose=10, random_state=42,
                    learning_rate_init=learning_rate_init)

# Train the MLP model
model_mlp.fit(X_train, y_train)

# Evaluate MLP model
mlp_train_pred = model_mlp.predict(X_train)
mlp_val_pred = model_mlp.predict(X_val)
mlp_test_pred = model_mlp.predict(X_test)
print(f"MLP Train Accuracy: {accuracy_score(y_train, mlp_train_pred)}")
print(f"MLP Validation Accuracy: {accuracy_score(y_val, mlp_val_pred)}")
print(f"MLP Test Accuracy: {accuracy_score(y_test, mlp_test_pred)}")


# Some plots
# Plot accuracies
svm_accuracies = [accuracy_score(y_train, svm_train_pred), accuracy_score(y_val, svm_val_pred), accuracy_score(y_test, svm_test_pred)]
mlp_accuracies = [accuracy_score(y_train, mlp_train_pred), accuracy_score(y_val, mlp_val_pred), accuracy_score(y_test, mlp_test_pred)]

labels = ['Train', 'Validation', 'Test']
x = np.arange(len(labels))

fig, ax = plt.subplots()
rects1 = ax.bar(x - 0.2/2, svm_accuracies, 0.2, label='SVM')
rects2 = ax.bar(x + 0.2/2, mlp_accuracies, 0.2, label='MLP')

ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by model and dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()
