import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Function to preprocess data
def normalize(data):
    #columns_to_drop = ['processName', 'hostName', 'eventName', 'stackAddresses', 'args']
    #data = data.drop(columns=columns_to_drop)
    #data['processId'] = data['processId'].apply(lambda x: 1 if x in [0, 1, 2] else 0)
    #data['parentProcessId'] = data['parentProcessId'].apply(lambda x: 1 if x in [0, 1, 2] else 0)
    data['userId'] = data['userId'].apply(lambda x: 1 if x < 1000 else 0)
    #data['mountNamespace'] = data['mountNamespace'].apply(lambda x: 1 if x == 4026531840 else 0)
    #data['returnValue'] = data['returnValue'].apply(lambda x: -1 if x < 0 else 0 if x == 0 else 1)
    return data

# Load the data
train_data = normalize(pd.read_csv('../labelled_training_data.csv'))
valid_data = normalize(pd.read_csv('../labelled_validation_data.csv'))
test_data = normalize(pd.read_csv('../labelled_testing_data.csv'))

# Select the columns
cols = ['timestamp', 'userId', 'sus', 'evil']
train_data = train_data[cols]
valid_data = valid_data[cols]
test_data = test_data[cols]

# Generate labels and add to the dataframe
def generate_label(row):
    if row['evil'] == 1 and row['sus'] == 1:
        return 1  # 'malicious'
    elif row['evil'] == 1 or row['sus'] == 1:
        return 2  # 'suspicious'
    else:
        return 0  # 'normal'

for df in [train_data, valid_data, test_data]:
    df['label'] = df.apply(generate_label, axis=1)

# Normalize your features
scaler = preprocessing.StandardScaler()
for df in [train_data, valid_data, test_data]:
    df[cols] = scaler.fit_transform(df[cols])

# Split the data into features and targets
X_train, y_train = train_data[cols], train_data['label']
X_valid, y_valid = valid_data[cols], valid_data['label']
X_test, y_test = test_data[cols], test_data['label']

# Convert the data into PyTorch tensors
X_train, y_train = torch.tensor(X_train.values), torch.tensor(y_train.values)
X_valid, y_valid = torch.tensor(X_valid.values), torch.tensor(y_valid.values)
X_test, y_test = torch.tensor(X_test.values), torch.tensor(y_test.values)

# Define the neural network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(4, 10)
        self.fc2 = torch.nn.Linear(10, 3)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the network, loss function and optimizer
model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the network
for epoch in range(100):  # number of epochs
    optimizer.zero_grad()
    outputs = model(X_train.float())
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# Validate the model
model.eval()  # set the model to evaluation mode
with torch.no_grad():
    outputs = model(X_valid.float())
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == y_valid).sum().item()
    print(f'Validation accuracy: {100 * correct / y_valid.size(0)}%')

# Test the model
model.eval()  # set the model to evaluation mode
with torch.no_grad():
    outputs = model(X_test.float())
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == y_test).sum().item()
    print(f'Test accuracy: {100 * correct / y_test.size(0)}%')


# Assuming 'y_test' are your true labels and 'predicted' are your predicted labels
#cf_matrix = confusion_matrix(y_test, predicted)
#plt.figure(figsize=(10,7))
#sns.heatmap(cf_matrix, annot=True, cmap='Blues')
#plt.xlabel('Predicted')
#plt.ylabel('Actual')
#plt.show()

# Assuming the initial 'zones' before model prediction are stored in 'initial_zones'
# And the model's predicted 'zones' are stored in 'predicted_zones'

labels = ['normal', 'malicious', 'suspicious']

# Define a mapping from numeric labels to string labels
label_mapping = {0: 'normal', 1: 'malicious', 2: 'suspicious'}

# Use the mapping to convert y_test and predicted to string labels
initial_zones = [label_mapping[label] for label in y_test.numpy()]
predicted_zones = [label_mapping[label] for label in predicted.numpy()]

# Count the number of instances in each zone
initial_counts = [initial_zones.count(label) for label in labels]
predicted_counts = [predicted_zones.count(label) for label in labels]

# Set up the plot
x = np.arange(len(labels))  # label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, initial_counts, width, label='Initial')
rects2 = ax.bar(x + width/2, predicted_counts, width, label='Predicted')

# Add some text for labels, title and custom x-axis tick labels
ax.set_ylabel('Count')
ax.set_title('Count by zone and type')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()