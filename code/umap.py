import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys


# Function to preprocess data
def preprocess_data(data):
    columns_to_drop = ['timestamp','processName', 'hostName', 'eventName', 'stackAddresses', 'args']
    #columns_to_drop = ['processName', 'hostName', 'eventName', 'stackAddresses', 'args', 'processId', 'parentProcessId', 'mountNamespace', 'returnValue', 'threadId', 'eventId', 'argsNum']
    data = data.drop(columns=columns_to_drop)
    data['processId'] = data['processId'].apply(lambda x: 1 if x in [0, 1, 2] else 0)
    data['parentProcessId'] = data['parentProcessId'].apply(lambda x: 1 if x in [0, 1, 2] else 0)
    data['userId'] = data['userId'] #.apply(lambda x: 0 if x < 1000 else 1)
    data['mountNamespace'] = data['mountNamespace'].apply(lambda x: 1 if x == 4026531840 else 1)
    data['returnValue'] = data['returnValue'].apply(lambda x: -1 if x < 0 else 0 if x == 0 else 1)
    return data

# Preprocess the data
train_data = preprocess_data(pd.read_csv('../labelled_training_data.csv')).sample(10000)
valid_data = preprocess_data(pd.read_csv('../labelled_validation_data.csv')).sample(10000)
test_data = preprocess_data(pd.read_csv('../labelled_testing_data.csv')).sample(10000)

#print(test_data.head(10))
#sys.exit()

# Convert categorical variables to numeric
train_data = pd.get_dummies(train_data)
valid_data = pd.get_dummies(valid_data)
test_data = pd.get_dummies(test_data)

# Fit the TSNE model on training data and transform both training and testing data
tsne = TSNE(n_components=2, random_state=42)
embedding_train = tsne.fit_transform(train_data[['processId', 'parentProcessId', 'userId', 'mountNamespace', 'eventId', 'argsNum', 'returnValue']])
embedding_test = tsne.fit_transform(test_data[['processId', 'parentProcessId', 'userId', 'mountNamespace', 'eventId', 'argsNum', 'returnValue']])
#embedding_train = tsne.fit_transform(train_data[['userId', 'timestamp']])
#embedding_test = tsne.fit_transform(test_data[['userId','timestamp']])

# t-SNE visualization showing the overlap between the training and testing dataset
plt.figure(figsize=(12, 8))
plt.scatter(embedding_train[:, 0], embedding_train[:, 1], label='Train', alpha=0.5)
plt.scatter(embedding_test[:, 0], embedding_test[:, 1], label='Test', alpha=0.5)
plt.legend()
plt.title('t-SNE of training and testing dataset')
plt.show()

# t-SNE visualization highlighting the trails of evil events
# Assuming "evilEvent" column is your boolean column indicating evil events
evil_train = train_data['evil']
evil_test = test_data['evil']

plt.figure(figsize=(12, 8))
plt.scatter(embedding_train[evil_train, 0], embedding_train[evil_train, 1], label='Evil Train', alpha=0.5)
plt.scatter(embedding_test[evil_test, 0], embedding_test[evil_test, 1], label='Evil Test', alpha=0.5)
plt.legend()
plt.title('t-SNE highlighting trails of evil events')
plt.show()