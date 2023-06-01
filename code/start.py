#import pandas as pd
#from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LogisticRegression
#import matplotlib.pyplot as plt

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # viz
import matplotlib.pyplot as plt # viz
from scipy import stats
import json
from typing import List, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn import metrics, linear_model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

# Load datasets
train_data = pd.read_csv('../labelled_training_data.csv')
test_data = pd.read_csv('../labelled_testing_data.csv')
valid_data = pd.read_csv('../labelled_validation_data.csv')

# Validate if all datasets have the same columns
assert train_data.columns.all() == test_data.columns.all() == valid_data.columns.all()

# Print all columns types
#print(train_data.dtypes)
#print(train_data.head())
#print(train_data.describe(include=['object', 'float', 'int']))

# Train Plot
#train_data.evil.value_counts().plot(kind='bar', title='Label Frequency for evil label in Train Dataset')
#plt.show()
#train_data.sus.value_counts().plot(kind='bar', title='Label Frequency for sus label in Train Dataset')
#plt.show()

# Test Plot
#test_data.evil.value_counts().plot(kind='bar', title='Label Frequency for evil label in Test Dataset')
#plt.show()
#test_data.sus.value_counts().plot(kind='bar', title='Label Frequency for sus label in Test Dataset')
#plt.show()

# Validation Plot
#valid_data.evil.value_counts().plot(kind='bar', title='Label Frequency for evil label in Validation Dataset')
#plt.show()
#valid_data.sus.value_counts().plot(kind='bar', title='Label Frequency for sus label in Validation Dataset')
#plt.show()

# print(valid_data.sus.value_counts())
#sus
#0    188181
#1       786


#print(test_data.groupby(['sus', 'evil'])[['timestamp']].count())
#test_data.groupby(['sus', 'evil'])[['timestamp']].count().plot(kind='bar')
#plt.show()





def process_args_row(row):
    """
    Takes an single value from the 'args' column
    and returns a processed dataframe row
    
    Args:
        row: A single 'args' value/row
        
    Returns:
        final_df: The processed dataframe row
    """
    
    row = row.split('},')
    row = [string.replace("[", "").replace("]", "").replace("{", "").replace("'", "").replace("}", "").lstrip(" ") for string in row]
    row = [item.split(',') for item in row]
    
    processed_row = []
    for lst in row:
        for key_value in lst:
            key, value = key_value.split(': ', 1)
            if not processed_row or key in processed_row[-1]:
                processed_row.append({})
            processed_row[-1][key] = value
    
    json_row = json.dumps(processed_row)
    row_df = pd.json_normalize(json.loads(json_row))
    
    final_df = row_df.unstack().to_frame().T.sort_index(1,1)
    final_df.columns = final_df.columns.map('{0[0]}_{0[1]}'.format)
    
    return final_df

"""
sample = train_data['args'].sample(n=15, random_state=1)
sample_df = pd.DataFrame(sample)

data = sample_df['args'].tolist()
processed_dataframes = []

for row in data:
    ret = process_args_row(row)
    processed_dataframes.append(ret)

processed = pd.concat(processed_dataframes).reset_index(drop=True)
processed.columns = processed.columns.str.lstrip()
processed

sample_df = sample_df.reset_index(drop=True)
merged_sample = pd.concat([sample_df, processed], axis=1)
merged_sample

# Taken from here - https://github.com/jinxmirror13/BETH_Dataset_Analysis
train_data["processId"] = train_data["processId"].map(lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
train_data["parentProcessId"] = train_data["parentProcessId"].map(lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
train_data["userId"] = train_data["userId"].map(lambda x: 0 if x < 1000 else 1)  # Map to OS/not OS
train_data["mountNamespace"] = train_data["mountNamespace"].map(lambda x: 0 if x == 4026531840 else 1)  # Map to mount access to mnt/ (all non-OS users) /elsewhere
train_data["eventId"] = train_data["eventId"]  # Keep eventId values (requires knowing max value)
train_data["returnValue"] = train_data["returnValue"].map(lambda x: 0 if x == 0 else (1 if x > 0 else 2))  # Map to success/success with value/error

train_data.head(5)

"""

def process_args_dataframe(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Processes the `args` column within the dataset
    """
    
    processed_dataframes = []
    data = df[column_name].tolist()
    
    # Debug counter
    counter = 0
    
    for row in data:
        if row == '[]': # If there are no args
            pass
        else:
            try:
                ret = process_args_row(row)
                processed_dataframes.append(ret)
            except:
                print(f'Error Encounter: Row {counter} - {row}')

            counter+=1
        
    processed = pd.concat(processed_dataframes).reset_index(drop=True)
    processed.columns = processed.columns.str.lstrip()
    
    df = pd.concat([df, processed], axis=1)
    
    return df

def prepare_dataset(df: pd.DataFrame, process_args=False) -> pd.DataFrame:
    """
    Prepare the dataset by completing the standard feature engineering tasks
    """
    
    df["processId"] = train_data["processId"].map(lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
    df["parentProcessId"] = train_data["parentProcessId"].map(lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
    df["userId"] = train_data["userId"].map(lambda x: 0 if x < 1000 else 1)  # Map to OS/not OS
    df["mountNamespace"] = train_data["mountNamespace"].map(lambda x: 0 if x == 4026531840 else 1)  # Map to mount access to mnt/ (all non-OS users) /elsewhere
    df["eventId"] = train_data["eventId"]  # Keep eventId values (requires knowing max value)
    df["returnValue"] = train_data["returnValue"].map(lambda x: 0 if x == 0 else (1 if x > 0 else 2))  # Map to success/success with value/error
    
    if process_args is True:
        df = process_args_dataframe(df, 'args')
        
    features = df[["processId", "parentProcessId", "userId", "mountNamespace", "eventId", "argsNum", "returnValue"]]
    labels = df['sus']
        
    return features, labels

def metric_printer(y_true, y_pred):
    
    y_true[y_true == 1] = -1
    y_true[y_true == 0] = 1
    
    metric_tuple = precision_recall_fscore_support(y_true, y_pred, average="weighted", pos_label = -1)
    print(f'Precision:\t{metric_tuple[0]}')
    print(f'Recall:\t\t{metric_tuple[1]:.3f}')
    print(f'F1-Score:\t{metric_tuple[2]:.3f}')

def output_roc_plot(y, pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Isolation Forest')
    display.plot()
    plt.show()

def prepare_tensor_dataset(df: pd.DataFrame, feature_cols: List, label_col: str) -> Tuple[TensorDataset, DataLoader]:
    """
    Converts an inpurt Pandas DataFrame to a Tensor Dataset and Data Loader.
    """
    if all([col in df.columns for item in feature_cols]) and label_col in df.columns:
        
        labels = pd.DataFrame(df[[label_col]])
        features = pd.DataFrame(df[feature_cols])

        data_tensor = torch.as_tensor(features.values, dtype=torch.int32)
        label_tensor = torch.as_tensor(train_labels.values, dtype=torch.int32)
        
        tensor_dataset = TensorDataset(data_tensor, label_tensor)
        
        data_ldr = train_data_ldr = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        return tensor_dataset, data_ldr
    else:
        raise ValueError('Unable to find all columns')

#train_no_args_feats, train_no_args_labels = prepare_dataset(train_data)
#print(train_no_args_feats.head())

train_df_feats, train_df_labels = prepare_dataset(train_data)
test_df_feats, test_df_labels = prepare_dataset(test_data)
val_df_feats, val_df_labels = prepare_dataset(valid_data)


clf = IsolationForest(contamination=0.1, random_state=0).fit(train_df_feats)

y_pred= clf.predict(val_df_feats)
y_probas = clf.score_samples(val_df_feats)
metric_printer(val_df_labels, y_pred)


y_pred= clf.predict(test_df_feats)
y_probas = clf.score_samples(test_df_feats)
metric_printer(test_df_labels, y_pred)


train_non_outliers = train_df_feats[train_df_labels==0]
clf = linear_model.SGDOneClassSVM(random_state=0).fit(train_non_outliers)

y_preds = clf.predict(val_df_feats)
metric_printer(val_df_labels, y_preds)

y_preds = clf.predict(test_df_feats)
metric_printer(test_df_labels, y_preds)


train_data = pd.DataFrame(train_data[["processId", "parentProcessId", "userId", "mountNamespace", "eventId", "argsNum", "returnValue"]])
train_labels = pd.DataFrame(train_data[["sus"]])

data_tensor = torch.as_tensor(train_data.values, dtype=torch.int32)
label_tensor = torch.as_tensor(train_labels.values, dtype=torch.int32)

train_dataset = TensorDataset(data_tensor, label_tensor)

train_data_ldr = DataLoader(train_dataset, batch_size=64, shuffle=True)

for i, (x, y) in enumerate(train_data_ldr):
    if i == 1:
        break
    else:
        print(f'Index: {i} \n Data Tensor: {x} \n Label Tensor: {y}')

label_col = 'sus'
feat_cols = ["processId", "parentProcessId", "userId", "mountNamespace", "eventId", "argsNum", "returnValue"]

train_tensor_dataset, train_data_ldr = prepare_tensor_dataset(train_data, feat_cols, label_col)


# entropy between the different datasets
"""
datasets = [train_data, test_data, valid_data]

entropy_values = []
for dataset in datasets:
    dataset_entropy_values = []
    for col in dataset.columns:
        if col == 'timestamp':
            pass
        else:
            counts = dataset[col].value_counts()
            col_entropy = stats.entropy(counts)
            dataset_entropy_values.append(col_entropy)
            
    entropy_values.append(dataset_entropy_values)

plt.boxplot(entropy_values)
plt.title('Boxplot of Entropy Values')
plt.ylabel("entropy values")
plt.xticks([0,1,2,3],labels=['','train', 'test', 'validate'])
plt.show()
"""

#correlaçao de dados entre os diferentes datasets
"""
def dataset_to_corr_heatmap(dataframe, title, ax):
    corr = dataframe.corr()
    sns.heatmap(corr, ax = ax, annot=True, cmap="YlGnBu")
    ax.set_title(f'Correlation Plot for {title}')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (15,20))
fig.tight_layout(pad=10.0)
datasets = [train_data, test_data, valid_data]
dataset_names = ['train', 'test', 'validation']
axs = [ax1, ax2, ax3]

for dataset, name, ax in zip(datasets, dataset_names, axs):
    dataset_to_corr_heatmap(dataset, name, ax)

"""

"""
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

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Validate model
accuracy = model.score(X_valid, y_valid)
print(f'Validation Accuracy: {accuracy*100:.2f}%')

# Test model
accuracy = model.score(X_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')
"""