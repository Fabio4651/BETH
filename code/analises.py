import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
train_data = pd.read_csv('../labelled_training_data.csv')

# You mentioned this preprocessing step, so let's include it:
train_data["processId"] = train_data["processId"].map(lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
train_data["parentProcessId"] = train_data["parentProcessId"].map(lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
train_data["userId"] = train_data["userId"].map(lambda x: 0 if x < 1000 else 1)  # Map to OS/not OS
train_data["mountNamespace"] = train_data["mountNamespace"].map(lambda x: 0 if x == 4026531840 else 1)  # Map to mount access to mnt/ (all non-OS users) /elsewhere
train_data["returnValue"] = train_data["returnValue"].map(lambda x: 0 if x == 0 else (1 if x > 0 else 2))  # Map to success/success with value/error


#train_data = normalize(pd.read_csv('../labelled_training_data.csv'))
#valid_data = normalize(pd.read_csv('../labelled_validation_data.csv'))
#test_data = normalize(pd.read_csv('../labelled_testing_data.csv'))


# Distribution of processId, parentProcessId, userId, mountNamespace, and returnValue
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
sns.countplot(data=train_data, x='processId', ax=axs[0, 0])
sns.countplot(data=train_data, x='parentProcessId', ax=axs[0, 1])
sns.countplot(data=train_data, x='userId', ax=axs[1, 0])
sns.countplot(data=train_data, x='mountNamespace', ax=axs[1, 1])
plt.tight_layout()
plt.show()

sns.countplot(data=train_data, x='returnValue')
plt.title('Distribution of Return Values')
plt.show()

# Event count for each unique eventId
plt.figure(figsize=(12,6))
train_data['eventId'].value_counts().plot(kind='bar')
plt.title('Number of Occurrences for Each EventId')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
