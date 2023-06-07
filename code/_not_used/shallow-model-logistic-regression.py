import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load datasets
train_data = pd.read_csv('../labelled_training_data.csv')
test_data = pd.read_csv('../labelled_testing_data.csv')
valid_data = pd.read_csv('../labelled_validation_data.csv')

# Validate if all datasets have the same columns
assert train_data.columns.all() == test_data.columns.all() == valid_data.columns.all()

# Print all columns types
print(train_data.dtypes)

print(train_data.head())

# print(train_data.describe(include=['object', 'float', 'int']))

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