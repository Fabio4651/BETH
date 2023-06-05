import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import SparseCategoricalCrossentropy
import ast



# Load the training and testing CSV files
train_df = pd.read_csv('labelled_training_data.csv')
test_df = pd.read_csv('labelled_testing_data.csv')

# Preprocessing for training data

data_str = train_df['args']

data = ast.literal_eval(data_str)

for item in data:
    if isinstance(item['value'], str):
        item['value'] = float(item['value'])



# Preprocessing for training data
# Handling missing values
imputer = SimpleImputer(strategy='mean')
data = imputer.fit_transform([data])


print(data)


# Encoding categorical variables
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(train_df[['categorical_feature']]).toarray()
train_df_encoded = pd.concat([train_df, pd.DataFrame(encoded_features)], axis=1)
train_df_encoded.drop(['categorical_feature'], axis=1, inplace=True)


# Normalizing numerical features
scaler = MinMaxScaler()
train_df_encoded['numeric_feature'] = scaler.fit_transform(train_df_encoded[['numeric_feature']])

