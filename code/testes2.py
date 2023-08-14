import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import OneClassSVM

# import data
data = pd.read_csv("../labelled_testing_data.csv")
# input data
df = data[["timestamp", "userId"]]
print(df.describe())

# Read the csv files
#train_df = pd.read_csv('../labelled_training_data.csv')
#valid_df = pd.read_csv('../labelled_validation_data.csv')
#test_df = pd.read_csv('../labelled_testing_data.csv')