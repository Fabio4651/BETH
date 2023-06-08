import pandas as pd
import matplotlib.pyplot as plt

# Read the csv files
#train_df = pd.read_csv('../labelled_training_data.csv')
#valid_df = pd.read_csv('../labelled_validation_data.csv')
#test_df = pd.read_csv('../labelled_testing_data.csv')

test_df = pd.read_csv('../labelled_training_data.csv')

# Convert 'timestamp' to the index
test_df.set_index('timestamp', inplace=True)

# Convert 'userId' to categorical if it's not
test_df['userId'] = test_df['userId'].astype('category')

# Get the number of unique categories
num_categories = len(test_df['userId'].cat.categories)

# Create a new column 'signal' to spread out the user ids as different signals
test_df['signal'] = test_df['userId'].cat.codes + 1

# Plot the data
plt.figure(figsize=(12, 8))
#plt.plot(test_df.index, test_df['signal'], marker='o', linestyle='', ms=2)
plt.plot(test_df.index, test_df['signal'])
plt.yticks(range(1, num_categories+1), test_df['userId'].cat.categories)
plt.xlabel('Timestamp')
plt.ylabel('User ID')
plt.show()