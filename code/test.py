# import relavant libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create train and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# print the original dataset
# print('Original Train: \n', X_train[0:5])
# print('Original Test: \n', X_test[0:5])

# print("-"*50)
# print('Using transform()')

# use fit() and transform separately
std_slc = StandardScaler()
std_slc.fit(X_train)
X_train_std = std_slc.transform(X_train)
X_test_std = std_slc.transform(X_test)

# print('Transformed Train: \n', X_train_std[0:5])
# print('Transformed Test: \n', X_test_std[0:5])

# print("-"*50)
# print('Using fit_transform()')
# use fit and transform in a single function call
std_slc2 = StandardScaler()
print(X_train)
# X_train_std2 = std_slc2.fit_transform(X_train)
# X_test_std2 = std_slc2.transform(X_test)

# print('Transformed Train: \n', X_train_std[0:5])
# print('Transformed Test: \n', X_test_std[0:5])

# # verify that using fit_transform() equates to using fit() and transform() together
# if (X_train_std2 == X_train_std).all() and (X_test_std2 == X_test_std).all():
#     print ('both are equivalent')