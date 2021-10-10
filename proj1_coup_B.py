from sklearn import datasets  # read the data sets
import numpy as np  # needed for arrays
from sklearn.model_selection import train_test_split  # splits database
from sklearn.preprocessing import StandardScaler  # standardize data
from sklearn.linear_model import Perceptron  # Perceptron algorithm
from sklearn.metrics import accuracy_score  # grade the results
from pandas import DataFrame, read_csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression  # Logistic Regression algorithm
from sklearn.svm import SVC  # Support Vector Machine algorithm
from sklearn.tree import DecisionTreeClassifier  # Decision Tree algorithm
from sklearn.ensemble import RandomForestClassifier  # Random Forest algorithm
from sklearn.neighbors import KNeighborsClassifier  # KNN algorithm

# Perceptron ##########################################################################################################
coupon = read_csv('coupon.csv')  # load the data set
X = coupon.iloc[:, :-1]  # separate the features we want
y = coupon.iloc[:, -1]  # extract the classifications
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)

# scale X by removing the mean and setting the variance to 1 on all features.
# the formula is z=(x-u)/s where u is the mean and s is the standard deviation.
# (mean and standard deviation may be overridden with options...)

# this block prepares the data for all the algorithms
sc = StandardScaler()  # create the standard scalar
sc.fit(X_train)  # compute the required transformation
X_train_std = sc.transform(X_train)  # apply to the training data
X_test_std = sc.transform(X_test)  # and SAME transformation of test data

# perceptron linear
# epoch is one forward and backward pass of all training samples
# (also known as an iteration)
# eta0 is rate of convergence
# max_iter, tol, if it is too low it is never achieved
# and continues to iterate to max_iter when above tol
# fit_intercept, fit the intercept or assume it is 0
# slowing it down is very effective, eta is the learning rate
ppn = Perceptron(max_iter=10, tol=1e-9, eta0=0.001,
                 fit_intercept=True, random_state=0, verbose=False)
ppn.fit(X_train_std, y_train)  # do the training
print('\n')
print('Perceptron Results')
print('Number in test ', len(y_test))
y_pred = ppn.predict(X_test_std)  # now try with the test data
# Note that this only counts the samples where the predicted value was wrong
print('Misclassified samples: %d' % (y_test != y_pred).sum())  # how'd we do?
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# this block prepares the data for all the algorithms
# vstack puts first array above the second in a vertical stack
# hstack puts first array to left of the second in a horizontal stack
# NOTE the double parens!
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print('Number in combined ', len(y_combined))
# we did the stack so we can see how the combination of test and train data did

# testing the combined data
y_combined_pred = ppn.predict(X_combined_std)
print('Misclassified combined samples: %d' % \
      (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
print('\n')

# Linear Regression ###################################################################################################
# create logistic regression component.
# C is the inverse of the regularization strength. Smaller -> stronger!
# C is used to penalize extreme parameter weights.
# solver is the particular algorithm to use
# multi_class determines how loss is computed
# ovr -> binary problem for each label
lr = LogisticRegression(C=10, solver='liblinear', multi_class='ovr', random_state=0)
lr.fit(X_train_std, y_train)  # apply the algorithm to training data
y_pred = lr.predict(X_test_std)  # now try with the test data
print('Logistic Regression Results')
print('Number in test ', len(y_test))
# Note that this only counts the samples where the predicted value was wrong
print('Misclassified samples: %d' % (y_test != y_pred).sum())  # how'd we do?
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Number in combined ', len(y_combined))
# testing the combined data
y_combined_pred = lr.predict(X_combined_std)
print('Misclassified combined samples: %d' % \
      (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
print('\n')

# Support Vector Machine ###############################################################################################
# Kernel rbf radial bias function
# gamma increases influence of each sample
# increasing C increases error penalties

svm = SVC(kernel='rbf', tol=1e-3, random_state=0, gamma=0.3, C=10.0, verbose=False)
svm.fit(X_train_std, y_train)  # apply the algorithm to training data
y_pred = svm.predict(X_test_std)  # now try with the test data
print('Support Vector Machine Results')
print('Number in test ', len(y_test))
# Note that this only counts the samples where the predicted value was wrong
print('Misclassified samples: %d' % (y_test != y_pred).sum())  # how'd we do?
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Number in combined ', len(y_combined))
# testing the combined data
y_combined_pred = svm.predict(X_combined_std)
print('Misclassified combined samples: %d' % \
      (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
print('\n')

# Decision Tree #######################################################################################################
# create the classifier and train it
tree = DecisionTreeClassifier(criterion='entropy', max_depth=25, random_state=0)
tree.fit(X_train, y_train)

# now try with the test data
y_pred = tree.predict(X_test)
print('Decision Tree Results')
print('Number in test ', len(y_test))
# Note that this only counts the samples where the predicted value was wrong
print('Misclassified samples: %d' % (y_test != y_pred).sum())  # how'd we do?
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Number in combined ', len(y_combined))
# combine the train and test data, also used for Random Forest
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
# testing the combined data
y_combined_pred = tree.predict(X_combined)
print('Misclassified combined samples: %d' % \
      (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
print('\n')

# Random Forest ########################################################################################################
# create the classifier and train it
# n_estimators is the number of trees in the forest
# the entropy choice grades based on information gained
# n_jobs allows multiple processors to be used, 6 cores on my PC at home, adjust as needed
forest = RandomForestClassifier(criterion='entropy', n_estimators=101, random_state=1, n_jobs=6)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)  # see how we do on the test data
print('Random Forest Results')
print('Number in test ', len(y_test))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Number in combined ', len(y_combined))
# see how we do on the combined data
y_combined_pred = forest.predict(X_combined)
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
print('\n')

# K-Nearest Neighbors ##################################################################################################
# create the classifier and fit it
# where p=2 specifies sqrt(sum of squares). (p=1 is Manhattan distance)
# for neighs in [1,5,51]:
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
# run on the test data and print results and check accuracy
y_pred = knn.predict(X_test_std)
print('K-Nearest Neighbors Results')
print('Number in test ', len(y_test))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
# combine the train and test data
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print('Number in combined ', len(y_combined))
# check results on combined data
y_combined_pred = knn.predict(X_combined_std)
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
