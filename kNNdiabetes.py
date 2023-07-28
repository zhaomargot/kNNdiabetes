# Margot Zhao

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

# (1) read data in a dataframe "diabetes_knn"
diabetes_knn = pd.read_csv("diabetes.csv")
# display all columns
pd.set_option("display.max_columns", None)

# (2) determine dimensions of the dataframe
print("Dimensions: ", diabetes_knn.ndim)
print()
# the dataframe has 2 dimensions

# (3) update the dataframe to account for missing values
print("Number of Missing Values:")
print(diabetes_knn.isnull().sum())
print()
# hence, assume there are no missing values,

# (4) create feature matrix
X_diabetes = diabetes_knn.drop(columns="Outcome")
print(X_diabetes.shape)
# create target vector
Y_diabetes = diabetes_knn["Outcome"]
print(Y_diabetes.shape)

# (5) standardize attributes of feature matrix
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# call to fit
scaler.fit(X_diabetes)
# save new matrix with z-score values
X_diabetes_norm = scaler.transform(X_diabetes)

# (6) split into training and test sets (25% for testing)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_diabetes_norm, Y_diabetes,
                                                    test_size=0.25, random_state=2021, stratify=Y_diabetes)

# (7) develop a KNN-based model and obtain KNN score (accuracy) for k values from 1 to 8
# range of k neighbors from 1 to 8
neighbors = np.arange(1, 9)
train_accuracy = np.empty(8)
test_accuracy = np.empty(8)

from sklearn.neighbors import KNeighborsClassifier

# for k values 1 to 8
for k in neighbors:
    kNN = KNeighborsClassifier(n_neighbors=k)
    kNN.fit(X_train, Y_train)

    # obtain k-scores
    train_accuracy[k-1] = kNN.score(X_train, Y_train)
    test_accuracy[k-1] = kNN.score(X_test, Y_test)

# (8) plot a graph of train and test score and determine the best value of k
plt.plot(neighbors, test_accuracy, label="Testing Accuracy")
plt.plot(neighbors, train_accuracy, label="Training Accuracy")
plt.xticks(neighbors)
plt.legend()
# set title
plt.title("kNN: Varying Number of Neighbors")
# set axis labels
plt.xlabel("k = Number of Neighbors")
plt.ylabel("Accuracy")
# show graphs
plt.show()

# best value of k is: 3

# (9) display the test score of the model with best value of k
kNN = KNeighborsClassifier(n_neighbors=3)
kNN.fit(X_train, Y_train)
print("Test Score: ", kNN.score(X_test, Y_test))
print()

# print and plot confusion matrix for best value of k
# print
from sklearn.metrics import confusion_matrix
Y_pred = kNN.predict(X_test)
cf = confusion_matrix(Y_test, Y_pred)
print(cf)
# plot
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(kNN, X_test, Y_test, display_labels=["Yes", "No"])
plt.show()

# (10) predict outcome for a person with:
# 6 pregnancies, 140 glucose, 60 bp, 12 skin thickness, 300 insulin, 28 bmi, 0.4 diabetes pedigree, 45 age

x = np.array([[6, 140, 60, 12, 300, 28, 0.4, 45]])
x_norm = scaler.transform(x)
x_final = x_norm.reshape(1, -1)
print("Outcome Prediction: ")
print(kNN.predict(x_final))

# [0] indicates an individual with the metrics listed above is not predicted to have diabetes