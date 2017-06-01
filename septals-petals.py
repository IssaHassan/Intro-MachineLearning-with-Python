from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'],iris_dataset['target'],random_state=0
)
##above x_train contains 75% of the datasets and x_test has to contain 25% of the datasets

## use k nearest neighbors classifier, instantiate the class and set the numnber of neighbors
##that will be comparing our input to 1

knn = KNeighborsClassifier(n_neighbors=1)

##we then need our KNeighborsClassifier to hold the data of our training set, to build the model on our training
#set we need to call the 'fit' method of knn

knn.fit(X_train,y_train)
##the fit method returns the knn object, and modifies it, so we get a string of our object and it's paramaters2
##ex. this info is pretty useless to us, since we don't need the other attributes except in special ocasions
"""
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
metric_params=None, n_jobs=1, n_neighbors=1, p=2,
weights='uniform')
"""

X_new = np.array([[5,2.9,1,0.2]])
#print("X_new.shape: {}".format(X_new.shape))
"""
prediction = knn.predict(X_new)
print("Prediction: {}".format(iris_dataset["target_names"][prediction]))
"""

#get mean for the number of times the test set's label was equal to the
#prediction given
#
print("Test set score is {:.2f}".format(knn.score(X_test,y_test)))
