import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import DeepLearning
import SpecialFunction
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# split Train and Test dataset  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Building a model
classifier = DeepLearning.NeuralNetwork()
classifier.add(DeepLearning.Layer(neural_network=classifier,output_dim=20,activation ='sigmoid',initializer = 'uniform',input_dim=11,random_state = 1))
classifier.add(DeepLearning.Layer(neural_network=classifier,output_dim=20,activation ='sigmoid',initializer = 'uniform',random_state = 3))
classifier.add(DeepLearning.Layer(neural_network=classifier,output_dim=3,activation ='sigmoid',initializer = 'uniform',random_state = 3))

# Train the model
classifier.fit(X_train = X_train, y_train = y_train,epoch =15,learning_rate =0.01)

# Test the model
y_pred = classifier.predict(X_test)

# Analize the result
SpecialFunction.confusion_matrix(y_test =y_test,y_pred= y_pred)
print(SpecialFunction.binary_confusion_matrix(y_test =y_test,y_pred= y_pred))
SpecialFunction.accuracy_map(neural_network = classifier,costom = 'b-',lw=1.0)

