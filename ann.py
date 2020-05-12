# Artificial Neural Networks

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values #roll no,customer id,surname doesnt have any impact on the dependent variable so we wont include them in the matrix,rest all variables are dependent on the exitted
y = dataset.iloc[:, 13].values  #so 3-12 indexes are for independent variables and 13th index is the exitted one-the dependent variable

# Encoding categorical data-we have the categorical variables so have to take care of them first(encode first)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()  #so we have 2 independent categorical variables which are country,gender
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) #So france,germany,spain-0,1,2 values accordingly
labelencoder_X_2 = LabelEncoder() #For gender variable create a new object now and have to care of the indexes also
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) #So index 2
onehotencoder = OneHotEncoder(categorical_features = [1]) #So now female 0 and male 1
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]  #create dummy varibles but shud not get into the dummy variable trap(so take all the columns and just remove the first column-so now 0 and from 1 to end)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#So test set size is 20% and training set size is 80% so 8000 obs here and 2000 obs in test set

# Feature Scaling-there will be high computations and paralle computations so we need to apply feature scaling for neural networks-just to ease the calculations
#So feature scaling is compulsory for the neural networks

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 

# Part 2 - Now let's make the ANN!
#This is one of the most powerful tool in machine learning-the ANN

# Importing the Keras libraries and packages
import keras  #import keras library and the required packages-it will build the backend using tensorflow
from keras.models import Sequential #so we import 2 modules here-for the model and for the layer
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential() #This neural network model will be a classifier,so classifier object is a sequential class object
#There are many activation functions for eh.the sigmoid function,rectifier function etc.The best one and the one which we will use the most is rectifier function
#So we use rectifier function for hidden layer and sigmoid function for the output layer

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
#So the no of units in the hidden layer-here we take it as the average of input and output layers so now its 11+1/2 so 6 units in the first hidden layer
#uniform-so it initialises the nodes to small value close to 0,activation function so in the hidden layer its relu-rectifier function,last argument is compulsory-no of units/nodes in the input layer which is 11(all the attributes of the customer)

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
#This is a neural network so its better to create more hidden layers-so here the same method to create the hidden layer
#So now input layer,then 2 hidden layers and one output layer in this network

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')) #here the activation function is the sigmoid function(only for the output layer)-we want a binary outcome (either 0 or 1)so thats why its better to use the sigmoid function
#So till here we are done with creating the layers of our ANN

# Compiling the ANN-basically it means applying stochastic gradient descent to our neural network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Optimizer is for the optimal weights for the neural network,There are many types of stochastic gradient descent but the best type of SGD is Adam algorithm,loss function is within the adam algorithm
#Loss function is kind of same for the linear regression-here its the sum of squared errors,if the dependent variable has a binary outcome then its called binary,if the dependent variable has more than 2 outcomes(3 categories)then categorical cross_entropy
#So now the loss function will be binary_crossentropy,metrics-the criterion we choose for our mode-so we chose accuracy for our model(as always),the accuracy tries to improve the model's performance


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100) #nb_epoch-no of times we are training our ANN on the whole training set,batch_size-no of obs after which you want to update the weights
#The accuracy keeps on increasing after every epoch-in the training ste the obtained accuracy was 86%, so next we check the accuracy on the test set
#epoch-when you go through the dataset again and again and again throuhg a no of iterations

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)  #If y_pred is greater than 0.5 then it return true or else false(sigmoid function)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) #here we got 1720 correct predictions out of 2000 which means that our accuracy is 86%
