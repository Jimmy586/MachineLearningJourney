
import pandas
import matplotlib.pyplot as plt
import numpy as np


#Gradient descent implementation
def gradient_descent(X, y, theta, alpha, num_iters):
  m = len(X)
  for i in range(num_iters):
    h = theta[1]*X + theta[0]#hypothesis function
    cost = (1/m) * sum([val**2 for val in  (h-y)])
    theta_1 = (1/m)*sum(X*(h-y))
    theta_0 = (1/m)*sum(h-y)
    theta[0] = theta[0] - (alpha*theta_0)
    theta[1] = theta[1] - (alpha*theta_1)
    print ("theta1 {}, theta0 {},cost {} ,iteration {}".format(theta[1],theta[0],cost,i))
  return theta


#Load the dataset
data = pandas.read_csv('data.csv')

X = data['population']
y = data['profit']


#Splitting training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 0)


#Initialize theta parameters
theta0 = 0
theta1 =0
#Some gradient descent settings
iterations = 1500
alpha = 0.02


#USING GRADIENT DESCENT
theta_train= gradient_descent(X_train, y_train, [theta0,theta1], alpha, iterations) #To be completed by students
print(theta_train)

y_pred = theta_train[0] + theta_train[1] * X_train

#plotting the training model
plt.scatter(X_train,y_train, color= 'red')
plt.plot(X_train, y_pred, 'b')
plt.xlabel("Populations of city in 10,000's ")
plt.ylabel("Profit in 10,000$")
plt.title("Profit distrubution TRAINING SET") 
plt.show()

#plotting the testing model
plt.scatter(X_test,y_test, color= 'red')
plt.plot(X_train, y_pred, 'b')
plt.xlabel("Populations of city in 10,000's ")
plt.ylabel("Profit in 10,000$")
plt.title("Profit distrubution TESTING SET") 
plt.show()

#Predict values for population sizes of 3.5 and 7.0
print("predicting profit for a population size 3.5 is {} and 7.0 is {}".format(theta_train[0]+ theta_train[1]*3.5*10000 ,theta_train[0]+ theta_train[1]*7*10000))


#WITH  NUMPY POLYFIT LIB
Theta = np.polyfit(X_train, y_train, 1)
plt.scatter(X_train,y_train, color= 'red')
plt.plot(X_train, Theta[1]+ Theta[0]*X_train, 'b')

plt.scatter(X_test,y_test, color= 'red')
plt.plot(X_train, Theta[1]+ Theta[0]*X_train, 'b')
print("predicting profit for a population size 3.5 is {} and 7.0 is {}".format(Theta[1]+ Theta[0]*3.5*10000 ,Theta[1]+ Theta[0]*7*10000))


#WITH SKLEARN LINEAR MODEL
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
X_train= X_train.values.reshape(-1, 1)
y_train= y_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)
regressor.fit(X_train, y_train)

y_pred = regressor.predict (X_test)
#training
plt.scatter(X_train,y_train, color= 'red')
plt.plot(X_train, regressor.predict (X_train), 'b')
#plotting the testing model
plt.scatter(X_test,y_test, color= 'red')
plt.plot(X_train, regressor.predict (X_train), 'b')