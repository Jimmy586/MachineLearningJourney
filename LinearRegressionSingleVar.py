
import pandas
import matplotlib.pyplot as plt



def gradient_descent(X, y, theta, alpha, num_iters):
  m = len(X)
  for i in range(num_iters):
    h = theta[1]*X + theta[0]#hypothesis function
    #cost = (1/m) * sum([val**2 for val in  (h-y)])
    theta_1 = (1/m)*sum(X*(h-y))
    theta_0 = (1/m)*sum(h-y)
    theta[0] = theta[0] - (alpha*theta_0)
    theta[1] = theta[1] - (alpha*theta_1)
    #print ("theta1 {}, theta0 {},cost {} ,iteration {}".format(theta[1],theta[0],cost,i))
    y_line = theta[0] + theta[1] * X
    plt.scatter(X, y)
    plt.plot(X, y_line, 'r')
    plt.xlabel("Populations of city in 10,000's ")
    plt.ylabel("Profit in 10,000$")
    plt.title("Profit distrubution") 
  return theta


#Load the dataset
data = pandas.read_csv('data.csv')

'''plt.plot(data['population'],data['profit'],'bo',label='Profit regarding the population')
plt.legend()
plt.show()
'''
X = data['population']
y = data['profit']
#number of training samples
#m = y.size
#Initialize theta parameters
theta0 = 0
theta1 =0
#Some gradient descent settings
iterations = 2000
alpha = 0.02
#compute and display initial cost
theta= gradient_descent(X, y, [theta0,theta1], alpha, iterations) #To be completed by students
print(theta)
#Theta = np.polyfit(X, y, 1)
y_line = theta[0] + theta[1] * X
plt.scatter(X, y)
plt.plot(X, y_line, 'b')
plt.xlabel("Populations of city in 10,000's ")
plt.ylabel("Profit in 10,000$")
plt.title("Profit distrubution") 

plt.show()
#Predict values for population sizes of 3.5 and 7.0
#Students write prediction code

