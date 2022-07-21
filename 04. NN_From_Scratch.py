# A neural network is a series of matrix operations we perform over a given matrix of data. Today we are going to code a neural network from scratch, which means we 
# have to code all those operations and run them in a specific order in loop to train the network. 

# Prerequisites for this tutorial are - 
# 1. Python - You should be well versed with Python, specially Object Oriented Programming. I will walk through all the code and logic in an algorithmic manner but I
# assume that you have a decent Python backgorund. 
# 2. Math - We are going to use concepts of Probability, Statistics, Calculus, and Linear Algebra. Although we dont have to perform calculations but knowing the concepts 
# of these subjects will help you understand the concept of Neural Networks even if you are a beginner in Python. 
# 3. Machine Learning - I am going to assume a background in Machine Learning concepts like Gradient Descent and Optimization, cuzz we are going to use some of those. 

# First of all, lets import the dependencies. We will use Numpy for our mathematical calculations and Matplotlib to plot the measurements we make. 

import numpy as np
import matplotlib.pyplot as plt

# Lets create a Neural Network Class, which  will serve as a High level function to use a simple network. We will only ask the user for the data and labels of a 
# specific shape. 

class NeuralNetworks(object):
    def __init__(self, data, labels):
        
        # Our Neural Network will have three layers. One input layer, one hidden layer and one output layer. Input layer will have two nodes,
        # hidden layer will have three nodes and output layer will have just one node. Which means the dataset should have two features and based on that,
        # our model has to predict one output. 
 
        self.inputLayerSize = 2
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1

        # z1 would be the input of first layer. and corresponding labels as y1. 
        # z2 would be the input  of second layer which we have already initialized. 
        
        self.z1 = data
        self.y1 = labels
        self.z2 = 0
        self.a2 = 0
        self.z3 = 0
        self.yHat = 0

        self.W1 = np.random.normal(size=(self.inputLayerSize, self.hiddenLayerSize))
        self.W2 = np.random.normal(size=(self.hiddenLayerSize, self.outputLayerSize))

        self.mse = 0

        self.djdw1 = 0
        self.djdw2 = 0

        self.learningRate = 0.01

        self.cost_calc = []

        self.Optimization()

    def forward(self):
        self.z2 = np.array(np.dot(self.z1, self.W1), dtype=np.float32)
        self.a2 = np.array(self.sigmoid(self.z2))
        self.z3 = np.array(np.dot(self.a2, self.W2), dtype=np.float32)
        self.yHat = self.sigmoid(self.z3)
        return self.yHat

    def sigmoid(self, x):
        return 1/ (1 + np.exp(-1 * x))

    def sigmoidPrime(self, x):
        return (np.exp(-x))/ (1 + np.exp(-1 * x)) ** 2

    def loss(self):
        self.mse = sum(self.yHat - self.y1)**2 / 2
        return self.mse

    def costPrime(self, x, d):
        return np.dot(d.T, np.multiply((-1 * (self.y1 - self.yHat)), self.sigmoidPrime(x)))

    def BackProp(self):
        self.djdw2 = self.costPrime(self.z3, self.a2)
        self.djdw1 = self.costPrime(self.z3, self.z1)

    def GradinetDescent(self):
        self.W1 = self.W1 - (self.learningRate * self.djdw1)
        self.W2 = self.W2 - (self.learningRate * self.djdw2)

    def Optimization(self):
        for i in range(10):
            self.forward()
            l = self.loss()
            self.BackProp()
            self.GradinetDescent()
            self.cost_calc.append(l)

data = np.array([[3, 5], [5, 1], [10, 2]], dtype=np.float32)
labels = np.array([[75], [82], [93]], dtype=np.float32)
nn = NeuralNetworks(data=data, labels=labels)
print("Label: {}".format(nn.W1))
print("Prediction: {}".format(nn.W2))
print("Cost: {}".format(nn.mse))

loss = nn.cost_calc
plt.plot(range(len(loss)), loss)
plt.show()
