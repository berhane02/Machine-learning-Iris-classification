#!/usr/bin/python

from maxheap import Maxheap
import numpy as np
import sys

"""
Task 3: record the results of your experiments here.

Changing random seed:
    3521: 0.96
    3000: 0.88
    300: 0.888
    1000: 0.96
    10000: 0.92
    the result increase and decrease the seed value doesn't give constant result, accuracy are randomninized
    when you increase or decrease the seed value. 
Changing distance metric:
    L2: 0.96
    L3: 0.40
    L4: 0.76
    when i increase the distance metric test accuracy decrease and if the (LX) x is odd number
    the accuracy even decrease more. 
"""


class Perceptron:
    """ Implementation of the perceptron classification algorithm """

    def __init__(self, X, Y, params):
        np.random.seed(3521)
        self.W = np.random.rand(len(X[0]))
        self.Bias = 0
        self.N = params[0] #Number of Iterations
        self.train(X,Y)

    def train(self, X, Y):
        
        #print('check',self.W+(self.predict_batch(X)-Y).dot(X))
        #print('biase', self.Bias)
        A = [1]*len(X)
        #print("A", A)
        for i in range(self.N):
            Yhat = self.predict_batch(X)
            self.Bias = (Y-Yhat).dot(A)+self.Bias
            self.W = self.W +(Y-Yhat).dot(X) #
            """
            Train the Perceptron given the training data using Gradient Descent.
        
            Step 1: Predict Yhat for training inputs
            Step 2: Update Each Parameter (including bias) as per the Gradient:
                    wi = wi + (Y - Yhat) . Xi
            Step 3: Repeat above steps N times
        
            Tips:
            A. It is faster to use a matrix operation to compute the gradient
            B. Inputs and Parameters are provided as 'Numpy' Arrays.
            C. Numpy provides a dot product operation, and other matrix operations.
            D. You can print the parameters to see what's going on: self.W, self.Bias, etc.
            """
        

    def predict(self, input_x):
        """ Predict the output of a single input """
        weighted_sum = input_x.dot(self.W) + self.Bias
        return (weighted_sum > 0).astype(int)

    def predict_batch(self, X_test):
        """ Predict the outputs of a set of inputs """
        return [self.predict(point) for point in X_test]

class KNN:
    """ Implementation of the kNN classification algorithm """

    def __init__(self, X, Y, params):
        self.X=X
        self.Y=Y
        self.k=params[0]
        self.train(X,Y)

    def train(self,X,Y):
        """ No training step needed. """
        pass

    def predict(self, input_x):
        
        myHeap = Maxheap(self.Y[:self.k], [self.distance(x,input_x) for x in self.X[:self.k]])
        for i in range(self.k,len(self.X)):
            arrayX = self.X[i]
            dis = self.distance(arrayX,input_x)
            if(dis<myHeap.peekMaxValue):
                myHeap.pop()
                myHeap.push(self.Y[i],dis)
            #####
        result = myHeap.counts()
        toRurn = 0
        if(len(result)==1):
            toRurn = result[0][0]
        else:
           zero = result[0]
           one = result[1]
           if(zero>one):
               toRurn = 0
           else:
               toRurn = 1
        return toRurn;
       
    def predict_batch(self, X_test):
        """
        Predict labels using k-nearest neigbors for a batch.
        X_test : batch of inputs to predict
        """
        return [self.predict(point) for point in X_test]

    def distance(self,x1,x2):
        """ Calculate the distance between two points x1 and x2 """
        return sum((x1 - x2) ** 2) # L2 distance
 
