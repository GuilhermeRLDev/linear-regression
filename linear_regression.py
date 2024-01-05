'''
Created on Thurs Jan 04 21:50:42 2024

@Author: Guilherme Rossetti Lima
'''
import os 
import numpy as np
import matplotlib.pylab as plt

class LinearRegression:

    def __init__(self, loss_function, accuracy_function, learning_rate=0.1):
        '''
            Constructor for linear regression model 
            Parameters:
                learning_rate: float  default 0.001
                loss_function: loss_fuction
        '''
        self.learning_rate = learning_rate
        self.theta = 0.0
        self.beta = 0.0

        #Set a loss function for the model 
        self.loss_function = loss_function
        self.accuracy_funciton = accuracy_function

    def plot_line(self, i, x, y):
        a = np.arange(0, 1, 0.001)
        y_hat =  self.predict(a)
        
        plt.figure(i)

        plt.title(f"Image {i}")
        #scatter
        plt.scatter(x, y)
        #plot line
        plt.plot(a, y_hat)

        plt.show()

    def predict(self, x): 
        '''
            Run an simple linear function based on omega and bias
        '''
        return (x * self.theta) + self.beta

 
    def derivate_params(self, y, y_prime,  x):
        '''
            Partial derivative for parameter omega in the linear function for update
            Parameters: 
                x: list with inputs for the model
                y: list with outputs for the model 
        ''' 
        return np.mean((y_prime - y) * x)

    def derivate_bias(self, y, y_prime):
        '''
            Partial derivative for parameter bias in the linear function for update
            Parameters: 
                x: list with inputs for the model
                y: list with outputs for the model 
        ''' 
        return  np.mean((y_prime -  y))

    def train(self, x, y, verbose=False, epochs=10):
        
        best_loss = -1
        best_theta = 0
        best_beta = 0
        
        losses = []
        accuracies = []

        for ephoc in range(epochs):
            y_prime = self.predict(x)
            
            self.theta -=  self.learning_rate  * self.derivate_params(y, y_prime, x) #Adjust parameters based on learning rate
            self.beta -=  self.learning_rate * self.derivate_bias(y, y_prime) #Adjust parameters based on the learning rate

            loss = self.loss_function(y, y_prime)
            
            #Keep the best set of parameters found 
            if best_loss == -1 or loss < best_loss: 
                best_loss = loss 
                best_theta = self.theta
                best_beta = self.beta

            if verbose and (ephoc % 30 == 0):
                self.plot_line(ephoc+1, x, y)

            #Save loss and accuracy to plot graph
            losses.append(loss)
            accuracies.append(self.accuracy_funciton(y, y_prime))
            
        x_axis = np.arange(epochs)

        #Set best values to model 
        self.theta = best_theta
        self.beta = best_beta
        
        #Plot graph with loss 
        if verbose:
            plt.show()
            plt.figure()
            plt.title("Loss per ephoc")
            plt.plot(x_axis, losses)
            plt.show()

        

    

