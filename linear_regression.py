'''
@Author: Guilherme Rossetti Lima
Title: Example of linear regression for Medium article
'''
import os 
import numpy as np


class LinearRegression:

    def __init__(self, loss_function, learning_rate=0.001):
        '''
            Constructor for linear regression model 
            Parameters:
                learning_rate: float  default 0.001
                loss_function: loss_fuction
        '''
        self.learning_rate = learning_rate
        self.omega = 0.0
        self.bias = 0.0

        #Set a loss function for the model 
        self.loss_function = loss_function

    def predict(self, x): 
        '''
            Run an simple linear function based on omega and bias
        '''
        return (x * self.omega) + self.bias

 
    def derivate_params(self, y, y_prime):
        '''
            Partial derivative for parameter omega in the linear function for update
            Parameters: 
                x: list with inputs for the model
                y: list with outputs for the model 
        ''' 
        return np.mean((y_prime - y) * self.omega)

    def derivate_bias(self, y, y_prime):
        '''
            Partial derivative for parameter bias in the linear function for update
            Parameters: 
                x: list with inputs for the model
                y: list with outputs for the model 
        ''' 
        return  np.mean((y_prime -  y))

    def train(self, x, y, ephocs=10):

        for ephoc in range(ephocs): 
            y_prime = self.predict(x)
            
            self.omega -=  self.learning_rate  * self.derivate_params(y, y_prime) #Adjust parameters based on learning rate
            self.bias -=  self.learning_rate * self.derivate_bias(y, y_prime) #Adjust parameters based on the learning rate

            loss = self.loss_function(y, y_prime)
            
            if (ephoc % 100) == 0:
                print(f"Current loss on ephoc {ephoc}: {loss}")




    

