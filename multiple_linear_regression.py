'''
Created on Thurs Jan 04 21:50:42 2024

@Author: Guilherme Rossetti Lima
'''
import os 
import numpy as np


class MLinearRegression:

    def __init__(self, loss_function, accuracy_function, size, learning_rate=0.1):
        '''
            Constructor for linear regression model 
            Parameters:
                learning_rate: float  default 0.001
                loss_function: loss_fuction
        '''
        self.learning_rate = learning_rate
        
        #Modified theta to contain a list of parameters
        self.theta = np.random.rand(size)
        self.bias = 1

        #Set a loss function for the model
        self.loss_function = loss_function
        self.accuracy_function = accuracy_function

    #Implementatio for hypotesis
    def predict(self, x):
        '''
            Run an simple linear function based on theta and bias
        '''
        #Multuply theta by its corresponding feature and sum all together 
        #Adds the bias(beta) at the end
        return  np.sum((self.theta * x)+self.bias, axis=1)
        

    #Partial derivative for loss with respect to theta
    def derivate_params(self, y, y_prime, x):
        '''
            Partial derivative for parameter theta in the linear function for update
            Parameters: 
                x: list with inputs for the model
                y: list with outputs for the model 
        ''' 
        return np.sum(((y_prime -y) * x.T), axis=1) / len(y_prime)
        

    #Partial derivative for loss with respect to bias(beta)
    def derivate_bias(self, y, y_prime):
        '''
            Partial derivative for parameter bias in the linear function for update
            Parameters: 
                x: list with inputs for the model
                y: list with outputs for the model 
        ''' 
        return  np.sum(y_prime - y) / len(y_prime)

    def train(self, x, y, x_validation=None, y_validation=None, ephocs=30):
        best_loss = 0
        best_theta = []
        best_bias = 0

        # prepate the dictionary for the history
        history = {}
        history['training'] = {}
        history['training']['loss'] = []
        history['training']['accuracy'] = []
        history['validation'] = {}
        history['validation']['loss'] = []
        history['validation']['accuracy'] = []

        for ephoc in range(ephocs): 
            y_prime = self.predict(x)

            self.bias  -=  self.learning_rate  * self.derivate_bias(y, y_prime) #Adjust parameters based on learning rate
            self.theta -=  self.learning_rate  * self.derivate_params(y, y_prime, x) #Adjust parameters based on learning rate
            
            #Calculate and keep track of validation
            loss = self.loss_function(y, y_prime) #Predict again post update to get always most updates y_prime 
            history['training']['loss'].append(loss)

            #Calculate and keep track of accuracy 
            accuracy = self.accuracy_function(y, y_prime) #Predict again post update to get always most updates y_prime 
            history['training']['accuracy'].append(accuracy)

            if (x_validation is not None and x_validation is not None):
                y_prime_validation = self.predict(x_validation)
                #Calculate and keep track of validation
                validation_loss = self.loss_function(y_validation, y_prime_validation)
                history['validation']['loss'].append(validation_loss)

                #Calculate and keep track of accuracy 
                validation_accuracy = self.accuracy_function(y_validation, y_prime_validation)
                history['validation']['accuracy'].append(validation_accuracy)

            #Keep best solution found so far
            if best_loss == 0 or loss < best_loss:
                best_loss = loss
                best_theta = self.theta 
                best_bias = self.bias 

            if (ephoc % 10) == 0:
                print(f"Current loss on ephoc {ephoc}: {loss}")
        
        #Set best parameters found during optmization to the params
        self.theta = best_theta
        self.bias = best_bias

        return history

    

