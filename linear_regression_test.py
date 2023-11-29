import os
from linear_regression import LinearRegression
import numpy as np 

def get_value(value):
     return 1 if value > 0.5 else 0

def load_dataset():
     x_train = np.random.normal(0, 1, 1000)
     y_train = x_train > 0.5
     y_train = [get_value(v) for v in y_train]

     x_test = np.random.normal(0, 1, 250)
     y_test = x_test > 0.5
     y_test = [get_value(v) for v in y_test]
    
     return x_train, y_train, x_test, y_test 

def mean_squared_loss(y, y_prime):
     return np.mean((y_prime - y)**2) 

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_dataset()
    model = LinearRegression(loss_function=mean_squared_loss)
    model.train(x_train, y_train, 10000)





