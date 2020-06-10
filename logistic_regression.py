import numpy as np

class logistic_regression:
    def __init__(self, learning_rate=0.01, epochs=100, fit_intercept=True, verbose=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        
        
    def __add_intercept(self, x):
        ones = np.ones((x.shape[0], 1))
        return np.concatenate((ones, x), axis=1)
   
    def __softmax(self, z):
        return 1/(1 + np.exp(-z))
   
    def predict(self, x):
        h = x.dot(self.theta)
        y_pred = self.__softmax(h)
        if y_pred >= 0.5:
            return 1
        else:
            return 0
        
    def __compute_cost(self, x, y):
        z = self.__softmax(x.dot(self.theta))
        cost = y.dot(np.log(z)) + (1 - y).dot(np.log(1 - z))
        return -(cost/x.shape[0])
    
    def __compute_gradient(self, x, y):
        z = self.__softmax(x.dot(self.theta))
        gradient = x.T.dot(z - y)
        return gradient/x.shape[0]
    
    def fit(self, x, y):
        if fit_intercept == True:
            x = self.__add_intercept(x)
        self.theta = np.zeros((x.shape[1], 1))
        for epoch in range(self.epochs):
            gradient = self.__compute_gradient(x, y)
            self.theta = self.theta - (self.learning_rate * gradient)
            if self.verbose == True and epoch % 10 == 0:
                print('Epoch {} => Loss = {}'.format(str(epoch), str(self.__compute_cost(x, y))))