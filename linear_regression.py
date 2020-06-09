import numpy as np
class linear_regression:
    def fit(self, x_train, y_train):
    	'''
    	m -> no of samples in x_train
    	n -> no of features in x_train
    	x_train has shape (m, n)
    	y_train has shape (m, 1)
    	used Normal Equation method to compute the best possible value of the parameters. 
    	'''
        self.theta = np.zeros((x_train.shape[1]+1, 1))
        x_new = np.c_[np.ones((x_train.shape[0], 1)), x_train]
        self.theta = np.linalg.pinv(x_new).dot(y_train) #used pseudoinverse method to handle corner cases of Normal Equation method
        self.intercept = self.theta[0]
        self.features = self.theta[1:]
    
    def predict(self, x_test):
        x_test_new = np.c_[np.ones((x_test.shape[0], 1)), x_test]
        y_pred = x_test_new.dot(self.theta)
        return y_pred