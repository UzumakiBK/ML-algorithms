import numpy as np
class linear_regression:
    def fit(self, x_train, y_train):
        self.theta = np.zeros((x_train.shape[1]+1, 1))
        x_new = np.c_[np.ones((x_train.shape[0], 1)), x_train]
        self.theta = np.linalg.pinv(x_new).dot(y_train)
        #self.theta = np.linalg.inv(x_new.T.dot(x_new)).dot(x_new.T).dot(y_train)
        self.intercept = self.theta[0]
        self.features = self.theta[1:]
    
    def predict(self, x_test):
        x_test_new = np.c_[np.ones((x_test.shape[0], 1)), x_test]
        y_pred = x_test_new.dot(self.theta)
        return y_pred