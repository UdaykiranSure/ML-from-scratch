import numpy as np

from linear_model import LinearModel
import utils

class GDA(LinearModel):
    '''
    Gausain Discrminant analysis
    Example:
    > cls = GDA()
    > cls.fit(x_train,y_train)
    > cls.predict(x_test)
    '''

    def fit(self,x,y):
        '''Fit given data x and y to the model
            x: Tranining examples inputs
            y: Tranining examples ouputs
        '''
        
        m,_ = x.shape
        print(x.shape)

        phi = np.sum(y)/m
        mu_0 = np.dot(x.T,1-y)/np.sum(1-y)
        mu_1 = np.dot(x.T,y)/np.sum(y)

        y_reshaped = np.reshape(y,(-1,1))

        mu_x = y_reshaped*mu_1 + (1-y_reshaped)*mu_0

        x_centered = x - mu_x

        sigma = np.dot(x_centered.T,x_centered)/m
        sigma_inv = np.linalg.inv(sigma)

        theta = np.dot(sigma_inv,mu_1-mu_0)
        theta_0 = 1/2*mu_0@sigma_inv@ mu_0 - 1 / 2 * mu_1 @ sigma_inv @ mu_1 - np.log((1 - phi) / phi)
        self.theta = np.insert(theta,0,theta_0)

    def predict(self, x):
        
        return utils.add_intercept(x)@self.theta >=0
