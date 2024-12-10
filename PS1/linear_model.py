import numpy as np

class LinearModel(object):
    "Base class from linear models"
    def __init__(self,step_size = 0.2,max_iter = 100,eps = 1e-5,theta_0 = None,verbose = True):
        """
        Args:
            step_size : learning rate for iterative solvers
            max_iters : Maximum no. of iterations
            eps: Threshold for determining convergence
            Theta_0 : Initial guess for theta, if None use zero
            Verbose: print losss values during training
        """
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.theta_0 = theta_0
        self.verbose = verbose
    def fit(self, x, y):
        """Run sovler
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        """

        raise NotImplementedError('Subclass of LinearModel must implement fit method.')

    def predict(self,x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        raise NotImplementedError('Subclass of LinearModel must implement predict method.')


