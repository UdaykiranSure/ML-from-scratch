import numpy as np
from utils import load_dataset

from linear_model import LinearModel



def main(train_path,eval_path,pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train,y_train = load_dataset(train_path,add_intercept = True)

class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self,x,y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        def h(x,theta):
            '''Vectorized implementation of h_theta(x) = 1 / (1 + exp(-theta^T x)).
            '''
            return 1/(1+np.exp(-np.dot(x,theta)))

        def gradient(theta,x,y):
            """Vectorized implementation of the gradient of J(theta).

            :param theta: Shape (n,).
            :param x:     All training examples of shape (m, n).
            :param y:     All labels of shape (m,).
            :return:      The gradient of shape (n,).
            """
            m,_ = x.shape
            return -1/m*np.dot(x.T,(y-h(x,theta) ))

        def hessian(theta,x):
            """Vectorized implementation of the Hessian of J(theta).

            :param theta: Shape (n,).
            :param x:     All training examples of shape (m, n).
            :return:      The Hessian of shape (n, n).
            """
            m,_ = x.shape
            h_theta_x  = np.reshape(h(x,theta),(-1,1))
            return 1 / m * np.dot(x.T, h_theta_x * (1 - h_theta_x) * x)

        def next_theta(theta,x,y):
            """The next theta updata by newtons method.
            Args:
                :param theta: Shape (n,).
                :return:      The updated theta of shape (n,).
            """
            return theta - np.dot(np.linalg.inv(hessian(theta,x)),gradient(theta,x,y))

        m,n = x.shape
        if self.theta_0 is None:
            self.theta_0 = np.zeros(n)

        old_theta = self.theta_0
        new_theta = next_theta(old_theta,x,y)
        while np.linalg.norm(new_theta - old_theta,1) >= self.eps:
            old_theta = new_theta
            new_theta = next_theta(old_theta,x,y)
        self.theta = new_theta




    def predict(self,x):
        """Return prediction of inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        return x@self.theta >= 0
