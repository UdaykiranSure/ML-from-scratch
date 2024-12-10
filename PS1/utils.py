import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def add_intercept(x):
    """Add intercept to matrix x.
    Args:
        x: 2D Numpy array.
    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    
    new_x = np.zeros((x.shape[0],x.shape[1]+1), dtype =x.dtype)
    new_x[:,0] = 1
    new_x[:,1:] = x

    return new_x


def plot(x,y,theta= None,theta2 = None,save_path = None,legend1= None,legend2 = None,title = None,correction = 1.0):
    """Plot dataset and fitted logistic regression parameters.
    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
        correction: Correction factor to apply, if any.
    """
    # Plot dataset
    plt.figure()
    plt.plot(x[y ==1,-  2], x[y == 1, -1], 'bx', linewidth = 2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth = 2)

    '''

    '''

    # Plot decison boundary (found by solving for thete^T x = 0)
    margin1 = (max(x[:, -2]) - min(x[:, -2])) * 0.2
    margin2  = (max(x[:, -1]) - min(x[:, -1])) * 0.2
    x1 = np.arange(min(x[:, -2]) - margin1, max(x[:, -2]) + margin1, 0.01)
    x2 = -(theta[0] / theta[2] + theta[1] / theta[2] * x1)
    plt.plot(x1, x2, c='red',label = legend1, linewidth = 2)
    plt.xlim([min(x[:, -2]) - margin1, max(x[:, -2]) + margin1])
    plt.ylim([min(x[:, -1]) - margin2, max(x[:, -1]) + margin2])

    if theta2 is not None:
        margin1 = (max(x[:, -2]) - min(x[:, -2])) * 0.2
        margin2  = (max(x[:, -1]) - min(x[:, -1])) * 0.2
        x1 = np.arange(min(x[:, -2]) - margin1, max(x[:, -2]) + margin1, 0.01)
        x2 = -(theta[0] / theta[2] + theta[1] / theta[2] * x1)
        plt.plot(x1, x2, c='black',label = legend2, linewidth = 2)
        plt.xlim([min(x[:, -2]) - margin1, max(x[:, -2]) + margin1])
        plt.ylim([min(x[:, -1]) - margin2, max(x[:, -1]) + margin2])
    

    plt.xlabel('x1')
    plt.ylabel('x2')
    if legend1 is not None or legend2 is not None:
        plt.legend(loc="upper left")
    if title is not None:
        plt.suptitle(title, fontsize=12)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def load_dataset(csv_path,label_col = 'y', add_intercept = False):
    """Load dataset from csv
    Args:
        csv_path : path to CSV file containing dataset
        label_col: Name of column to use as labels (should be 'y' or 't')
        add_intercept: Add an intercept entry to x

    Returns:
        xs: Numpy array of x-values (inputs)
        ys: Numpy array of y-values (labels)
    """
    def add_intercept_fn(x):
        global add_intercept
        return add_intercept(x)

    #validate label col argument
    allowed_label_cols = ('y','t')
    if label_col not in allowed_label_cols:
        raise ValueError('Invalid label_col: {} (expected {})'.format(label_col,allowed_label_cols))

    df = pd.read_csv(csv_path)
    labels = df[label_col].to_numpy()
    df.drop(label_col,axis = 1,inplace = True)
    inputs = df.to_numpy()

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels


def download_dataset(url, filename):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        with open(filename, 'wb') as f:

            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Dataset downloaded successfully as '{filename}'")
        return filename  # Return the filename for further use
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        return None

