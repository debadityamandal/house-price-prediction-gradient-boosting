import numpy as np
import pandas as pd
def handle_non_numerical_data(df):
    columns=df.columns.values
    for column in columns:
        text_digit_vals={}
        def convert_to_int(val):
            return text_digit_vals[val]
        if(df[column].dtype!=np.int64 and df[column].dtype!=np.float64):
            column_contents=df[column].values.tolist()
            unique_elements=set(column_contents)
            x=0
            for unique in unique_elements:
                if(unique not in text_digit_vals):
                    text_digit_vals[unique]=x
                    x+=1
            df[column]=list(map(convert_to_int,df[column]))
    return df
def featureNormalize(X):
    """
    Make all features have mean 0 and std 1. Use pandas!
    """
    # Goodness, this is easy with pandas.
    X_norm = (X - X.mean())/X.std()
    mu = X.mean()
    sigma = X.std()
    return X_norm, mu, sigma
def predictValues(X, theta):
    """
    With X and theta as numpy arrays, finding y for all m samples is just
    a dot product.
    """
    predictedValues = np.dot(X, theta)
    return predictedValues
def computeCost(X, y, theta):
    """
    Compute the cost function with numpy arrays as imput.
    """
    m = len(y)
    predictedValues = predictValues(X, theta)
    sumOfSquareErrors = np.square(predictedValues-y).sum()
    cost = sumOfSquareErrors / (2*m)
    return cost

def gradientDescent(X, y, theta, alpha, num_iters, verbose=False):
    """
    Takes numpy arrays and does the gradient descent.
    """
    m = len(y)

##    if verbose:
##    print('theta input ', theta)
    print('initial cost %e' % computeCost(X, y, theta))

    for i in range(num_iters):
        predictedValues = predictValues(X, theta)
        theta = theta - alpha / m * np.dot((predictedValues - y), X)

    cost = computeCost(X, y, theta)

##    if verbose:
##    print('    %04i theta' % i, theta)
    print('    %04i cost %e' % (i, cost))
    return theta
def normalEqn(X, y):
    """
    Find the closed form solution using numpy matrix methods.
    """
    xtx = np.dot(X.transpose(), X)
    pinv = np.linalg.pinv(xtx)
    theta = np.dot(pinv, np.dot(X.transpose(), y))
    theta = theta.flatten()

    return theta
if __name__=="__main__":
    train_data=pd.read_csv("train.csv")
    train_data=train_data.drop(["Id"],1)
    train_data=handle_non_numerical_data(train_data)
    m=train_data.shape[0]
    n=train_data.shape[1]
    X_train=train_data.iloc[:,0:n-1]
    X_train,mu,sigma=featureNormalize(X_train)    
    ones=np.ones([X_train.shape[0],1])
    X_train=np.concatenate((ones,X_train),axis=1)
    X_train=np.array(X_train)
    Y_train=train_data.iloc[:,n-1:n].values
    Y_train=np.array(Y_train).flatten()
    alpha=0.001
    num_iters=1000
    theta=np.zeros(X_train.shape[1])
    theta = gradientDescent(np.nan_to_num(X_train), Y_train, theta,
                                                   alpha, num_iters,
                                                   verbose=False)
    # Analysis the price of all test data
    test_data=pd.read_csv("test.csv")
    test_data=handle_non_numerical_data(test_data)
    X_test=test_data.iloc[:,0:n-1]
    ones=np.ones([X_test.shape[0],1])
    X_test=np.concatenate((ones,X_test),axis=1)
    X_test=np.array(X_test)
    price_test=predictValues(np.nan_to_num(X_test),theta)
    print(price_test)
