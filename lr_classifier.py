import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten =  test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

def sigmoid(z):
    s = 1 /(1 + np.exp(-z))
    return s
'print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))'

def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim,1), dtype=np.float32)
    b = 0
    return w, b

dim = 2
w, b = initialize_with_zeros(dim)

"""print (train_set_x_orig)
print (train_set_x_orig.shape[0])
print (train_set_x_orig.shape[1])
print (train_set_x_orig.shape[2])
print (train_set_x_orig.shape[3])
print (train_set_y)
print (train_set_y.shape[0])
print (train_set_y.shape[1])"""

def propagate(w, b, X, Y):
    
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1. / m) * np.sum((Y*np.log(A) + (1 - Y)*np.log(1-A)), axis=1)

    dw = (1./m)*np.dot(X,((A-Y).T))
    db = (1./m)*np.sum(A-Y, axis=1)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    gradients = {"dw": dw, "db": db}
    return gradients, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
        gradients, cost = propagate(w=w, b=b, X=X, Y=Y)
        dw = gradients["dw"]
        db = gradients["db"]
        w = w - learning_rate*dw
        b = b -  learning_rate*db
        if i % 100 == 0:
            costs.append(cost) # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w, "b": b}
    
    gradients = {"dw": dw, "db": db}
    
    return params, gradients, costs

w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])

params, gradients, costs = optimize(w, b, X, Y, num_iterations= 5000, learning_rate = 0.1, print_cost = False)

'''print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(gradients["dw"]))
print ("db = " + str(gradients["db"]))
print (costs)'''

def predict(w, b, X):
    z = np.dot(w.T, X) + b
    A = sigmoid(z)
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    for i in range(A.shape[1]):

        if A[0, i] >= 0.5:
            Y_prediction[0, i] = 1     
        else:
            Y_prediction[0, i] = 0
            
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction
 
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, gradients, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    d = {"costs": costs, "Y_prediction_test": Y_prediction_test, "Y_prediction_train" : Y_prediction_train, "w" : w, "learning_rate" : learning_rate, "num_iterations": num_iterations}
    return d 

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

