
# coding: utf-8

# In[3]:


import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


# In[4]:


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    # your code here
    sigmoid_func = 1/(1+ np.exp(-z))
    return sigmoid_func


# In[5]:


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary
    
    train_data = np.zeros(shape=(0,784))
#     print(train_data)
    test_data = np.zeros(shape=(0,784))
    validation_data = np.zeros(shape=(0,784))
    train_label = np.zeros(shape=(0,1))
    test_label = np.zeros(shape=(0,1))
    validation_label = np.zeros(shape=(0,1))
    
    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    for key in range(10):
#         print(key)
        total_training_data = mat.get('train'+str(key))
        total_testing_data = mat.get('test'+str(key))
#         print(len(total_training_data))
        
        training_count = len(total_training_data)
        testing_count = len(total_testing_data)
        
        exact_training_count = training_count - 1000
        validation_count = training_count - exact_training_count
#         print(validation_count)
        
        for training_label in range(exact_training_count):
            train_label = np.append(train_label,key)
        for testing_label in range(testing_count):
            test_label = np.append(test_label,key)
        for validating_label in range(validation_count):
            validation_label = np.append(validation_label,key)
        
#         print(len(train_data_label))
#         print(len(validation_data_label))
        
        index = range(total_training_data.shape[0])
#         print(index)
        random_indices=np.random.permutation(index)
#         print(permute)
        
        random_train = total_training_data[random_indices[0:exact_training_count],:]
        random_validation = total_training_data[random_indices[exact_training_count:],:]
        
        train_data = np.vstack([train_data,random_train])
        validation_data = np.vstack([validation_data,random_validation])
        test_data = np.vstack([test_data,total_testing_data])
        
#         print(len(test_data))
        #Normalize Data
    train_data = train_data/255.0
    validation_data = validation_data/255.0
    test_data = test_data/255.0

#         print(train_data[0])
    # Feature selection
    # Your code here.
#         print(type(train_data))
    columns = np.concatenate((train_data,validation_data,test_data), axis = 0)
#     print(len(columns))
#     zero_indices = np.argwhere(np.all(columns[0, :] == 0, axis=0))
#         print(zero_indices)
    list_indices_deletion = np.all(columns == columns[0,:], axis=0)
    del_indices = []
    for i in range(len(list_indices_deletion.tolist())):
        if(list_indices_deletion[i] == True):
            del_indices.append(i)
    train_data = np.delete(train_data, del_indices, axis = 1)
    test_data = np.delete(test_data, del_indices, axis = 1)
    validation_data = np.delete(validation_data, del_indices, axis = 1)
    print("LEngth-->",len(train_data[0]))
#     print(len(train_data))
#     print(len(test_data))
#     print(len(validation_data))

    print('preprocess done')
#     return 0
    return train_data, train_label, validation_data, validation_label, test_data, test_label


# In[6]:


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    #
    #
    #
    #
    #
#     print(training_data.shape)
    '''Forward Pass starts'''
    # Adding a bias term
    ones_array = np.array(np.ones(training_data.shape[0]))
#     print(ones_array)
    training_data = np.append(training_data, np.ones((training_data.shape[0], 1)), 1)
#     print(training_data)
    
    aj = np.dot(training_data,w1.T)
    zj = sigmoid(aj)
    
    zj_bias_term = np.append(zj, np.ones((zj.shape[0],1)),1)
    bl = np.dot(zj_bias_term,w2.T)
    ol = sigmoid(bl)
    
#     print(ol)
    true_label = np.zeros((training_data.shape[0],n_class))
    
    i=0
    for i in range(training_label.shape[0]):
        pos = int(training_label[i])
        true_label[i][pos] = 1

    
#     print(true_label)
    
    
    '''Calculate Error function'''
    error = 0.0
    err_first_part = np.multiply(true_label, np.log(ol))
    err_second_part = np.multiply(np.subtract(1,true_label),np.log(np.subtract(1,ol)))
    err = np.add(err_first_part,err_second_part)
    error = error + np.sum(err)
    error = ((-1)*error)/training_data.shape[0]
#     print("Error---->",error)
    
    
    reg_first_part = np.add(np.sum(np.square(w1)),np.sum(np.square(w2)))
    reg_second_part = np.divide(lambdaval, (2*training_data.shape[0]))
    reg_parameter = reg_first_part + reg_second_part
    
    error_reg = error + reg_parameter
    
    w2_gradient_without_regularized = np.dot(np.subtract(ol,true_label).T, zj_bias_term)
    w2_gradient_regularized = np.add(w2_gradient_without_regularized, np.multiply(lambdaval,w2))
    grad_w2 = np.divide(w2_gradient_regularized, training_data.shape[0])
    
    w2_no_bias = w2[:,0:w2.shape[1]-1]
    
    w1_gradient_without_regularized = np.multiply(np.multiply(np.subtract(1,zj),zj), np.dot(np.subtract(ol,true_label),w2_no_bias))
    w1_gradient_regularized = np.add(np.dot(w1_gradient_without_regularized.T,training_data), np.multiply(lambdaval,w1))
    grad_w1 = np.divide(w1_gradient_regularized, training_data.shape[0])
    obj_val = error
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
#     obj_grad = np.array([])

    return (obj_val, obj_grad)


# In[11]:


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    data = np.insert(data, data.shape[1], 1, axis=1)
    hid_layer_output = sigmoid(np.dot(data, w1.T))
    hid_layer_output = np.insert(hid_layer_output, hid_layer_output.shape[1], 1, axis=1)
    output = sigmoid(np.dot(hid_layer_output, w2.T))
    labels = np.double(np.argmax(output, axis=1))
    
    return labels


# In[13]:


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

 #Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 5

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 100}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# # # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# # # and nnObjGradient. Check documentation for this function before you proceed.
# # # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# # # Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# # Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# # find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# # find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# # find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

