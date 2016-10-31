# -*- coding: utf-8 -*-
"""
This is our solution, the one we used for the last submission.
BE CAREFUL, we optimized it as well as we could for memory issues,
but this code should be run with a 16GB RAM machine. It was decided
that it would take too much time optimizing for a computer with 8GB
RAM rather than focusing on getting a better result (See the report for explanations).

The code is made to be functional, and easy to modulate and test
different models, see the implementations, build_polynomial and
standardization files.
"""
import numpy as np
from implementations import *
from build_polynomial import *
from standardization import *
from proj1_helpers import *

# Greeter
print("This is our solution, the one we used for the last submission.\nBE CAREFUL, we optimized it as well as we could for memory issues, but this code should be run with a 16GB RAM machine. It was decided that it would take too much time optimizing for a computer with 8GB RAM rather than focusing on getting a better result (See the report for explanations).\n\nThe code is made to be functional, and easy to modulate and test different models, see the implementations, build_polynomial and standardization files.\n=====================================================================\n\n")

print("Loading the train data")
y, tX, ids = load_csv_data('train.csv')


print("Sanitizing and standardizing the data")
# Removing bothering data and centering
standardized_tX, stdevtrain, m = standardize_original(tX)


print("Building polynomial basis")
"""
You can choose your own polynomial basis. Checkout the build_polynomial
file.
"""
mat = build_poly_combinations_st_all(standardized_tX)



print("Standardizing again")
standardized_mat, stdev, m2 = standardize_basis(mat)
tx = np.c_[np.ones(standardized_mat.shape[0]), standardized_mat]


print("Training...")
"""
You can choose your own model, and load your own parameters, checkout the
implementations file
"""
# Cross validating
x1=np.delete(tx,range(200000,250000),axis=0)
x2=np.delete(tx,range(150000,200000),axis=0)
x3=np.delete(tx,range(100000,150000),axis=0)
x4=np.delete(tx,range(50000,100000),axis=0)
x5=np.delete(tx,range(0,50000),axis=0)
y1=np.delete(y,range(200000,250000),axis=0)
y2=np.delete(y,range(150000,200000),axis=0)
y3=np.delete(y,range(100000,150000),axis=0)
y4=np.delete(y,range(50000,100000),axis=0)
y5=np.delete(y,range(0,50000),axis=0)

# Best found parameters for reg_logistic_regression
lambda_ = 10**-8
gamma = 10**5
max_iters = 25
initial_w = np.zeros((len(x1[0])))
reg_logistic_w1,loss = reg_logistic_regression(y1,x1,lambda_,initial_w,max_iters,gamma)
reg_logistic_w2,loss = reg_logistic_regression(y2,x2,lambda_,initial_w,max_iters,gamma)
reg_logistic_w3,loss = reg_logistic_regression(y3,x3,lambda_,initial_w,max_iters,gamma)
reg_logistic_w4,loss = reg_logistic_regression(y4,x4,lambda_,initial_w,max_iters,gamma)
reg_logistic_w5,loss = reg_logistic_regression(y5,x5,lambda_,initial_w,max_iters,gamma)
logistic_weights = (reg_logistic_w1+reg_logistic_w2+reg_logistic_w3+reg_logistic_w4+reg_logistic_w5)/5

print("Loading the testing data")
_, testx, ids_test = load_csv_data('test.csv')


print("Applying the same standardization to the testing set")
standardized_testx = standardize_test_original(testx, m, stdevtrain)


print("Building polynomial basis")
"""
Don't forget to use the same polynomial basis as before.
"""
mat = build_poly_combinations_st_all(standardized_testx)


print("Standardizing again")
standardized_testmat = standardized_testx_basis(mat, m2, stdev)
tX_test = np.c_[np.ones(standardized_testmat.shape[0]), standardized_testmat]


y_pred = predict_labels(logistic_weights, tX_test)
create_csv_submission(ids_test, y_pred, 'predictions.csv')
print("The prediction has been stored in the predictions.csv file")
