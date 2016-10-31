# -*- coding: utf-8 -*-
"""
This file is the file used to standardize all data.
"""
import numpy as np

def standardize(centered_tX):
  centered_tX[centered_tX==0] = float('nan')
  stdevtrain = np.nanstd(centered_tX, axis=0)
  centered_tX[centered_tX==float('nan')] = 0
  standardized_tX = centered_tX / stdevtrain

  return standardized_tX, stdevtrain

def standardize_original(tX):
  # Removing bothering data and centering
  tX[tX==-999] = 0
  s_mean = np.mean(tX, axis=0)
  centered_tX = tX - s_mean
  stdtX, stdevtrain = standardize(centered_tX)

  return stdtX, stdevtrain, s_mean

def standardize_basis(tX):
  # Resetting all the data
  b_mean = np.mean(tX,axis=0)
  centered_mat = tX - b_mean
  centered_mat[tX==0] = 0
  standardized_tX, stdevtrain = standardize(centered_mat)

  return standardized_tX, stdevtrain, b_mean

def standardize_test_original(tX, training_original_mean, stdevtrain):
  tX[tX==-999] = 0
  centered_testx = tX - training_original_mean
  centered_testx[tX==-999] = 0
  standardized_testx = centered_testx / stdevtrain

  return standardized_testx

def standardized_testx_basis(tX, basis_original_mean, stdev):
  centered_mat = tX - basis_original_mean
  centered_mat[tX==0] = 0
  standardized_testmat = centered_mat / stdev

  return standardized_testmat
