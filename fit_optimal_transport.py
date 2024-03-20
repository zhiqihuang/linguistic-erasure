import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import cosine sim
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
from sklearn.linear_model import SGDClassifier
# import mlp
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

import pickle
import argparse

import torch
import scipy

import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.covariance import ShrunkCovariance
from sklearn.covariance import oas
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import scipy

def get_optimal_gaussian_transport_func(source_x, target_x, diag_cov=False, use_shrinkage=False):
      # cov

      if use_shrinkage:
        cov_source = ShrunkCovariance().fit(source_x).covariance_.real + 1e-8
        cov_target = ShrunkCovariance().fit(target_x).covariance_.real + 1e-8
      else:
        cov_source = np.cov(source_x.T).real + 1e-8
        cov_target = np.cov(target_x.T).real + 1e-8
      if diag_cov:
        cov_source = np.diag(np.diag(cov_source))
        cov_target = np.diag(np.diag(cov_target))


      # optimal transport

      cov_source_sqrt = matrix_squared_root(cov_source)
      cov_source_sqrt_inv = matrix_inv_squared_root(cov_source) #scipy.linalg.inv(cov_source_sqrt)

      A = cov_source_sqrt_inv @ matrix_squared_root(cov_source_sqrt @ cov_target @ cov_source_sqrt) @ cov_source_sqrt_inv
      #A = A.real
      return A

def matrix_squared_root(A):
    return scipy.linalg.sqrtm(A)


def matrix_inv_squared_root(A):

    return np.linalg.inv(matrix_squared_root(A))



def optimal_transport(x_train, z_train, x_dev, dev_z=None, source=0, target=1):
  """
  x_train: np.array, shape (n_train, n_features)
  z_train: np.array, shape (n_train,)
  x_dev: np.array, shape (n_dev, n_features)
  dev_z: np.array, shape (n_dev,) (if not provided, will be predicted using a classifier trained on x_train, z_train)
  source: int, source class
  target: int, target class

  Returns:
  mean_source: np.array, shape (n_features,)
  mean_target: np.array, shape (n_features,)
  A: np.array, shape (n_features, n_features)
  train_x_transformed: np.array, shape (n_train, n_features)
  dev_x_transformed: np.array, shape (n_dev, n_features)

  The function computes A, the optimal transport matrix, and then transforms the source class to the target class using A.
  NOTE: it only transforms the source class to the target class, and leaves the rest of the classes UNCHANGED.
  """

  source_x_train = x_train[z_train == source]
  target_x_train = x_train[z_train == target]

  mean_source = np.mean(source_x_train, axis=0)
  mean_target = np.mean(target_x_train, axis=0)
  A = get_optimal_gaussian_transport_func(source_x_train, target_x_train, diag_cov=False, use_shrinkage=False)

  dev_x_transformed = x_dev.copy()
  train_x_transformed = x_train.copy()

  if dev_z is None:
    clf = SGDClassifier(random_state=0, loss='hinge', alpha=1e-3, max_iter=500000, tol=1e-5)
    clf.fit(x_train, z_train)
    dev_z = clf.predict(x_dev)

  dev_x_transformed[dev_z == source] = mean_target + (dev_x_transformed[dev_z == source] - mean_source) @ A
  train_x_transformed[z_train == source] = mean_target + (train_x_transformed[z_train == source] - mean_source) @ A

  return mean_source, mean_target, A, train_x_transformed, dev_x_transformed
