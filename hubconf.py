import torch
from torch import nn
import torch.optim as optim
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_circles, load_digits

###### PART 1 ######

def get_data_blobs(n_points=100):
  X, y = make_blobs(n_samples=n_points)
  return X,y

def get_data_circles(n_points=100):
  X, y = make_circles(n_samples=n_points)
  return X,y
