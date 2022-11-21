import torch
from torch import nn
import torch.optim as optim
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_circles, load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import homogeneity_score,completeness_score,v_measure_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,recall_score,roc_auc_score,precision_score,f1_score

###### PART 1 ######

def get_data_blobs(n_points=100):
  X, y = make_blobs(n_samples=n_points)
  return X,y

def get_data_circles(n_points=100):
  X, y = make_circles(n_samples=n_points)
  return X,y

def get_data_mnist():
  digits = load_digits(n_class=10, return_X_y=False, as_frame=False)
  X = digits.data
  y = digits.target
  return X,y

def build_kmeans(X=None,k=10):
  km = KMeans(n_clusters=k).fit(X)
  return km

def assign_kmeans(km,X):
  ypred = km.predict(X)
  return ypred

def compare_clusterings(ypred_1,ypred_2):
  h = homogeneity_score(ypred_1,ypred_2)
  c = completeness_score(ypred_1,ypred_2)
  v = v_measure_score(ypred_1,ypred_2)
  return h,c,v

###### PART 2 ##########

def build_lr_model(X, y):
  lr_model = LogisticRegression(random_state=0).fit(X, y)
  return lr_model

def build_rf_model(X, y):
  rf_model=RandomForestClassifier(max_depth=4, random_state=0)
  return rf_model

def get_metrics(model,X,y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2,stratify=y)
  model.fit(X_train,y_train)
  
  y_pred_test = model.predict(X_test)
  acc=accuracy_score(y_test, y_pred_test)
  rec=recall_score(y_test,y_pred_test)
  prec=precision_score(y_test,y_pred_test)
  f1=f1_score(y_test,y_pred_test)
  auc=roc_auc_score(y_test,y_pred_test)
  return acc, prec, rec, f1, auc

def get_paramgrid_lr():
  lr_param_grid = {'penalty' : ['l1','l2']}
  return lr_param_grid

def get_paramgrid_rf():
  rf_param_grid = { 'n_estimators' : [1,10,100],'criterion' :["gini", "entropy"], 'max_depth' : [1,10,None]  }
  return rf_param_grid
