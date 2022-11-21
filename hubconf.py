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
import torch.nn.functional as Fun

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

def build_lr_model(X,y):
  lr_model = LogisticRegression()
  if X.ndim > 2:
      n_samples = len(X)
      X= X.reshape((n_samples, -1))
  lr_model.fit(X,y)
  return lr_model

def build_rf_model(X,y):
  rf_model = RandomForestClassifier()
  if X.ndim > 2:
      n_samples = len(X)
      X= X.reshape((n_samples, -1))
  rf_model.fit(X,y)
  return rf_model

def get_metrics(model,X,y):
  if X.ndim > 2:
      n_samples = len(X)
      X= X.reshape((n_samples, -1))
  classes = set()
  for i in y:
      classes.add(i)
  num_classes = len(classes)

  ypred = model.predict(X)
  acc, prec, rec, f1, auc = 0,0,0,0,0
  acc = accuracy_score(y,ypred)
  if num_classes == 2:
    prec = precision_score(y,ypred)
    recall = recall_score(y,ypred)
    f1 = f1_score(y,ypred)
    auc = roc_auc_score(y,ypred)

  else:
    prec = precision_score(y,ypred,average='macro')
    recall = recall_score(y,ypred,average='macro')
    f1 = f1_score(y,ypred,average='macro')
    pred_prob = model.predict_proba(X)
    roc_auc_score(y, pred_prob, multi_class='ovr')

  return acc, prec, rec, f1, auc

def get_paramgrid_lr():
  lr_param_grid = {'penalty' : ['l1','l2']}
  return lr_param_grid

def get_paramgrid_rf():
  rf_param_grid = { 'n_estimators' : [1,10,100],'criterion' :["gini", "entropy"], 'max_depth' : [1,10,None]  }
  return rf_param_grid


def perform_gridsearch_cv_multimetric(model, param_grid, cv=5, X=None, y=None, metrics=['accuracy','roc_auc']):
  top1_scores = []
  
  if X.ndim > 2:
      n_samples = len(X)
      X= X.reshape((n_samples, -1))
      
  for score in metrics:
      grid_search_cv = GridSearchCV(model,param_grid,scoring = score,cv=cv)
      grid_search_cv.fit(X,y)
      top1_scores.append(grid_search_cv.best_estimator_.get_params())
      
  return top1_scores

####### PART 3 #################

class cs21m010NN(nn.Module):
  def __init__(self,inp_dim=64,hid_dim=13,num_classes=10):
    super(cs21m010NN,self).__init__()
    self.fc_encoder = nn.Linear(inp_dim, hid_dim)
    self.fc_decoder = nn.Linear(hid_dim, inp_dim)
    self.fc_classifier = nn.Linear(hid_dim, num_classes)
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1) 

  def forward(self,x):
    x = nn.Flatten()
    x_enc = self.fc_encoder(x)
    x_enc = self.relu(x_enc)
    y_pred = self.fc_classifier(x_enc)
    y_pred = self.softmax(y_pred)
    x_dec = self.fc_decoder(x_enc)
    
    return y_pred, x_dec
