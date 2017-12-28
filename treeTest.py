import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
pd.set_option('display.max_rows', 1000)


########## Dataset 1 - Abalone age #############
# colnames = ["sex", "length", "diameter", "height", "whole_weight","shucked_weight", "viscera_weight", "shell_weight", "rings"]
# dataset1 = pd.read_csv("F:\\Apps\\Code Projects\\GradientBoostingSVR\\datasets\\abalone.data",names=colnames)
#
# X1 = dataset1.drop("rings", axis=1)
# y1 = dataset1["rings"].values
# X1 = pd.get_dummies(X1,columns=["sex"])
# X1 = X1.values
# params = {'n_estimators': 100, 'min_samples_leaf': 4,
#           'learning_rate': 0.01}
# print("Abalone age dataset 1")
# print("dataset size: "+str(len(dataset1)))
# print(params)
# clf = ensemble.GradientBoostingRegressor(**params)
# scores = cross_val_score(clf, X1,y1, cv=10,scoring='neg_mean_squared_error')

# print("MSE 10 Fold Cross Validation average:")
# print(scores.mean()*(-1))
# print("################################################")
# print()
########### Dataset 2 - Concrete Data #############
colnames = ["cement", "burst_furnace_slag", "fly_ash", "water", "superplasticizer","coarse_aggregate", "fine_aggregate", "age", "mpa"]
dataset2 = pd.read_excel("F:\\Apps\\Code Projects\\GradientBoostingSVR\\datasets\\Concrete_Data.xls",names=colnames)

X2 = dataset2.drop("mpa", axis=1)
y2 = dataset2["mpa"].values
# X1 = pd.get_dummies(X1,columns=["sex"])
X2 = X2.values
params = {'n_estimators': 100, 'min_samples_leaf': 30,
          'learning_rate': 0.01}
print("Concrete dataset 2")
print("dataset size: "+str(len(dataset2)))
print(params)
clf = ensemble.GradientBoostingRegressor(**params)
scores = cross_val_score(clf, X2,y2, cv=10,scoring='neg_mean_squared_error')
print("MSE 10 Fold Cross Validation average:")
print(scores.mean()*(-1))
print("################################################")
print()
########### Dataset 3 - Forest Fires Data #############

colnames = ['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']
dataset3 = pd.read_csv("F:\\Apps\\Code Projects\\GradientBoostingSVR\\datasets\\forestfires.csv")
X3 = dataset3.drop("area", axis=1)
y3 = np.log1p(dataset3["area"].values)
X3 = pd.get_dummies(X3,columns=["month","day"])
X3 = X3.values
params = {'n_estimators': 100, 'min_samples_leaf': 30,
          'learning_rate': 0.01}
print("Forest Fires dataset 3")
print("dataset size: "+str(len(dataset3)))
print(params)
clf = ensemble.GradientBoostingRegressor(**params)
scores = cross_val_score(clf, X3,y3, cv=10,scoring='neg_mean_squared_error')
print("MSE 10 Fold Cross Validation average:")
print(scores.mean()*(-1))
print("################################################")
print()
########### Dataset 4 - Computer Hardware Data #############

colnames = ['vendor','Model','MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP','ERP']
dataset4 = pd.read_csv("F:\\Apps\\Code Projects\\GradientBoostingSVR\\datasets\\machine.data.txt",names=colnames)
# X3 = dataset3.drop(dataset3.index[:1])
X4 = dataset4.drop(["vendor","Model","ERP","PRP"], axis=1)
y4 = np.log1p(dataset4["PRP"].values)
# X4 = pd.get_dummies(X4,columns=["month","day"])
X4 = X4.values
params = {'n_estimators': 100, 'min_samples_leaf': 30,
          'learning_rate': 0.01}
print("Computer Hardware dataset 4")
print("dataset size: "+str(len(dataset4)))
print(params)
clf = ensemble.GradientBoostingRegressor(**params)
scores = cross_val_score(clf, X4,y4, cv=10,scoring='neg_mean_squared_error')
print("MSE 10 Fold Cross Validation average:")
print(scores.mean()*(-1))
print("################################################")
print()
########### Dataset 5 - Twiter Data #############

# colnames = []
# dataset5 = pd.read_csv("F:\\Apps\\Code Projects\\GradientBoostingSVR\\datasets\\OnlineNewsPopularity.csv")
# # X3 = dataset3.drop(dataset3.index[:1])
# X5 = dataset5.drop(["url"," shares"," timedelta"], axis=1)
# y5 = np.log1p(dataset5[" shares"].values)
# # X4 = pd.get_dummies(X4,columns=["month","day"])
# X5 = X5.values
# params = {'n_estimators': 10, 'min_samples_leaf': 4,
#           'learning_rate': 0.01}
# print("Twiter dataset 5")
# print("dataset size: "+str(len(dataset5)))
# print(params)
# clf = ensemble.GradientBoostingRegressor(**params)
# scores = cross_val_score(clf, X5,y5, cv=10,scoring='neg_mean_squared_error')
# print("MSE 10 Fold Cross Validation average:")
# print(scores.mean()*(-1))
