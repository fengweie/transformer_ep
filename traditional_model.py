from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing, neighbors,svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from util_1 import *
from sklearn.metrics import roc_auc_score
import random
import torch.backends.cudnn as cudnn
import argparse
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,
from imblearn.metrics import specificity_score,sensitivity_score
# ,sensitivity_score
import pdb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import *
params = {'legend.fontsize': 13,
          'axes.labelsize': 15,
          'axes.titlesize': 15,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15}  # define pyplot parameters
pylab.rcParams.update(params)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV,cross_val_score, StratifiedKFold, learning_curve
train_arg_parser = argparse.ArgumentParser(description="parser")
train_arg_parser.add_argument('--seed', type=int, default=111, metavar='S',
                    help='random seed (default: 1)')
args = train_arg_parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
TRAIN_RANDOM_SEED = args.seed
print("training seed used:", TRAIN_RANDOM_SEED)
np.random.seed(TRAIN_RANDOM_SEED)
random.seed(TRAIN_RANDOM_SEED)
# os.environ['CUDA_VISIBLE_DEVICES'] = str(4)

batch_size = 64  ## Because we have a small data set, keep it as 1
# # ### Making the dataset class for reading
from imblearn.over_sampling import SMOTE,KMeansSMOTE,SVMSMOTE,ADASYN,BorderlineSMOTE
from sklearn.model_selection import train_test_split
import xgboost as xgb

from sklearn.calibration import CalibratedClassifierCV
def compute_metrics(gt, pred, competition=True):
    """
    Computes accuracy, precision, recall and F1-score from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
        competition: whether to use competition tasks. If False,
          use all tasks
    Returns:
        List of AUROCs of all classes.
    """
    # print(gt[0])
    # print(gt_onehot[0])
    AUROCs, Accus, Senss, Recas, Specs = [], [], [], [], []
    gt_np = gt
    THRESH = 0.35
    pred_np = pred>=THRESH

    # AUROCs.append(roc_auc_score(gt_np, pred_np))
    Accus.append(accuracy_score(gt_np, pred_np))
    Senss.append(sensitivity_score(gt_np, pred_np))
    Specs.append(specificity_score(gt_np, pred_np))
    return Accus, Senss, Specs
from sklearn.metrics import balanced_accuracy_score
sm = SMOTE(random_state=42)
# sm = SVMSMOTE(random_state=42)
# sm = ADASYN(random_state=42)
# sm = BorderlineSMOTE(random_state=42)
# sm = KMeansSMOTE(random_state=42)

# ####################################
all_path = '/mnt/workdir/fengwei/transformer_master-master/data/validation/all_single_all_cols_v8.csv'

all_data = pd.DataFrame(pd.read_csv(all_path))
chongqing_path = '/mnt/workdir/fengwei/transformer_master-master/data/validation/chongqing_single_all_cols_v8.csv'
chongqing_path_2 = '/mnt/workdir/fengwei/transformer_master-master/data/validation/chongqing_single_all_cols_v8_2.csv'
chongqingdata_1 = pd.DataFrame(pd.read_csv(chongqing_path))
chongqingdata_2 = pd.DataFrame(pd.read_csv(chongqing_path_2))
chongqingdata = pd.concat([chongqingdata_1,chongqingdata_2])
print("chongqingdata.columns:",chongqingdata.columns)
Glasgowdata = all_data[all_data['cohort']=='Glasgow']
perthdata = all_data[all_data['cohort']=='Perth']
Chinadata = all_data[all_data['cohort']=='Guangzhou']
Malaysiadata = all_data[all_data['cohort']=='Kuala Lumpur']

X_Malaysia, y_Malaysia = Malaysiadata.iloc[:,2:-1],Malaysiadata.iloc[:,-1]

X_chongqing, y_chongqing = chongqingdata.iloc[:,2:-1],chongqingdata.iloc[:,-1]

X_China, y_China = Chinadata.iloc[:,2:-1],Chinadata.iloc[:,-1]

X_perth,y_perth = perthdata.iloc[:,2:-1],perthdata.iloc[:,-1]

print('Glasgowdata',Glasgowdata['psuedo_outcome'].value_counts())
print('perthdata',perthdata['psuedo_outcome'].value_counts())
print('Malaysiadata',Malaysiadata['psuedo_outcome'].value_counts())
print('Chinadata',Chinadata['psuedo_outcome'].value_counts())
print('chongqingdata',chongqingdata['psuedo_outcome'].value_counts())
# traindata = pd.concat([Glasgowdata,Chinadata,perthdata,Malaysiadata])
traindata = pd.concat([Malaysiadata,Glasgowdata,Chinadata,perthdata,chongqingdata])
# Glasgowdata

print('all_data',traindata['psuedo_outcome'].value_counts())
externaldata = Chinadata
# externaldata = pd.concat([Glasgowdata,Chinadata,perthdata,Malaysiadata])
print('externaldata',externaldata['psuedo_outcome'].value_counts())
# print("traindata.columns:",traindata.columns)
X_train, y_train = traindata.iloc[:,2:-1],traindata.iloc[:,-1]
# print(traindata.columns)

# scaler = StandardScaler()
#
# X_train = scaler.fit_transform(X_train)
# print(X_train)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)
# X_test, y_test = X_train, y_train
# X_train, y_train = sm.fit_resample(X_train, y_train)

X_external, y_external = externaldata.iloc[:,2:-1],externaldata.iloc[:,-1]
print("len of Glasgow train dataset:",len(X_train))
print("len of Glasgow test dataset:",len(X_test))
print("len of Malaysia dataset:",len(Malaysiadata))
print("len of chongqing dataset:",len(chongqingdata))
print("len of China dataset:",len(Chinadata))
print("len of perth dataset:",len(perthdata))
input_dim = X_train.shape[1]

src_var = input_dim
# kfold = StratifiedKFold(n_splits=5)
# random_state = 2
# classifiers = []
# classifiers.append(SVC(random_state=random_state))
# classifiers.append(DecisionTreeClassifier(random_state=random_state))
# classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),
#                                       random_state=random_state, learning_rate=0.2))
# classifiers.append(RandomForestClassifier(random_state=random_state))
# classifiers.append(ExtraTreesClassifier(random_state=random_state))
# classifiers.append(GradientBoostingClassifier(random_state=random_state))
# classifiers.append(MLPClassifier(random_state=random_state))
# classifiers.append(KNeighborsClassifier())
# classifiers.append(LogisticRegression(random_state=random_state))
# classifiers.append(LinearDiscriminantAnalysis())
#
# cv_results = []
# for classifier in classifiers:
#     cv_results.append(cross_val_score(classifier, X_train, y_train, scoring='accuracy',
#                                       cv=kfold, n_jobs=4))
#
# cv_means = []
# for cv_result in cv_results:
#     cv_means.append(cv_result.mean())

# cv_res = pd.DataFrame({'CrossValMeans': cv_means, 'AlgoClassif': ['SVC', 'DecisionTree', 'AdaBoost',
#                                                                   'RandomForest', 'ExtraTrees', 'GradientBoosting',
#                                                                   'MultipleLayerPerceptron', 'KNeighboors',
#                                                                   'LogisticRegression', 'LinearDiscriminantAnalysis']})
# g = sns.barplot('CrossValMeans', 'AlgoClassif', data=cv_res, orient='h')

# hyperparameter
# 1
# # # 分类器使用 xgboost
# #
# # # 设定网格搜索的xgboost参数搜索范围，值搜索XGBoost的主要6个参数
# param_dist = {
#     # 'n_estimators': range(80, 200, 4),
#     # 'max_depth': range(2, 15, 1),
#     'n_estimators': [10,20,50, 100,300],
#     'max_depth': [2, 5,10,20],
#     'learning_rate': np.linspace(0.01, 2, 20),
#     'subsample': np.linspace(0.7, 0.9, 20),
#     'colsample_bytree': np.linspace(0.5, 0.98, 10),
#     'min_child_weight': range(1, 9, 1)
# }
# model_xgb = GridSearchCV(xgb.XGBClassifier(), param_grid=param_dist, n_jobs=-1, verbose=1, scoring='roc_auc',
#                          cv=4)
# model_xgb.fit(X_train, y_train)
# y_val_pred = model_xgb.best_estimator_.predict(X_test)
# Accus, Senss, Specs = compute_metrics(y_test, y_val_pred, competition=True)
# print(Accus, Senss, Specs)
# print('Best parameter:%s' % model_xgb.best_estimator_)

params = {
    "C":[100,],
    "kernel":["rbf"],
    # "gamma":[0.1,0.01,0.001,]
    # "C":[1e-5,1e-3,0.1,1,10,100],
    # "kernel":["linear", "poly", "rbf"],
    # "gamma":[0.0001,0.1,10,50,100]
}
# "sigmoid"
svm = SVC(probability=True)

grid = GridSearchCV(svm, params,scoring='balanced_accuracy', cv=5, n_jobs=4, verbose=1)
grid.fit(X_train, y_train)
print(grid.best_estimator_)
y_val_pred = grid.best_estimator_.predict(X_test)
pre_y = grid.best_estimator_.predict_proba(X_test)[:, 1]
Accus, Senss, Specs = compute_metrics(y_test, pre_y, competition=True)
## 可视化在验证集上的Roc曲线

fpr_Nb, tpr_Nb, _ = roc_curve(y_test, pre_y)
aucval = auc(fpr_Nb, tpr_Nb)    # 计算auc的取值

print(aucval, Accus, Senss, Specs)
LogReg = LogisticRegression()

LG_param_grid = {'random_state': [3, 5, 7, 10], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [1000]}

gsLG = GridSearchCV(LogReg, param_grid=LG_param_grid, cv=5, scoring='balanced_accuracy',
                    n_jobs=4, verbose=1)

gsLG.fit(X_train, y_train)
LG_best = gsLG.best_estimator_

print(LG_best)
# print(gsLG.best_score_)

y_val_pred = LG_best.predict(X_test)
pre_y = LG_best.predict_proba(X_test)[:, 1]
Accus, Senss, Specs = compute_metrics(y_test, pre_y , competition=True)
## 可视化在验证集上的Roc曲线

fpr_Nb, tpr_Nb, _ = roc_curve(y_test, pre_y)
aucval = auc(fpr_Nb, tpr_Nb)    # 计算auc的取值

print(aucval, Accus, Senss, Specs)

DTC = DecisionTreeClassifier()
adaDTC = AdaBoostClassifier(DTC, random_state=7)
ada_param_grid = {"base_estimator__criterion": ["gini", "entropy"],
                  # "base_estimator__splitter": ["best", "random"],
                  # "algorithm": ["SAMME", "SAMME.R"],
                  "n_estimators": [1,50,100, 500,1000,2000],
                  'learning_rate': [1.5, 0.5, 0.25, 0.01, 0.001]}

gsadaDTC = GridSearchCV(adaDTC, param_grid=ada_param_grid, cv=5,
                        scoring='balanced_accuracy', n_jobs=4, verbose=1)

gsadaDTC.fit(X_train, y_train)
ada_best = gsadaDTC.best_estimator_
y_val_pred = ada_best.predict(X_test)
# print(y_val_pred)
pre_y = ada_best.predict_proba(X_test)[:, 1]
Accus, Senss, Specs = compute_metrics(y_test, pre_y, competition=True)
## 可视化在验证集上的Roc曲线

print(gsadaDTC.best_score_)
fpr_Nb, tpr_Nb, _ = roc_curve(y_test, pre_y)
aucval = auc(fpr_Nb, tpr_Nb)    # 计算auc的取值

print(aucval, Accus, Senss, Specs)
print(ada_best)
# print(gsadaDTC.best_score_)


# 2
RandFor = RandomForestClassifier()

RF_param_frid = {"max_depth": [1,5,10,50, 100, 200], "max_features": [1, 3, 10,20],
                 "min_samples_split": [2, 3, 10, 20], "min_samples_leaf": [1, 3, 10,20],
                 "bootstrap": [False], "n_estimators": [5, 20, 100, 300, 500], "criterion": ["gini", "entropy"]}

gsRandFor = GridSearchCV(RandFor, param_grid=RF_param_frid, cv=5, scoring='balanced_accuracy',
                         n_jobs=4, verbose=1)

gsRandFor.fit(X_train, y_train)
RF_best = gsRandFor.best_estimator_

print(RF_best)
# print(gsRandFor.best_score_)
y_val_pred = RF_best.predict(X_test)
pre_y = RF_best.predict_proba(X_test)[:, 1]
Accus, Senss, Specs = compute_metrics(y_test, pre_y, competition=True)
## 可视化在验证集上的Roc曲线

fpr_Nb, tpr_Nb, _ = roc_curve(y_test, pre_y)
aucval = auc(fpr_Nb, tpr_Nb)    # 计算auc的取值

print(aucval, Accus, Senss, Specs)
# 3
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss': ["deviance"], 'n_estimators': [5, 20, 100, 300, 500], 'learning_rate': [5,1, 0.1, 0.05, 0.01],
                 'max_depth': [4, 8, 10, 20], 'min_samples_leaf': [50,100, 150], 'max_features': [0.5,0.3, 0.1]}

gsGBC = GridSearchCV(GBC, param_grid=gb_param_grid, cv=5, scoring='balanced_accuracy',
                     n_jobs=4, verbose=1)

gsGBC.fit(X_train, y_train)
GBC_best = gsGBC.best_estimator_

print(GBC_best)
# print(gsGBC.best_score_)
y_val_pred = GBC_best.predict(X_test)
pre_y = GBC_best.predict_proba(X_test)[:, 1]
Accus, Senss, Specs = compute_metrics(y_test, pre_y, competition=True)
## 可视化在验证集上的Roc曲线

fpr_Nb, tpr_Nb, _ = roc_curve(y_test, pre_y)
aucval = auc(fpr_Nb, tpr_Nb)    # 计算auc的取值

print(aucval, Accus, Senss, Specs)
# 4

# 5
MLPC = MLPClassifier()

mlp_param_grid = {'solver': ['sgd', 'adam', 'lbfgs'], 'activation': ['relu', 'tanh'],
                  'hidden_layer_sizes': [100,  200, 50, 500, 1000], 'max_iter': [2000, 10000]}

gsMLP = GridSearchCV(MLPC, param_grid=mlp_param_grid, cv=5, scoring='balanced_accuracy',
                     n_jobs=-1, verbose=1)

gsMLP.fit(X_train, y_train)
MLP_best = gsMLP.best_estimator_

print(MLP_best)
print(gsMLP.best_score_)
y_val_pred = MLP_best.predict(X_test)
pre_y = MLP_best.predict_proba(X_test)[:, 1]
Accus, Senss, Specs = compute_metrics(y_test, pre_y, competition=True)
## 可视化在验证集上的Roc曲线

fpr_Nb, tpr_Nb, _ = roc_curve(y_test, pre_y)
aucval = auc(fpr_Nb, tpr_Nb)    # 计算auc的取值

print(aucval, Accus, Senss, Specs)



# ensemble modeling
EnseModel = VotingClassifier(estimators=[('adac', ada_best), ('gbc', GBC_best), ('mlp', MLP_best),
                                         ('lg', LG_best,), ('rfc', RF_best)], voting='soft', n_jobs=-1)

EnseModel = EnseModel.fit(X_train, y_train)

# predict = EnseModel.predict(X_test)
# dataset = dataset.fillna(np.nan)

y_val_pred = EnseModel.predict(X_test)
## 可视化在验证集上的Roc曲线
pre_y = EnseModel.predict_proba(X_test)[:, 1]
fpr_Nb, tpr_Nb, _ = roc_curve(y_test, pre_y)
aucval = auc(fpr_Nb, tpr_Nb)    # 计算auc的取值

Accus, Senss, Specs = compute_metrics(y_test, pre_y, competition=True)
print("voting",aucval, Accus, Senss, Specs)



## Create a best_model variable for loading
# ##################随机森林
# DTC = RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=20, random_state=10)
# DTC.fit(X_train,y_train)
# accuracy=DTC.score(X_test,y_test)
# print("RandomForest",accuracy)
#
# bg = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators = 20)
# bg.fit(X_train,y_train)
# accuracy1=bg.score(X_test,y_test)
# print("BaggingClassifier",accuracy1)
#
# adb = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 20, learning_rate = 1)
# adb.fit(X_train,y_train)
# accuracy = adb.score(X_test,y_test)
# print("AdaBoost",accuracy)
#
# gbc = GradientBoostingClassifier()
# gbc.fit(X_train,y_train)
# accuracy3 = gbc.score(X_test,y_test)
# print("GradientBoosting",accuracy3)
#
# clf7=AdaBoostClassifier(RandomForestClassifier(n_estimators=100,criterion='entropy',
#                 max_depth=20, random_state=10), n_estimators=100, learning_rate=1.0, random_state=10)
# clf7.fit(X_train,y_train)
# accuracy7=clf7.score(X_test,y_test)
# print("AdaBoostClassifier,RandomForest",accuracy7)
#
# BG = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators = 20)
# RF = RandomForestClassifier(max_depth=2, random_state=0)
# NN = MLPClassifier(solver='lbfgs', alpha=1e-1,hidden_layer_sizes=(200,), random_state=1,max_iter=5000)
# SV = svm.SVC(kernel='linear',C=10,)
# ADA = AdaBoostClassifier()
# KNN = KNeighborsClassifier()
# DTC = tree.DecisionTreeClassifier()
# MNB = MultinomialNB()
# GBC = GradientBoostingClassifier()
#
# evs = VotingClassifier(estimators=[('BG',BG),('RF',RF),('NN',NN),('SV',SV),('ADA',ADA),('KNN',KNN),('DTC',DTC),('MNB',MNB),('GBC',GBC)],voting='hard')
# evs.fit(X_train,y_train)
# accuracy=evs.score(X_test,y_test)
# print("VotingClassifier",accuracy)
