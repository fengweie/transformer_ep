
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score
import random
import numpy as np
import argparse
import time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
# # ### Making the dataset class for reading
from imblearn.over_sampling import SMOTE,KMeansSMOTE,SVMSMOTE,ADASYN,BorderlineSMOTE
from sklearn.model_selection import train_test_split
import xgboost as xgb
def auc_confint_cal(gt, pred_soft_list, pred_list):
    n_bootstraps = 1000
    bootstrapped_auc = []
    bootstrapped_acc = []
    bootstrapped_sp = []
    bootstrapped_se = []
    bootstrapped_precision = []
    rng_seed = 42
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(gt), len(gt))
        if len(np.unique(np.array(gt)[indices])) < 2:
            continue
        auc = roc_auc_score(np.array(gt)[indices], np.array(pred_soft_list)[indices])
        confusion = confusion_matrix(np.array(gt)[indices], np.array(pred_list)[indices])
        accuracy = 0
        if float(np.sum(confusion)) != 0:
            accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
        specificity = 0
        if float(confusion[0, 0] + confusion[0, 1]) != 0:
            specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
        sensitivity = 0
        if float(confusion[1, 1] + confusion[1, 0]) != 0:
            sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
        precision = 0
        if float(confusion[1, 1] + confusion[0, 1]) != 0:
            precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
        bootstrapped_auc.append(auc)
        bootstrapped_acc.append(accuracy)
        bootstrapped_sp.append(specificity)
        bootstrapped_se.append(sensitivity)
        bootstrapped_precision.append(precision)
    auc_scores = np.array(bootstrapped_auc)
    auc_scores.sort()
    acc_scores = np.array(bootstrapped_acc)
    acc_scores.sort()
    sp_scores = np.array(bootstrapped_sp)
    sp_scores.sort()
    se_scores = np.array(bootstrapped_se)
    se_scores.sort()
    precision_scores = np.array(bootstrapped_precision)
    precision_scores.sort()
    print("95ci auc",auc_scores.mean(),np.median(auc_scores), auc_scores[int(0.05*len(auc_scores))], auc_scores[int(0.95*len(auc_scores))])
    print("95ci acc",acc_scores.mean(),np.median(acc_scores), acc_scores[int(0.05*len(acc_scores))], acc_scores[int(0.95*len(acc_scores))])
    print("95ci sp",sp_scores.mean(), np.median(sp_scores),sp_scores[int(0.05*len(sp_scores))], sp_scores[int(0.95*len(sp_scores))])
    print("95ci se",se_scores.mean(),np.median(se_scores), se_scores[int(0.05*len(se_scores))], se_scores[int(0.95*len(se_scores))])
    print("95ci precision",precision_scores.mean(), np.median(precision_scores),precision_scores[int(0.05*len(precision_scores))], precision_scores[int(0.95*len(precision_scores))])

def compute_metrics(real_results, pred_results,thres_val):
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

    print(real_results.shape, pred_results.shape)
    test_AUC_ROC = roc_auc_score(real_results, pred_results)
    pred_res = pred_results >= thres_val

    TP = sum((pred_res == 1) & (real_results == 1))
    FN = sum((pred_res == 0) & (real_results == 1))
    TN = sum((pred_res == 0) & (real_results == 0))
    FP = sum((pred_res == 1) & (real_results == 0))
    test_sens = TP / (TP + FN)
    test_spec = TN / (TN + FP)
    test_accuracy_calculated = (TN + TP) / (TN + TP + FP + FN)
    test_weighted_acc = (TN / (FP + TN)) * 0.5 + (TP / (TP + FN)) * 0.5
    auc_confint_cal(real_results, pred_results, pred_res)
    return test_sens, test_spec, test_accuracy_calculated,test_weighted_acc,test_AUC_ROC

train_arg_parser = argparse.ArgumentParser(description="parser")
train_arg_parser.add_argument('--seed', type=int, default=111, metavar='S',
                    help='random seed (default: 1)')
args = train_arg_parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
TRAIN_RANDOM_SEED = args.seed
print("training seed used:", TRAIN_RANDOM_SEED)
np.random.seed(TRAIN_RANDOM_SEED)
random.seed(TRAIN_RANDOM_SEED)
# os.environ['CUDA_VISIBLE_DEVICES'] = str(4)
# perth_val = True
batch_size = 64  ## Because we have a small data set, keep it as 1
# # ### Making the dataset class for reading
sm = SMOTE(random_state=42)
# sm = SVMSMOTE(random_state=42)
# sm = ADASYN(random_state=42)
# sm = BorderlineSMOTE(random_state=42)
# sm = KMeansSMOTE(random_state=42)

# ####################################
use_age_category = True
# ####################################
if use_age_category:
    print("use use_age_category")
    all_path = '/mnt/workdir/fengwei/transformer_master-master/data/validation/all_single_all_cols_v8.csv'
    chongqing_path = '/mnt/workdir/fengwei/transformer_master-master/data/validation/chongqing_single_all_cols_v8.csv'
    chongqing_path_2 = '/mnt/workdir/fengwei/transformer_master-master/data/validation/chongqing_single_all_cols_v8_2.csv'
else:
    print("do not use use_age_category")
    all_path = '/mnt/workdir/fengwei/transformer_master-master/data/validation/age_no_three/all_single_all_cols_v8.csv'
    chongqing_path = '/mnt/workdir/fengwei/transformer_master-master/data/validation/age_no_three/chongqing_single_all_cols_v8_1.csv'
    chongqing_path_2 = '/mnt/workdir/fengwei/transformer_master-master/data/validation/age_no_three/chongqing_single_all_cols_v8_2.csv'
all_data = pd.DataFrame(pd.read_csv(all_path))
chongqingdata_1 = pd.DataFrame(pd.read_csv(chongqing_path))
chongqingdata_2 = pd.DataFrame(pd.read_csv(chongqing_path_2))
chongqingdata = pd.concat([chongqingdata_1,chongqingdata_2])
# chongqingdata = chongqingdata[chongqingdata['focal']==0]

if use_age_category:
    drugs_to_keep = ['pid', 'cohort', 'sex', 'age_init_29', 'age_init_46', 'age_init_>46',
           'pretrt_sz_5', 'focal', 'fam_hx', 'febrile', 'ci', 'birth_t', 'head',
           'drug', 'alcohol', 'cvd', 'psy', 'ld', 'lesion_A', 'lesion_E',
           'lesion_N', 'eeg_cat_A', 'eeg_cat_E', 'eeg_cat_N', 'CBZ', 'LEV', 'LTG',
           'OXC', 'PHT', 'TPM', 'VPA', 'psuedo_outcome']
else:
    drugs_to_keep = ['pid', 'cohort', 'sex', 'age_init',
           'pretrt_sz_5', 'focal', 'fam_hx', 'febrile', 'ci', 'birth_t', 'head',
           'drug', 'alcohol', 'cvd', 'psy', 'ld', 'lesion_A', 'lesion_E',
           'lesion_N', 'eeg_cat_A', 'eeg_cat_E', 'eeg_cat_N', 'CBZ', 'LEV', 'LTG',
           'OXC', 'PHT', 'TPM', 'VPA', 'psuedo_outcome']
         #        ['pid', 'cohort','sex', 'age_init_29', 'age_init_46', 'age_init_>46',
         # 'pretrt_sz_5', 'focal', 'fam_hx', 'febrile', 'ci', 'birth_t', 'head',
         # 'drug', 'alcohol', 'cvd', 'psy', 'ld', 'lesion_A', 'lesion_E',
         #  'lesion_N', 'eeg_cat_A', 'eeg_cat_E', 'eeg_cat_N', 'psuedo_outcome']
chongqingdata = chongqingdata[[c for c in drugs_to_keep if c in chongqingdata]]
# print("chongqingdata.columns:",chongqingdata.columns)
# all_data = all_data[all_data['focal']==0]

all_data = all_data[[c for c in drugs_to_keep if c in all_data]]
# print("chongqingdata.columns:",chongqingdata.columns)
Glasgowdata = all_data[all_data['cohort']=='Glasgow']
perthdata = all_data[all_data['cohort']=='Perth']
Chinadata = all_data[all_data['cohort']=='Guangzhou']
Malaysiadata = all_data[all_data['cohort']=='Kuala Lumpur']
# print("Glasgowdata.columns:",Glasgowdata.columns)
# #####################################
# Glasgow_path = '/mnt/workdir/fengwei/transformer_master-master/data/validation/glasgow_single_all_cols_v8.csv'
# Glasgowdata = pd.DataFrame(pd.read_csv(Glasgow_path))
print("len of Glasgow dataset:",len(Glasgowdata))
X_Malaysia, y_Malaysia = Malaysiadata.iloc[:,2:-1],Malaysiadata.iloc[:,-1]
print("len of Malaysia dataset:",len(Malaysiadata))
X_chongqing, y_chongqing = chongqingdata.iloc[:,2:-1],chongqingdata.iloc[:,-1]
print("len of chongqing dataset:",len(chongqingdata))
# China_path = '/mnt/workdir/fengwei/transformer_master-master/data/validation/China_all_cols_v8.csv'
# Chinadata = pd.DataFrame(pd.read_csv(China_path))
X_China, y_China = Chinadata.iloc[:,2:-1],Chinadata.iloc[:,-1]
print("len of China dataset:",len(Chinadata))
# perth_path = '/mnt/workdir/fengwei/transformer_master-master/data/validation/perth_single_all_cols_v8.csv'
# perthdata = pd.DataFrame(pd.read_csv(perth_path))
X_perth,y_perth = perthdata.iloc[:,2:-1],perthdata.iloc[:,-1]
print("len of perth dataset:",len(perthdata))
print('Glasgowdata',Glasgowdata['psuedo_outcome'].value_counts())
print('perthdata',perthdata['psuedo_outcome'].value_counts())
print('Malaysiadata',Malaysiadata['psuedo_outcome'].value_counts())
print('Chinadata',Chinadata['psuedo_outcome'].value_counts())
print('chongqingdata',chongqingdata['psuedo_outcome'].value_counts())
cross_valid = False
if cross_valid:
    traindata = Glasgowdata
else:
    traindata = pd.concat([Malaysiadata,Glasgowdata,Chinadata,perthdata,chongqingdata])
#
print('all_data',traindata['psuedo_outcome'].value_counts())
externaldata = Chinadata
# externaldata = pd.concat([Glasgowdata,Chinadata,perthdata,Malaysiadata])
print('externaldata',externaldata['psuedo_outcome'].value_counts())
# print("traindata.columns:",traindata.columns)
X_train, y_train = traindata.iloc[:,2:-1],traindata.iloc[:,-1]
# print(traindata.columns)
#
if cross_valid:
    X_test, y_test = X_train, y_train
    X_train, y_train = sm.fit_resample(X_train, y_train)
else:
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)
    X_train, y_train = sm.fit_resample(X_train, y_train)

X_external, y_external = externaldata.iloc[:,2:-1],externaldata.iloc[:,-1]
input_dim = X_train.shape[1]

# # XGBoost, SVM, RF, penalized logistic regression
# MLPC = MLPClassifier()
#
# mlp_param_grid = {'solver': ['sgd', 'adam', 'lbfgs'], 'activation': ['relu', 'tanh'],
#                   'hidden_layer_sizes': [100,  200, 50, 500], 'max_iter': [20000]}
#
# gsMLP = GridSearchCV(MLPC, param_grid=mlp_param_grid, cv=5, scoring='balanced_accuracy',
#                      n_jobs=-1, verbose=1)
#
# gsMLP.fit(X_train, y_train)
# clf = gsMLP.best_estimator_
# print(clf)

clf = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(200,200), random_state=1)
clf.fit(X_train, y_train)
predict_results=clf.predict(X_test)
# print(clf.predict_proba(X_test).shape)
pre_y = clf.predict_proba(X_test)[:, 1]

if cross_valid:
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        external_sens, external_spec, external_accuracy_calculated,external_weighted_acc,external_AUC_ROC\
            = compute_metrics(y_test, pre_y, thres_val=i)
        print("Threshold: {0:.2f}".format(i),
              "\texternal_Sens: {0:.2f}".format(external_sens),
              "\texternal_Spec: {0:.2f}".format(external_spec),
              "\texternal_Acc: {0:.2f}".format(external_accuracy_calculated),
              '\texternal_Weighted Acc: {0:.2f}'.format(external_weighted_acc),
              "\texternal_AUC: {0:.2f}".format(external_AUC_ROC))
    print("Best model on perth dataset:")
    pre_perth = clf.predict_proba(X_perth)[:, 1]
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        perth_sens, perth_spec, perth_accuracy_calculated,perth_weighted_acc,perth_AUC_ROC\
            = compute_metrics(y_perth, pre_perth, thres_val=i)
        print("Threshold: {0:.2f}".format(i),
              "\tperth_Sens: {0:.2f}".format(perth_sens),
              "\tperth_Spec: {0:.2f}".format(perth_spec),
              "\tperth_Acc: {0:.2f}".format(perth_accuracy_calculated),
              '\tperth_Weighted Acc: {0:.2f}'.format(perth_weighted_acc),
              "\tperth_AUC: {0:.2f}".format(perth_AUC_ROC))
    print("Best model on malaysia dataset:")
    pre_Malaysia = clf.predict_proba(X_Malaysia)[:, 1]
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        malaysia_sens, malaysia_spec, malaysia_accuracy_calculated, malaysia_weighted_acc, malaysia_AUC_ROC \
            = compute_metrics(y_Malaysia, pre_Malaysia, thres_val=i)
        print("Threshold: {0:.2f}".format(i),
              "\tmalaysia_Sens: {0:.2f}".format(malaysia_sens),
              "\tmalaysia_Spec: {0:.2f}".format(malaysia_spec),
              "\tmalaysia_Acc: {0:.2f}".format(malaysia_accuracy_calculated),
              '\tmalaysia_Weighted Acc: {0:.2f}'.format(malaysia_weighted_acc),
              "\tmalaysia_AUC: {0:.2f}".format(malaysia_AUC_ROC))

    print("Best model on chongqing dataset:")
    pre_chongqing = clf.predict_proba(X_chongqing)[:, 1]
    chongqing_weight_arr = []
    chongqing_thres_arr = []
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        chongqing_sens, chongqing_spec, chongqing_accuracy_calculated, chongqing_weighted_acc, chongqing_AUC_ROC \
            = compute_metrics(y_chongqing, pre_chongqing, thres_val=i)
        chongqing_weight_arr.append(chongqing_weighted_acc)
        chongqing_thres_arr.append(i)
        print("Threshold: {0:.2f}".format(i),
              "\tchongqing_Sens: {0:.2f}".format(chongqing_sens),
              "\tchongqing_Spec: {0:.2f}".format(chongqing_spec),
              "\tchongqing_Acc: {0:.2f}".format(chongqing_accuracy_calculated),
              '\tchongqing_Weighted Acc: {0:.2f}'.format(chongqing_weighted_acc),
              "\tchongqing_AUC: {0:.2f}".format(chongqing_AUC_ROC))

    print("Best model on GUANGZHOU dataset:")
    pre_guangzhou = clf.predict_proba(X_China)[:, 1]
    guangzhou_weight_arr = []
    guangzhou_thres_arr = []
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        guangzhou_sens, guangzhou_spec, guangzhou_accuracy_calculated, guangzhou_weighted_acc, guangzhou_AUC_ROC \
            = compute_metrics(y_China, pre_guangzhou, thres_val=i)
        guangzhou_weight_arr.append(guangzhou_weighted_acc)
        guangzhou_thres_arr.append(i)
        print("Threshold: {0:.2f}".format(i),
              "\tguangzhou_Sens: {0:.2f}".format(guangzhou_sens),
              "\tguangzhou_Spec: {0:.2f}".format(guangzhou_spec),
              "\tguangzhou_Acc: {0:.2f}".format(guangzhou_accuracy_calculated),
              '\tguangzhou_Weighted Acc: {0:.2f}'.format(guangzhou_weighted_acc),
              "\tguangzhou_AUC: {0:.2f}".format(guangzhou_AUC_ROC))
else:
        for i in np.linspace(0, 1, 21):
            # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

            external_sens, external_spec, external_accuracy_calculated, external_weighted_acc, external_AUC_ROC \
                = compute_metrics(y_test, pre_y, thres_val=i)
            print("Threshold: {0:.2f}".format(i),
                  "\texternal_Sens: {0:.2f}".format(external_sens),
                  "\texternal_Spec: {0:.2f}".format(external_spec),
                  "\texternal_Acc: {0:.2f}".format(external_accuracy_calculated),
                  '\texternal_Weighted Acc: {0:.2f}'.format(external_weighted_acc),
                  "\texternal_AUC: {0:.2f}".format(external_AUC_ROC))


param_dist = {
    # 'n_estimators': range(80, 200, 4),
    # 'max_depth': range(2, 15, 1),
    'n_estimators': [100,300],
    'max_depth': [10],
    'learning_rate': [2,0.2],
    'subsample': [0.7,0.9],
    # 'colsample_bytree': 0.6,
    # 'min_child_weight': 2
}
# model_xgb = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
#               eval_metric='mlogloss', gamma=0, gpu_id=-1, importance_type=None,
#               interaction_constraints='', learning_rate=0.02, max_delta_step=0,
#               max_depth=10, min_child_weight=1,
#               monotone_constraints='()', n_estimators=30, n_jobs=96,
#               num_parallel_tree=1, predictor='auto', random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=0.7,
#               tree_method='exact', use_label_encoder=False,
#               validate_parameters=1, verbosity=None)

model_xgb = GridSearchCV(xgb.XGBClassifier(use_label_encoder =False,eval_metric='mlogloss'), param_grid=param_dist, n_jobs=-1, verbose=1, scoring='balanced_accuracy',
                         cv=4)
model_xgb.fit(X_train, y_train)
model_xgb = model_xgb.best_estimator_
y_val_pred = model_xgb.predict(X_test)
# print(model_xgb.best_estimator_)
time1 = time.time()
pre_y = model_xgb.predict_proba(X_test)[:, 1]
time2 = time.time()
print('XGB data cost {}'.format((time2-time1)/pre_y.shape[0]))
if cross_valid:
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        external_sens, external_spec, external_accuracy_calculated,external_weighted_acc,external_AUC_ROC\
            = compute_metrics(y_test, pre_y, thres_val=i)
        print("Threshold: {0:.2f}".format(i),
              "\texternal_Sens: {0:.2f}".format(external_sens),
              "\texternal_Spec: {0:.2f}".format(external_spec),
              "\texternal_Acc: {0:.2f}".format(external_accuracy_calculated),
              '\texternal_Weighted Acc: {0:.2f}'.format(external_weighted_acc),
              "\texternal_AUC: {0:.2f}".format(external_AUC_ROC))
    print("Best model on perth dataset:")
    pre_perth = model_xgb.predict_proba(X_perth)[:, 1]
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        perth_sens, perth_spec, perth_accuracy_calculated,perth_weighted_acc,perth_AUC_ROC\
            = compute_metrics(y_perth, pre_perth, thres_val=i)
        print("Threshold: {0:.2f}".format(i),
              "\tperth_Sens: {0:.2f}".format(perth_sens),
              "\tperth_Spec: {0:.2f}".format(perth_spec),
              "\tperth_Acc: {0:.2f}".format(perth_accuracy_calculated),
              '\tperth_Weighted Acc: {0:.2f}'.format(perth_weighted_acc),
              "\tperth_AUC: {0:.2f}".format(perth_AUC_ROC))
    print("Best model on malaysia dataset:")
    pre_Malaysia = model_xgb.predict_proba(X_Malaysia)[:, 1]
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        malaysia_sens, malaysia_spec, malaysia_accuracy_calculated, malaysia_weighted_acc, malaysia_AUC_ROC \
            = compute_metrics(y_Malaysia, pre_Malaysia, thres_val=i)
        print("Threshold: {0:.2f}".format(i),
              "\tmalaysia_Sens: {0:.2f}".format(malaysia_sens),
              "\tmalaysia_Spec: {0:.2f}".format(malaysia_spec),
              "\tmalaysia_Acc: {0:.2f}".format(malaysia_accuracy_calculated),
              '\tmalaysia_Weighted Acc: {0:.2f}'.format(malaysia_weighted_acc),
              "\tmalaysia_AUC: {0:.2f}".format(malaysia_AUC_ROC))

    print("Best model on chongqing dataset:")
    pre_chongqing = model_xgb.predict_proba(X_chongqing)[:, 1]
    chongqing_weight_arr = []
    chongqing_thres_arr = []
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        chongqing_sens, chongqing_spec, chongqing_accuracy_calculated, chongqing_weighted_acc, chongqing_AUC_ROC \
            = compute_metrics(y_chongqing, pre_chongqing, thres_val=i)
        chongqing_weight_arr.append(chongqing_weighted_acc)
        chongqing_thres_arr.append(i)
        print("Threshold: {0:.2f}".format(i),
              "\tchongqing_Sens: {0:.2f}".format(chongqing_sens),
              "\tchongqing_Spec: {0:.2f}".format(chongqing_spec),
              "\tchongqing_Acc: {0:.2f}".format(chongqing_accuracy_calculated),
              '\tchongqing_Weighted Acc: {0:.2f}'.format(chongqing_weighted_acc),
              "\tchongqing_AUC: {0:.2f}".format(chongqing_AUC_ROC))

    print("Best model on GUANGZHOU dataset:")
    pre_guangzhou = model_xgb.predict_proba(X_China)[:, 1]
    guangzhou_weight_arr = []
    guangzhou_thres_arr = []
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        guangzhou_sens, guangzhou_spec, guangzhou_accuracy_calculated, guangzhou_weighted_acc, guangzhou_AUC_ROC \
            = compute_metrics(y_China, pre_guangzhou, thres_val=i)
        guangzhou_weight_arr.append(guangzhou_weighted_acc)
        guangzhou_thres_arr.append(i)
        print("Threshold: {0:.2f}".format(i),
              "\tguangzhou_Sens: {0:.2f}".format(guangzhou_sens),
              "\tguangzhou_Spec: {0:.2f}".format(guangzhou_spec),
              "\tguangzhou_Acc: {0:.2f}".format(guangzhou_accuracy_calculated),
              '\tguangzhou_Weighted Acc: {0:.2f}'.format(guangzhou_weighted_acc),
              "\tguangzhou_AUC: {0:.2f}".format(guangzhou_AUC_ROC))
else:
        for i in np.linspace(0, 1, 21):
            # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

            external_sens, external_spec, external_accuracy_calculated, external_weighted_acc, external_AUC_ROC \
                = compute_metrics(y_test, pre_y, thres_val=i)
            print("Threshold: {0:.2f}".format(i),
                  "\texternal_Sens: {0:.2f}".format(external_sens),
                  "\texternal_Spec: {0:.2f}".format(external_spec),
                  "\texternal_Acc: {0:.2f}".format(external_accuracy_calculated),
                  '\texternal_Weighted Acc: {0:.2f}'.format(external_weighted_acc),
                  "\texternal_AUC: {0:.2f}".format(external_AUC_ROC))


params = {
    "C": [1000, 1,0.1],
    "kernel": ["rbf","linear"],
    "gamma":[0.001,0.1],
    # "C":[1e-5,1e-3,0.1,1,10,100],
    # "kernel":["linear", "poly", "rbf"],
    # "gamma":[0.0001,0.1,10,50,100]
}
# "sigmoid"
svm = SVC(probability=True)

grid = GridSearchCV(svm, params,scoring='balanced_accuracy', cv=5, n_jobs=4, verbose=1)
grid.fit(X_train, y_train)
print(grid.best_estimator_)
# y_val_pred = grid.best_estimator_.predict(X_test)
time1 = time.time()
pre_y = grid.best_estimator_.predict_proba(X_test)[:, 1]
svm_best = grid.best_estimator_
time2 = time.time()
print('SVM data cost {}'.format((time2-time1)/pre_y.shape[0]))
if cross_valid:
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        external_sens, external_spec, external_accuracy_calculated,external_weighted_acc,external_AUC_ROC\
            = compute_metrics(y_test, pre_y, thres_val=i)
        print("Threshold: {0:.2f}".format(i),
              "\texternal_Sens: {0:.2f}".format(external_sens),
              "\texternal_Spec: {0:.2f}".format(external_spec),
              "\texternal_Acc: {0:.2f}".format(external_accuracy_calculated),
              '\texternal_Weighted Acc: {0:.2f}'.format(external_weighted_acc),
              "\texternal_AUC: {0:.2f}".format(external_AUC_ROC))
    print("Best model on perth dataset:")
    pre_perth = svm_best.predict_proba(X_perth)[:, 1]
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        perth_sens, perth_spec, perth_accuracy_calculated,perth_weighted_acc,perth_AUC_ROC\
            = compute_metrics(y_perth, pre_perth, thres_val=i)
        print("Threshold: {0:.2f}".format(i),
              "\tperth_Sens: {0:.2f}".format(perth_sens),
              "\tperth_Spec: {0:.2f}".format(perth_spec),
              "\tperth_Acc: {0:.2f}".format(perth_accuracy_calculated),
              '\tperth_Weighted Acc: {0:.2f}'.format(perth_weighted_acc),
              "\tperth_AUC: {0:.2f}".format(perth_AUC_ROC))
    print("Best model on malaysia dataset:")
    pre_Malaysia = svm_best.predict_proba(X_Malaysia)[:, 1]
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        malaysia_sens, malaysia_spec, malaysia_accuracy_calculated, malaysia_weighted_acc, malaysia_AUC_ROC \
            = compute_metrics(y_Malaysia, pre_Malaysia, thres_val=i)
        print("Threshold: {0:.2f}".format(i),
              "\tmalaysia_Sens: {0:.2f}".format(malaysia_sens),
              "\tmalaysia_Spec: {0:.2f}".format(malaysia_spec),
              "\tmalaysia_Acc: {0:.2f}".format(malaysia_accuracy_calculated),
              '\tmalaysia_Weighted Acc: {0:.2f}'.format(malaysia_weighted_acc),
              "\tmalaysia_AUC: {0:.2f}".format(malaysia_AUC_ROC))

    print("Best model on chongqing dataset:")
    pre_chongqing = svm_best.predict_proba(X_chongqing)[:, 1]
    chongqing_weight_arr = []
    chongqing_thres_arr = []
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        chongqing_sens, chongqing_spec, chongqing_accuracy_calculated, chongqing_weighted_acc, chongqing_AUC_ROC \
            = compute_metrics(y_chongqing, pre_chongqing, thres_val=i)
        chongqing_weight_arr.append(chongqing_weighted_acc)
        chongqing_thres_arr.append(i)
        print("Threshold: {0:.2f}".format(i),
              "\tchongqing_Sens: {0:.2f}".format(chongqing_sens),
              "\tchongqing_Spec: {0:.2f}".format(chongqing_spec),
              "\tchongqing_Acc: {0:.2f}".format(chongqing_accuracy_calculated),
              '\tchongqing_Weighted Acc: {0:.2f}'.format(chongqing_weighted_acc),
              "\tchongqing_AUC: {0:.2f}".format(chongqing_AUC_ROC))

    print("Best model on GUANGZHOU dataset:")
    pre_guangzhou = svm_best.predict_proba(X_China)[:, 1]
    guangzhou_weight_arr = []
    guangzhou_thres_arr = []
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        guangzhou_sens, guangzhou_spec, guangzhou_accuracy_calculated, guangzhou_weighted_acc, guangzhou_AUC_ROC \
            = compute_metrics(y_China, pre_guangzhou, thres_val=i)
        guangzhou_weight_arr.append(guangzhou_weighted_acc)
        guangzhou_thres_arr.append(i)
        print("Threshold: {0:.2f}".format(i),
              "\tguangzhou_Sens: {0:.2f}".format(guangzhou_sens),
              "\tguangzhou_Spec: {0:.2f}".format(guangzhou_spec),
              "\tguangzhou_Acc: {0:.2f}".format(guangzhou_accuracy_calculated),
              '\tguangzhou_Weighted Acc: {0:.2f}'.format(guangzhou_weighted_acc),
              "\tguangzhou_AUC: {0:.2f}".format(guangzhou_AUC_ROC))
else:
        for i in np.linspace(0, 1, 21):
            # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

            external_sens, external_spec, external_accuracy_calculated, external_weighted_acc, external_AUC_ROC \
                = compute_metrics(y_test, pre_y, thres_val=i)
            print("Threshold: {0:.2f}".format(i),
                  "\texternal_Sens: {0:.2f}".format(external_sens),
                  "\texternal_Spec: {0:.2f}".format(external_spec),
                  "\texternal_Acc: {0:.2f}".format(external_accuracy_calculated),
                  '\texternal_Weighted Acc: {0:.2f}'.format(external_weighted_acc),
                  "\texternal_AUC: {0:.2f}".format(external_AUC_ROC))

LogReg = LogisticRegression()

LG_param_grid = {'random_state': [3, 5, 7, 10], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [1000]}

gsLG = GridSearchCV(LogReg, param_grid=LG_param_grid, cv=5, scoring='balanced_accuracy',
                    n_jobs=4, verbose=1)

gsLG.fit(X_train, y_train)
LG_best = gsLG.best_estimator_

print(LG_best)
# print(gsLG.best_score_)

y_val_pred = LG_best.predict(X_test)

time1 = time.time()
pre_y = LG_best.predict_proba(X_test)[:, 1]
time2 = time.time()
print('LG data cost {}'.format((time2-time1)/pre_y.shape[0]))
if cross_valid:
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        external_sens, external_spec, external_accuracy_calculated,external_weighted_acc,external_AUC_ROC\
            = compute_metrics(y_test, pre_y, thres_val=i)
        print("Threshold: {0:.2f}".format(i),
              "\texternal_Sens: {0:.2f}".format(external_sens),
              "\texternal_Spec: {0:.2f}".format(external_spec),
              "\texternal_Acc: {0:.2f}".format(external_accuracy_calculated),
              '\texternal_Weighted Acc: {0:.2f}'.format(external_weighted_acc),
              "\texternal_AUC: {0:.2f}".format(external_AUC_ROC))
    print("Best model on perth dataset:")
    pre_perth = LG_best.predict_proba(X_perth)[:, 1]
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        perth_sens, perth_spec, perth_accuracy_calculated,perth_weighted_acc,perth_AUC_ROC\
            = compute_metrics(y_perth, pre_perth, thres_val=i)
        print("Threshold: {0:.2f}".format(i),
              "\tperth_Sens: {0:.2f}".format(perth_sens),
              "\tperth_Spec: {0:.2f}".format(perth_spec),
              "\tperth_Acc: {0:.2f}".format(perth_accuracy_calculated),
              '\tperth_Weighted Acc: {0:.2f}'.format(perth_weighted_acc),
              "\tperth_AUC: {0:.2f}".format(perth_AUC_ROC))
    print("Best model on malaysia dataset:")
    pre_Malaysia = LG_best.predict_proba(X_Malaysia)[:, 1]
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        malaysia_sens, malaysia_spec, malaysia_accuracy_calculated, malaysia_weighted_acc, malaysia_AUC_ROC \
            = compute_metrics(y_Malaysia, pre_Malaysia, thres_val=i)
        print("Threshold: {0:.2f}".format(i),
              "\tmalaysia_Sens: {0:.2f}".format(malaysia_sens),
              "\tmalaysia_Spec: {0:.2f}".format(malaysia_spec),
              "\tmalaysia_Acc: {0:.2f}".format(malaysia_accuracy_calculated),
              '\tmalaysia_Weighted Acc: {0:.2f}'.format(malaysia_weighted_acc),
              "\tmalaysia_AUC: {0:.2f}".format(malaysia_AUC_ROC))

    print("Best model on chongqing dataset:")
    pre_chongqing = LG_best.predict_proba(X_chongqing)[:, 1]
    chongqing_weight_arr = []
    chongqing_thres_arr = []
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        chongqing_sens, chongqing_spec, chongqing_accuracy_calculated, chongqing_weighted_acc, chongqing_AUC_ROC \
            = compute_metrics(y_chongqing, pre_chongqing, thres_val=i)
        chongqing_weight_arr.append(chongqing_weighted_acc)
        chongqing_thres_arr.append(i)
        print("Threshold: {0:.2f}".format(i),
              "\tchongqing_Sens: {0:.2f}".format(chongqing_sens),
              "\tchongqing_Spec: {0:.2f}".format(chongqing_spec),
              "\tchongqing_Acc: {0:.2f}".format(chongqing_accuracy_calculated),
              '\tchongqing_Weighted Acc: {0:.2f}'.format(chongqing_weighted_acc),
              "\tchongqing_AUC: {0:.2f}".format(chongqing_AUC_ROC))

    print("Best model on GUANGZHOU dataset:")
    pre_guangzhou = LG_best.predict_proba(X_China)[:, 1]
    guangzhou_weight_arr = []
    guangzhou_thres_arr = []
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        guangzhou_sens, guangzhou_spec, guangzhou_accuracy_calculated, guangzhou_weighted_acc, guangzhou_AUC_ROC \
            = compute_metrics(y_China, pre_guangzhou, thres_val=i)
        guangzhou_weight_arr.append(guangzhou_weighted_acc)
        guangzhou_thres_arr.append(i)
        print("Threshold: {0:.2f}".format(i),
              "\tguangzhou_Sens: {0:.2f}".format(guangzhou_sens),
              "\tguangzhou_Spec: {0:.2f}".format(guangzhou_spec),
              "\tguangzhou_Acc: {0:.2f}".format(guangzhou_accuracy_calculated),
              '\tguangzhou_Weighted Acc: {0:.2f}'.format(guangzhou_weighted_acc),
              "\tguangzhou_AUC: {0:.2f}".format(guangzhou_AUC_ROC))
else:
        for i in np.linspace(0, 1, 21):
            # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

            external_sens, external_spec, external_accuracy_calculated, external_weighted_acc, external_AUC_ROC \
                = compute_metrics(y_test, pre_y, thres_val=i)
            print("Threshold: {0:.2f}".format(i),
                  "\texternal_Sens: {0:.2f}".format(external_sens),
                  "\texternal_Spec: {0:.2f}".format(external_spec),
                  "\texternal_Acc: {0:.2f}".format(external_accuracy_calculated),
                  '\texternal_Weighted Acc: {0:.2f}'.format(external_weighted_acc),
                  "\texternal_AUC: {0:.2f}".format(external_AUC_ROC))
#
RandFor = RandomForestClassifier()

RF_param_frid = {"max_depth": [20], "max_features": [1,10],
                 "min_samples_split": [2, 10, 20], "min_samples_leaf": [1,10,20],
                 "bootstrap": [False], "n_estimators": [5000], "criterion": ["gini", "entropy"]}

gsRandFor = GridSearchCV(RandFor, param_grid=RF_param_frid, cv=5, scoring='balanced_accuracy',
                         n_jobs=4, verbose=1)

gsRandFor.fit(X_train, y_train)
RF_best = gsRandFor.best_estimator_

print(RF_best)
# print(gsRandFor.best_score_)
y_val_pred = RF_best.predict(X_test)

time1 = time.time()
pre_y = RF_best.predict_proba(X_test)[:, 1]
time2 = time.time()
print('RF data cost {}'.format((time2-time1)/pre_y.shape[0]))
if cross_valid:
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        external_sens, external_spec, external_accuracy_calculated,external_weighted_acc,external_AUC_ROC\
            = compute_metrics(y_test, pre_y, thres_val=i)
        print("Threshold: {0:.2f}".format(i),
              "\texternal_Sens: {0:.2f}".format(external_sens),
              "\texternal_Spec: {0:.2f}".format(external_spec),
              "\texternal_Acc: {0:.2f}".format(external_accuracy_calculated),
              '\texternal_Weighted Acc: {0:.2f}'.format(external_weighted_acc),
              "\texternal_AUC: {0:.2f}".format(external_AUC_ROC))
    print("Best model on perth dataset:")
    pre_perth = RF_best.predict_proba(X_perth)[:, 1]
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        perth_sens, perth_spec, perth_accuracy_calculated,perth_weighted_acc,perth_AUC_ROC\
            = compute_metrics(y_perth, pre_perth, thres_val=i)
        print("Threshold: {0:.2f}".format(i),
              "\tperth_Sens: {0:.2f}".format(perth_sens),
              "\tperth_Spec: {0:.2f}".format(perth_spec),
              "\tperth_Acc: {0:.2f}".format(perth_accuracy_calculated),
              '\tperth_Weighted Acc: {0:.2f}'.format(perth_weighted_acc),
              "\tperth_AUC: {0:.2f}".format(perth_AUC_ROC))
    print("Best model on malaysia dataset:")
    pre_Malaysia = RF_best.predict_proba(X_Malaysia)[:, 1]
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        malaysia_sens, malaysia_spec, malaysia_accuracy_calculated, malaysia_weighted_acc, malaysia_AUC_ROC \
            = compute_metrics(y_Malaysia, pre_Malaysia, thres_val=i)
        print("Threshold: {0:.2f}".format(i),
              "\tmalaysia_Sens: {0:.2f}".format(malaysia_sens),
              "\tmalaysia_Spec: {0:.2f}".format(malaysia_spec),
              "\tmalaysia_Acc: {0:.2f}".format(malaysia_accuracy_calculated),
              '\tmalaysia_Weighted Acc: {0:.2f}'.format(malaysia_weighted_acc),
              "\tmalaysia_AUC: {0:.2f}".format(malaysia_AUC_ROC))

    print("Best model on chongqing dataset:")
    pre_chongqing = RF_best.predict_proba(X_chongqing)[:, 1]
    chongqing_weight_arr = []
    chongqing_thres_arr = []
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        chongqing_sens, chongqing_spec, chongqing_accuracy_calculated, chongqing_weighted_acc, chongqing_AUC_ROC \
            = compute_metrics(y_chongqing, pre_chongqing, thres_val=i)
        chongqing_weight_arr.append(chongqing_weighted_acc)
        chongqing_thres_arr.append(i)
        print("Threshold: {0:.2f}".format(i),
              "\tchongqing_Sens: {0:.2f}".format(chongqing_sens),
              "\tchongqing_Spec: {0:.2f}".format(chongqing_spec),
              "\tchongqing_Acc: {0:.2f}".format(chongqing_accuracy_calculated),
              '\tchongqing_Weighted Acc: {0:.2f}'.format(chongqing_weighted_acc),
              "\tchongqing_AUC: {0:.2f}".format(chongqing_AUC_ROC))

    print("Best model on GUANGZHOU dataset:")
    pre_guangzhou = RF_best.predict_proba(X_China)[:, 1]
    guangzhou_weight_arr = []
    guangzhou_thres_arr = []
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        guangzhou_sens, guangzhou_spec, guangzhou_accuracy_calculated, guangzhou_weighted_acc, guangzhou_AUC_ROC \
            = compute_metrics(y_China, pre_guangzhou, thres_val=i)
        guangzhou_weight_arr.append(guangzhou_weighted_acc)
        guangzhou_thres_arr.append(i)
        print("Threshold: {0:.2f}".format(i),
              "\tguangzhou_Sens: {0:.2f}".format(guangzhou_sens),
              "\tguangzhou_Spec: {0:.2f}".format(guangzhou_spec),
              "\tguangzhou_Acc: {0:.2f}".format(guangzhou_accuracy_calculated),
              '\tguangzhou_Weighted Acc: {0:.2f}'.format(guangzhou_weighted_acc),
              "\tguangzhou_AUC: {0:.2f}".format(guangzhou_AUC_ROC))
else:
        for i in np.linspace(0, 1, 21):
            # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

            external_sens, external_spec, external_accuracy_calculated, external_weighted_acc, external_AUC_ROC \
                = compute_metrics(y_test, pre_y, thres_val=i)
            print("Threshold: {0:.2f}".format(i),
                  "\texternal_Sens: {0:.2f}".format(external_sens),
                  "\texternal_Spec: {0:.2f}".format(external_spec),
                  "\texternal_Acc: {0:.2f}".format(external_accuracy_calculated),
                  '\texternal_Weighted Acc: {0:.2f}'.format(external_weighted_acc),
                  "\texternal_AUC: {0:.2f}".format(external_AUC_ROC))
#
# DTC = DecisionTreeClassifier()
# adaDTC = AdaBoostClassifier(DTC, random_state=7)
# ada_param_grid = {"base_estimator__criterion": ["gini", "entropy"],
#                   # "base_estimator__splitter": ["best", "random"],
#                   # "algorithm": ["SAMME", "SAMME.R"],
#                   "n_estimators": [1,50,100, 500,1000,2000],
#                   'learning_rate': [1.5, 0.5, 0.25, 0.01, 0.001]}
#
# gsadaDTC = GridSearchCV(adaDTC, param_grid=ada_param_grid, cv=5,
#                         scoring='balanced_accuracy', n_jobs=4, verbose=1)
#
# gsadaDTC.fit(X_train, y_train)
# ada_best = gsadaDTC.best_estimator_
# y_val_pred = ada_best.predict(X_test)
# # print(y_val_pred)
# time1 = time.time()
# pre_y = ada_best.predict_proba(X_test)[:, 1]
# time2 = time.time()
# print('ADAboost data cost {}'.format((time2-time1)/pre_y.shape[0]))
#
# if cross_valid:
#     for i in np.linspace(0, 1, 21):
#         # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01
#
#         external_sens, external_spec, external_accuracy_calculated,external_weighted_acc,external_AUC_ROC\
#             = compute_metrics(y_test, pre_y, thres_val=i)
#         print("Threshold: {0:.2f}".format(i),
#               "\texternal_Sens: {0:.2f}".format(external_sens),
#               "\texternal_Spec: {0:.2f}".format(external_spec),
#               "\texternal_Acc: {0:.2f}".format(external_accuracy_calculated),
#               '\texternal_Weighted Acc: {0:.2f}'.format(external_weighted_acc),
#               "\texternal_AUC: {0:.2f}".format(external_AUC_ROC))
#     print("Best model on perth dataset:")
#     pre_perth = ada_best.predict_proba(X_perth)[:, 1]
#     for i in np.linspace(0, 1, 21):
#         # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01
#
#         perth_sens, perth_spec, perth_accuracy_calculated,perth_weighted_acc,perth_AUC_ROC\
#             = compute_metrics(y_perth, pre_perth, thres_val=i)
#         print("Threshold: {0:.2f}".format(i),
#               "\tperth_Sens: {0:.2f}".format(perth_sens),
#               "\tperth_Spec: {0:.2f}".format(perth_spec),
#               "\tperth_Acc: {0:.2f}".format(perth_accuracy_calculated),
#               '\tperth_Weighted Acc: {0:.2f}'.format(perth_weighted_acc),
#               "\tperth_AUC: {0:.2f}".format(perth_AUC_ROC))
#     print("Best model on malaysia dataset:")
#     pre_Malaysia = ada_best.predict_proba(X_Malaysia)[:, 1]
#     for i in np.linspace(0, 1, 21):
#         # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01
#
#         malaysia_sens, malaysia_spec, malaysia_accuracy_calculated, malaysia_weighted_acc, malaysia_AUC_ROC \
#             = compute_metrics(y_Malaysia, pre_Malaysia, thres_val=i)
#         print("Threshold: {0:.2f}".format(i),
#               "\tmalaysia_Sens: {0:.2f}".format(malaysia_sens),
#               "\tmalaysia_Spec: {0:.2f}".format(malaysia_spec),
#               "\tmalaysia_Acc: {0:.2f}".format(malaysia_accuracy_calculated),
#               '\tmalaysia_Weighted Acc: {0:.2f}'.format(malaysia_weighted_acc),
#               "\tmalaysia_AUC: {0:.2f}".format(malaysia_AUC_ROC))
#
#     print("Best model on chongqing dataset:")
#     pre_chongqing = ada_best.predict_proba(X_chongqing)[:, 1]
#     chongqing_weight_arr = []
#     chongqing_thres_arr = []
#     for i in np.linspace(0, 1, 21):
#         # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01
#
#         chongqing_sens, chongqing_spec, chongqing_accuracy_calculated, chongqing_weighted_acc, chongqing_AUC_ROC \
#             = compute_metrics(y_chongqing, pre_chongqing, thres_val=i)
#         chongqing_weight_arr.append(chongqing_weighted_acc)
#         chongqing_thres_arr.append(i)
#         print("Threshold: {0:.2f}".format(i),
#               "\tchongqing_Sens: {0:.2f}".format(chongqing_sens),
#               "\tchongqing_Spec: {0:.2f}".format(chongqing_spec),
#               "\tchongqing_Acc: {0:.2f}".format(chongqing_accuracy_calculated),
#               '\tchongqing_Weighted Acc: {0:.2f}'.format(chongqing_weighted_acc),
#               "\tchongqing_AUC: {0:.2f}".format(chongqing_AUC_ROC))
#
#     print("Best model on GUANGZHOU dataset:")
#     pre_guangzhou = ada_best.predict_proba(X_China)[:, 1]
#     guangzhou_weight_arr = []
#     guangzhou_thres_arr = []
#     for i in np.linspace(0, 1, 21):
#         # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01
#
#         guangzhou_sens, guangzhou_spec, guangzhou_accuracy_calculated, guangzhou_weighted_acc, guangzhou_AUC_ROC \
#             = compute_metrics(y_China, pre_guangzhou, thres_val=i)
#         guangzhou_weight_arr.append(guangzhou_weighted_acc)
#         guangzhou_thres_arr.append(i)
#         print("Threshold: {0:.2f}".format(i),
#               "\tguangzhou_Sens: {0:.2f}".format(guangzhou_sens),
#               "\tguangzhou_Spec: {0:.2f}".format(guangzhou_spec),
#               "\tguangzhou_Acc: {0:.2f}".format(guangzhou_accuracy_calculated),
#               '\tguangzhou_Weighted Acc: {0:.2f}'.format(guangzhou_weighted_acc),
#               "\tguangzhou_AUC: {0:.2f}".format(guangzhou_AUC_ROC))
#     else:
#         for i in np.linspace(0, 1, 21):
#             # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01
#
#             external_sens, external_spec, external_accuracy_calculated, external_weighted_acc, external_AUC_ROC \
#                 = compute_metrics(y_test, pre_y, thres_val=i)
#             print("Threshold: {0:.2f}".format(i),
#                   "\texternal_Sens: {0:.2f}".format(external_sens),
#                   "\texternal_Spec: {0:.2f}".format(external_spec),
#                   "\texternal_Acc: {0:.2f}".format(external_accuracy_calculated),
#                   '\texternal_Weighted Acc: {0:.2f}'.format(external_weighted_acc),
#                   "\texternal_AUC: {0:.2f}".format(external_AUC_ROC))
# ## 可视化在验证集上的Roc曲线
#
# print(gsadaDTC.best_score_)
# print(ada_best)
# # print(gsadaDTC.best_score_)

# # 2
# RandFor = RandomForestClassifier()
#
# RF_param_frid = {"max_depth": [500], "max_features": [1],
#                  "min_samples_split": [10], "min_samples_leaf": [1, 3, 10,20],
#                  "bootstrap": [False], "n_estimators": [1000], "criterion": ["gini", "entropy"]}
#
# gsRandFor = GridSearchCV(RandFor, param_grid=RF_param_frid, cv=5, scoring='balanced_accuracy',
#                          n_jobs=4, verbose=1)
#
# gsRandFor.fit(X_train, y_train)
# RF_best = gsRandFor.best_estimator_
#
# print(RF_best)
# # print(gsRandFor.best_score_)
# y_val_pred = RF_best.predict(X_test)
# pre_y = RF_best.predict_proba(X_test)[:, 1]
# for i in np.linspace(0, 1, 21):
#     # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01
#
#     external_sens, external_spec, external_accuracy_calculated,external_weighted_acc,external_AUC_ROC\
#         = compute_metrics(y_test, pre_y, thres_val=i)
#     print("Threshold: {0:.2f}".format(i),
#           "\texternal_Sens: {0:.2f}".format(external_sens),
#           "\texternal_Spec: {0:.2f}".format(external_spec),
#           "\texternal_Acc: {0:.2f}".format(external_accuracy_calculated),
#           '\texternal_Weighted Acc: {0:.2f}'.format(external_weighted_acc),
#           "\texternal_AUC: {0:.2f}".format(external_AUC_ROC))


# # 3
# GBC = GradientBoostingClassifier()
# gb_param_grid = {'loss': ["deviance"], 'n_estimators': [1000], 'learning_rate': [5,1, 0.1, 0.05, 0.01],
#                  'max_depth': [8, 10, 20], 'min_samples_leaf': [20], 'max_features': [0.6]}
#
# gsGBC = GridSearchCV(GBC, param_grid=gb_param_grid, cv=5, scoring='balanced_accuracy',
#                      n_jobs=4, verbose=1)
#
# gsGBC.fit(X_train, y_train)
# GBC_best = gsGBC.best_estimator_
#
# print(GBC_best)
# # print(gsGBC.best_score_)
# y_val_pred = GBC_best.predict(X_test)
# pre_y = GBC_best.predict_proba(X_test)[:, 1]
# for i in np.linspace(0, 1, 21):
#     # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01
#
#     external_sens, external_spec, external_accuracy_calculated,external_weighted_acc,external_AUC_ROC\
#         = compute_metrics(y_test, pre_y, thres_val=i)
#     print("Threshold: {0:.2f}".format(i),
#           "\texternal_Sens: {0:.2f}".format(external_sens),
#           "\texternal_Spec: {0:.2f}".format(external_spec),
#           "\texternal_Acc: {0:.2f}".format(external_accuracy_calculated),
#           '\texternal_Weighted Acc: {0:.2f}'.format(external_weighted_acc),
#           "\texternal_AUC: {0:.2f}".format(external_AUC_ROC))
#
# # 5
# MLPC = MLPClassifier()
#
# mlp_param_grid = {'solver': ['sgd', 'adam', 'lbfgs'], 'activation': ['relu', 'tanh'],
#                   'hidden_layer_sizes': [100,  200, 50, 500], 'max_iter': [20000]}
#
# gsMLP = GridSearchCV(MLPC, param_grid=mlp_param_grid, cv=5, scoring='balanced_accuracy',
#                      n_jobs=-1, verbose=1)
#
# gsMLP.fit(X_train, y_train)
# MLP_best = gsMLP.best_estimator_
#
# print(MLP_best)
# print(gsMLP.best_score_)
# y_val_pred = MLP_best.predict(X_test)
# pre_y = MLP_best.predict_proba(X_test)[:, 1]
# for i in np.linspace(0, 1, 21):
#     # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01
#
#     external_sens, external_spec, external_accuracy_calculated,external_weighted_acc,external_AUC_ROC\
#         = compute_metrics(y_test, pre_y, thres_val=i)
#     print("Threshold: {0:.2f}".format(i),
#           "\texternal_Sens: {0:.2f}".format(external_sens),
#           "\texternal_Spec: {0:.2f}".format(external_spec),
#           "\texternal_Acc: {0:.2f}".format(external_accuracy_calculated),
#           '\texternal_Weighted Acc: {0:.2f}'.format(external_weighted_acc),
#           "\texternal_AUC: {0:.2f}".format(external_AUC_ROC))