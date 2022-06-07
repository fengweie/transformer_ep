
#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import cv2
from sklearn.metrics import roc_curve
import matplotlib

matplotlib.use('Agg')
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

params = {'legend.fontsize': 13,
          'axes.labelsize': 15,
          'axes.titlesize': 15,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15}  # define pyplot parameters
pylab.rcParams.update(params)
import seaborn as sns
from imblearn.over_sampling import SMOTE,KMeansSMOTE,SVMSMOTE,ADASYN,BorderlineSMOTE
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import argparse
from torch.optim.lr_scheduler import StepLR
from utils import *
from sklearn.metrics import roc_auc_score
import random
import time
from sklearn.metrics import confusion_matrix
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset
import pickle
from non_local import NONLocalBlock1D
import torch
import torch.nn as nn
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
    print("95ci auc", auc_scores.mean(), np.median(auc_scores), auc_scores[int(0.05 * len(auc_scores))],
          auc_scores[int(0.95 * len(auc_scores))])
    print("95ci acc", acc_scores.mean(), np.median(acc_scores), acc_scores[int(0.05 * len(acc_scores))],
          acc_scores[int(0.95 * len(acc_scores))])
    print("95ci sp", sp_scores.mean(), np.median(sp_scores), sp_scores[int(0.05 * len(sp_scores))],
          sp_scores[int(0.95 * len(sp_scores))])
    print("95ci se", se_scores.mean(), np.median(se_scores), se_scores[int(0.05 * len(se_scores))],
          se_scores[int(0.95 * len(se_scores))])
    print("95ci precision", precision_scores.mean(), np.median(precision_scores),
          precision_scores[int(0.05 * len(precision_scores))], precision_scores[int(0.95 * len(precision_scores))])


class Transformer(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self,input_dim):
        super(Transformer, self).__init__()  #
        self.fc1 = torch.nn.Linear(input_dim, 300)  # 第一个隐含层
        self.nl_1 = NONLocalBlock1D(in_channels=1)
        self.fc2 = torch.nn.Linear(300, 300)  # 第二个隐含层
        self.nl_2 = NONLocalBlock1D(in_channels=1)
        self.fc3 = torch.nn.Linear(300, 300)  # 第二个隐含层
        self.nl_3 = NONLocalBlock1D(in_channels=1)
        self.fc4= torch.nn.Linear(300, 300)  # 第二个隐含层
        self.nl_4 = NONLocalBlock1D(in_channels=1)
        self.fc5 = torch.nn.Linear(300, 300)  # 第二个隐含层
        self.nl_5 = NONLocalBlock1D(in_channels=1)
        # self.fc4 = torch.nn.Linear(1000, 1000)  # 第二个隐含层
        self.fc = torch.nn.Linear(300, 1)  # 输出层
        self.BN1 = nn.BatchNorm1d(300)
        self.BN2 = nn.BatchNorm1d(300)
        self.BN3 = nn.BatchNorm1d(300)
        self.BN4 = nn.BatchNorm1d(300)
        self.BN5 = nn.BatchNorm1d(300)
        self.dr = nn.Dropout(0.003)
        self.ac = nn.Sigmoid()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.BatchNorm1d):
                # print("=====================================>初始化bn层")
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, din):
        nl_feature_0, nl_map_0 = self.nl_1(din.unsqueeze(1), return_nl_map=True)
        feature_1 = self.dr(F.relu(self.BN1(self.fc1(nl_feature_0.squeeze(1)))))  #
        nl_feature_1, nl_map_1 = self.nl_1(feature_1.unsqueeze(1), return_nl_map=True)
        feature_2 = self.dr(F.relu(self.BN2(self.fc2(nl_feature_1.squeeze(1)))))
        # # nl_feature_2, nl_map_2 = self.nl_2(feature_2.unsqueeze(1), return_nl_map=True)
        # # feature_3 = self.dr(F.relu(self.BN3(self.fc3(nl_feature_2.squeeze(1))))) #
        # # nl_feature_3, nl_map_3 = self.nl_3(feature_3.unsqueeze(1), return_nl_map=True)
        # # #
        # # feature_4 = self.dr(F.relu(self.BN4(self.fc4(nl_feature_3.squeeze(1)))))
        # # nl_feature_4, nl_map_4 = self.nl_4(feature_4.unsqueeze(1), return_nl_map=True)
        # # feature_5 = self.dr(F.relu(self.BN5(self.fc5(nl_feature_4.squeeze(1))))) #
        # # nl_feature_5, nl_map_5 = self.nl_5(feature_5.unsqueeze(1), return_nl_map=True)
        # dout = self.fc(nl_feature_1.squeeze(1))  #
        dout = self.ac(self.fc(feature_2))  #
        return dout, nl_map_0
def test_model_thres(model, test_loader, thres_val=0.5,trg_label=1):
    # loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, (batch, labels) in enumerate(test_loader):
            src = batch.type(torch.float32)  # Turn into a batch
            time1 = time.time()
            preds, nl_map_0 = model(src.cuda())
            preds_after = preds[:].squeeze()
            # preds_soft = F.softmax(preds, dim=-1)
            time2 = time.time()
            # print('test data cost {}'.format((time2 - time1)/preds.shape[0]))
            # preds_after = preds_soft[:,1:].squeeze().cpu()
            ## Aggregate losses and their gradients based off label (for class imbalance)
            if (trg_label == 1):
                loss = F.binary_cross_entropy(preds_after, labels.cuda())
                # loss = loss_fn(preds, labels.type(torch.LongTensor).cuda())
            all_preds.append(preds_after)
            all_labels.append(labels)

            total_loss += loss.data

    real_results = torch.cat(all_labels, dim=0).flatten().type(torch.float32).cpu().numpy()  ## Flat tensor
    pred_results = torch.cat(all_preds, dim=0).flatten().type(torch.float32).cpu().detach().numpy()
    print(real_results.shape, pred_results.shape)
    test_AUC_ROC = roc_auc_score(real_results, pred_results)
    pred_res = pred_results>=thres_val

    TP = sum((pred_res == 1) & (real_results == 1))
    FN = sum((pred_res == 0) & (real_results == 1))
    TN = sum((pred_res == 0) & (real_results == 0))
    FP = sum((pred_res == 1) & (real_results == 0))
    test_sens = TP / (TP + FN)
    test_spec = TN / (TN + FP)
    test_accuracy_calculated = (TN + TP) / (TN + TP + FP + FN)
    test_weighted_acc = (TN / (FP + TN)) * 0.5 + (TP / (TP + FN)) * 0.5
    auc_confint_cal(real_results, pred_results, pred_res)
    return test_sens, test_spec, test_accuracy_calculated,test_weighted_acc,test_AUC_ROC, \
           total_loss/len(test_loader),real_results, pred_results
def save_nl_map(vis_dataLoader,net,use_age_category):
    itervis_dataLoader = iter(vis_dataLoader)
    img_batch, label_batch = itervis_dataLoader.__next__()
    img_batch = img_batch.cuda()
    # label_batch = label_batch.cuda()

    torch.set_grad_enabled(False)
    net.eval()

    _, nl_mep = net(img_batch)

    # (b, h1*w1, h2*w2)
    nl_map_1 = nl_mep.cpu().numpy()
    nl_map_1 = nl_map_1.mean(0)

    # define labels
    if use_age_category:
        labels = ['sex', 'age_init_29', 'age_init_46', 'age_init_>46',
           'pretrt_sz_5', 'focal', 'fam_hx', 'febrile', 'ci', 'birth_t', 'head',
           'drug', 'alcohol', 'cvd', 'psy', 'ld', 'lesion_A', 'lesion_E',
           'lesion_N', 'eeg_cat_A', 'eeg_cat_E', 'eeg_cat_N', 'CBZ', 'LEV', 'LTG',
           'OXC', 'PHT', 'TPM', 'VPA']
    else:
        labels = ['sex', 'age_init',
           'pretrt_sz_5', 'focal', 'fam_hx', 'febrile', 'ci', 'birth_t', 'head',
           'drug', 'alcohol', 'cvd', 'psy', 'ld', 'lesion_A', 'lesion_E',
           'lesion_N', 'eeg_cat_A', 'eeg_cat_E', 'eeg_cat_N', 'CBZ', 'LEV', 'LTG',
           'OXC', 'PHT', 'TPM', 'VPA']
    # labels = ['sex', 'age_init_29', 'age_init_46', 'age_init_>46',
    #    'pretrt_sz_5', 'focal', 'fam_hx', 'febrile', 'ci', 'birth_t', 'head',
    #    'drug', 'alcohol', 'cvd', 'psy', 'ld', 'lesion_A', 'lesion_E',
    #    'lesion_N', 'eeg_cat_A', 'eeg_cat_E', 'eeg_cat_N']
    sns.set(color_codes=True)
    plt.figure(1, figsize=(12, 9))

    # plt.title("Confusion Matrix")

    sns.set(font_scale=1.4)
    ax = sns.heatmap(nl_map_1, annot=False, cmap="YlGnBu", cbar_kws={'label': 'Scale'}, fmt='.20g')

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    # ax.set(ylabel='True label', xlabel='Predicted label')
    plt.savefig('nl_map_vis/nl_map_1.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    torch.set_grad_enabled(True)
    net.train()
train_arg_parser = argparse.ArgumentParser(description="parser")
train_arg_parser.add_argument('--seed', type=int, default=3424, metavar='S',
                    help='random seed (default: 1)')
args = train_arg_parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
TRAIN_RANDOM_SEED = args.seed
print("training seed used:", TRAIN_RANDOM_SEED)
cudnn.enabled = True
torch.manual_seed(TRAIN_RANDOM_SEED)
torch.cuda.manual_seed(TRAIN_RANDOM_SEED)
torch.cuda.manual_seed_all(TRAIN_RANDOM_SEED)  # 为所有GPU设置随机种子
np.random.seed(TRAIN_RANDOM_SEED)
random.seed(TRAIN_RANDOM_SEED)
# os.environ['CUDA_VISIBLE_DEVICES'] = str(4)
# perth_val = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64  ## Because we have a small data set, keep it as 1
# # ### Making the dataset class for reading
sm = SMOTE(random_state=42)
# sm = SVMSMOTE(random_state=42)
# sm = ADASYN(random_state=42)
# sm = BorderlineSMOTE(random_state=42)
# sm = KMeansSMOTE(random_state=42)

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
# drugs_to_keep = ['pid', 'cohort','sex', 'age_init_29', 'age_init_46', 'age_init_>46',
#  'pretrt_sz_5', 'focal', 'fam_hx', 'febrile', 'ci', 'birth_t', 'head',
#  'drug', 'alcohol', 'cvd', 'psy', 'ld', 'lesion_A', 'lesion_E',
#   'lesion_N', 'eeg_cat_A', 'eeg_cat_E', 'eeg_cat_N', 'psuedo_outcome']
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
# traindata = pd.concat([Glasgowdata,Chinadata,perthdata,Malaysiadata])
cross_valid = False
use_resample = False
use_smote = True
plot_map = True
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
X_all, y_all = traindata.iloc[:,2:-1],traindata.iloc[:,-1]
# print(traindata.columns)
if cross_valid:
    X_test, y_test = X_all, y_all
    if use_smote:
        X_train_smote, y_train_smote = sm.fit_resample(X_all, y_all)
    else:
        X_train_smote, y_train_smote = X_all, y_all
else:
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all)
    if use_smote:
        X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)
    else:
        X_train_smote, y_train_smote = X_train, y_train
X_external, y_external = externaldata.iloc[:,2:-1],externaldata.iloc[:,-1]
input_dim = X_all.shape[1]

train_dataset = TensorDataset(torch.Tensor(X_train_smote.to_numpy().astype(float)),
                              torch.Tensor(y_train_smote.values))  # 相当于zip函数
from torch.utils.data.sampler import WeightedRandomSampler

weights = [2 if label == 1 else 1 for data, label in train_dataset]
sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
if use_resample:
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              sampler=sampler
                                              # shuffle=True
                                              , num_workers=10)
    if not cross_valid:
        train_val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                  shuffle=True
                                                  , num_workers=10)
elif use_smote:
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              # sampler=sampler
                                              shuffle=True
                                              , num_workers=10)
    if not cross_valid:
        train_val_dataset = TensorDataset(torch.Tensor(X_train.to_numpy().astype(float)),
                                      torch.Tensor(y_train.values))  # 相当于zip函数
        train_val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=batch_size,
                                                  # sampler=sampler
                                                  shuffle=True
                                                  , num_workers=10)
else:
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              # sampler=sampler
                                              shuffle=True
                                              , num_workers=10)
    if not cross_valid:
        train_val_loader = trainloader
test_dataset = TensorDataset(torch.Tensor(X_test.to_numpy().astype(float))
                               ,torch.Tensor(y_test.values))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

malaysia_dataset = TensorDataset(torch.Tensor(X_Malaysia.to_numpy())
                               ,torch.Tensor(y_Malaysia.values))
malaysia_loader = torch.utils.data.DataLoader(malaysia_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

chongqing_dataset = TensorDataset(torch.Tensor(X_chongqing.to_numpy())
                               ,torch.Tensor(y_chongqing.values))
chongqing_loader = torch.utils.data.DataLoader(chongqing_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

china_dataset = TensorDataset(torch.Tensor(X_China.to_numpy())
                               ,torch.Tensor(y_China.values))
china_loader = torch.utils.data.DataLoader(china_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

perth_dataset = TensorDataset(torch.Tensor(X_perth.to_numpy())
                               ,torch.Tensor(y_perth.values))
perth_loader = torch.utils.data.DataLoader(perth_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

external_dataset = TensorDataset(torch.Tensor(X_external.to_numpy())
                               ,torch.Tensor(y_external.values))
external_loader = torch.utils.data.DataLoader(external_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

print("len of train dataset:",len(train_dataset))
print("len of test dataset:",len(test_dataset))
print("len of malaysia dataset:",len(malaysia_dataset))
print("len of perth dataset:",len(perth_dataset))
print("len of chongqing dataset:",len(chongqing_dataset))
print("len of china dataset:",len(china_dataset))
print("len of external dataset:",len(external_dataset))
trg_label = 1 # Classification of psuedo_outcome and DRE
## Create a best_model variable for loading
# best_model = MLP(input_dim = src_var).cuda()
# ### Training begins here
num_steps = 5
model = Transformer(input_dim = input_dim).cuda()

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)  # this code is very important! It initialises the parameters with a
# for m in model.modules():
#     if isinstance(m, nn.Conv1d):
#         nn.init.kaiming_normal_(m.weight)
#     elif isinstance(m, nn.BatchNorm1d):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.Linear):
#         nn.init.constant_(m.bias, 0)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2,weight_decay=0.05)

scheduler = StepLR(optimizer, step_size=num_steps, gamma=0.65)
test_loss_log, train_loss_log, train_acc_log, test_acc_log = [], [], [], []

best_test_weighted_acc = 0
best_test_AUC = 0
# best_external_Weighted_Acc = 0
ctr = 0
epochs = 4
if use_resample or use_smote:
    weight_CE = torch.FloatTensor([1,1]).cuda()
else:
    weight_CE = torch.FloatTensor([1, 2]).cuda()
loss_fn = torch.nn.CrossEntropyLoss(weight=weight_CE)
for epoch in range(epochs):
    model.train()  ## Model is in train mode (look at pytorch library to see meaning)
    total_loss = 0
    all_preds = []
    all_labels = []
    for i, (batch, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        src = batch.type(torch.float32)  # Turn into a batch
        preds, nl_map_0 = model(src.cuda())
        preds_after = preds[:].squeeze()
        # preds_soft = F.softmax(preds,dim=-1)
        # preds_after = preds_soft[:,1:].squeeze()
        # ## Aggregate losses and their gradients based off label (for class imbalance)
        # if (trg_label == 1):
        #     # loss = lossfuns(preds, results[:])
        #     w = [0.7, 1.0]  # 标签0和标签1的权重
        #     weight = torch.zeros(labels.shape).cuda()  # 权重矩阵
        #     for i in range(labels.shape[0]):
        #         weight[i] = w[int(labels[i])]
        #     # print(preds_after,labels)
        # loss = loss_fn(preds, labels.type(torch.LongTensor).cuda())
        loss = F.binary_cross_entropy(preds_after, labels.cuda())
        all_preds.append(preds_after)
        all_labels.append(labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    ## Rest should be self explanatory
    real_results = torch.cat(all_labels, dim=0).flatten().type(torch.float32).cpu().numpy()  ## Flat tensor
    pred_results = torch.cat(all_preds, dim=0).flatten().type(torch.float32).cpu().detach().numpy()
    train_AUC_ROC = roc_auc_score(real_results, pred_results)
    pred_res = pred_results>=0.5

    TP = sum((pred_res == 1) & (real_results == 1))
    FN = sum((pred_res == 0) & (real_results == 1))
    TN = sum((pred_res == 0) & (real_results == 0))
    FP = sum((pred_res == 1) & (real_results == 0))
    train_sens = TP / (TP + FN)
    train_spec = TN / (TN + FP)
    train_accuracy_calculated = (TN + TP) / (TN + TP + FP + FN)
    train_weighted_acc = (TN / (FP + TN)) * 0.5 + (TP / (TP + FN)) * 0.5
    train_loss = total_loss / len(trainloader)
    if epoch % 5 == 0:
        print("epoch {:3d}".format(epoch),"\ttrain_loss: {0:.2f}".format(train_loss),
              "\ttrain_Sens: {0:.2f}".format(train_sens), "\ttrain_Spec: {0:.2f}".format(train_spec),
              "\ttrain_Acc: {0:.2f}".format(train_accuracy_calculated),
              '\ttrain_Weighted Acc: {0:.2f}'.format(train_weighted_acc),
              "\ttrain_AUC: {0:.2f}".format(train_AUC_ROC))
    if (use_weighted_acc):
        test_sens, test_spec, test_acc,test_weighted_acc,test_AUC_ROC,test_loss,_, _ = test_model_thres(model, test_loader,
                                                                                  trg_label=trg_label)

    if (test_AUC_ROC > best_test_AUC):
        save_nl_map(test_loader, model,use_age_category)
        best_test_AUC = test_AUC_ROC
        best_model = copy.deepcopy(model)
        # if (perth_val):
        print("====================================")
        print("in best model, validating on testing subset")
        # test_sens, test_spec, test_acc,test_weighted_acc,test_AUC_ROC,test_loss \
        #     = test_model_thres(best_model, test_loader, trg_label=trg_label)
        print("epoch {:3d}".format(epoch),
              "\ttest_loss: {0:.2f}".format(test_loss),
              "\ttest_Sens: {0:.2f}".format(test_sens),
              "\ttest_Spec: {0:.2f}".format(test_spec),
              "\ttest_Acc: {0:.2f}".format(test_acc),
              '\ttest_Weighted Acc: {0:.2f}'.format(test_weighted_acc),
              "\ttest_AUC: {0:.2f}".format(test_AUC_ROC))
# # # # Thresholds are sensitive with respect to the proportioned learning rates between classes (norm_lr). This may have been adjusted since the last model training and may give different threshold values to what has been documented.

if cross_valid:

    print("Best model on testing dataset:")
    train_weight_arr = []
    train_thres_arr = []
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        test_sens, test_spec, test_accuracy_calculated,test_weighted_acc,test_AUC_ROC,test_loss,real_results, pred_results\
            = test_model_thres(best_model, test_loader, thres_val=i,
                                                       trg_label=trg_label)
        train_weight_arr.append(test_weighted_acc)
        train_thres_arr.append(i)
        print("Threshold: {0:.2f}".format(i),
              "\ttest_Sens: {0:.2f}".format(test_sens),
              "\ttest_Spec: {0:.2f}".format(test_spec),
              "\ttest_Acc: {0:.2f}".format(test_accuracy_calculated),
              '\ttest_Weighted Acc: {0:.2f}'.format(test_weighted_acc),
              "\ttest_AUC: {0:.2f}".format(test_AUC_ROC))

    from sklearn.metrics import roc_curve
    if plot_map:
        plt.cla()

        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        straight = [0, 1]
        ax.text(-0.17, 1, "A", fontsize=25, weight='bold')
        # ax.text(1.13, 1, "B", fontsize=25, weight='bold')
        # ax.text(2.48, 1, "C", fontsize=25, weight='bold')
        AUC_ROC = roc_auc_score(real_results, pred_results)
        fpr, tpr, _ = roc_curve(real_results, pred_results)
        # auc = metrics.roc_auc_score(real_res, pred_res)
        plt.plot(fpr, tpr, label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC, linewidth=4)
        # plt.plot([0, 0.225], [0.82, 0.82], label = "Threshold = 0.55",linestyle = ':', color = 'g', linewidth = 4, alpha = 0.7)
        # plt.plot([0.225, 0.225], [0, 0.82], linestyle = ':', color = 'g',linewidth = 4, alpha = 0.7)
        plt.plot(straight, straight, label="Random Guess", linestyle=':', linewidth=4)
        # plt.legend(loc = 'none')
        # ax.get_legend().remove()
        l = ax.get_xlabel()
        ax.set_xlabel(l, fontsize=24)
        l = ax.get_ylabel()
        ax.set_ylabel(l, fontsize=24)
        plt.legend(loc='lower center', markerscale=5, numpoints=3)
        plt.setp(ax.get_legend().get_texts(), fontsize='20')  # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='20')  # for legend title

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title("AUC="+str(auc))
        # plt.legend(, loc=4)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        # plt.show()
        plt.grid()
        plt.savefig("nl_map_vis/train_auc_weight_curve.pdf", dpi=300, bbox_inches='tight')


        plt.cla()
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        fpr, tpr, _ = roc_curve(real_results, pred_results)
        # auc = metrics.roc_auc_score(real_res, pred_res)
        plt.plot(train_thres_arr[:-2], train_weight_arr[:-2], label='Transformer Model', linewidth=4)
        max_weight = max(train_weight_arr)
        max_thres = train_thres_arr[train_weight_arr.index(max_weight)]
        plt.plot([0.26, max_thres], [max_weight, max_weight + 0.002], label="Threshold = %0.2f" % max_thres, linestyle=':',
                 color='g', linewidth=4, alpha=0.7)
        plt.plot([max_thres, max_thres], [0.41, max_weight], linestyle=':', color='g', linewidth=4, alpha=0.7)
        ax.text(0.18, 0.74, "B", fontsize=25, weight='bold')
        # plt.legend(loc = 'none')
        # ax.get_legend().remove()
        l = ax.get_xlabel()
        ax.set_xlabel(l, fontsize=24)
        l = ax.get_ylabel()
        ax.set_ylabel(l, fontsize=24)
        plt.legend(loc='lower center', markerscale=5, numpoints=3)
        plt.setp(ax.get_legend().get_texts(), fontsize='20')  # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='20')  # for legend title

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title("AUC="+str(auc))
        # plt.legend(, loc=4)
        plt.xlabel("Threshold values")
        plt.ylabel("Weighted accuracy")
        # plt.show()

        plt.grid()
        ax.set_ylim([0.5, 0.75])
        ax.set_xlim([0.25, 0.75])
        plt.savefig("nl_map_vis/train_weight_acc.pdf", dpi=300, bbox_inches='tight')

    print("Best model on perth dataset:")
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        perth_sens, perth_spec, perth_accuracy_calculated,perth_weighted_acc,perth_AUC_ROC,perth_loss,real_results, pred_results\
            = test_model_thres(best_model, perth_loader, thres_val=i,
                                                       trg_label=trg_label)
        print("Threshold: {0:.2f}".format(i),
              "\tperth_Sens: {0:.2f}".format(perth_sens),
              "\tperth_Spec: {0:.2f}".format(perth_spec),
              "\tperth_Acc: {0:.2f}".format(perth_accuracy_calculated),
              '\tperth_Weighted Acc: {0:.2f}'.format(perth_weighted_acc),
              "\tperth_AUC: {0:.2f}".format(perth_AUC_ROC))
    if plot_map:
        plt.cla()
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        straight = [0, 1]
        ax.text(-0.17, 1, "E", fontsize=20, weight='bold')
        # ax.text(1.13, 1, "B", fontsize=20, weight='bold')
        AUC_ROC = roc_auc_score(real_results, pred_results)
        fpr, tpr, _ = roc_curve(real_results, pred_results)
        # auc = metrics.roc_auc_score(real_res, pred_res)
        plt.plot(fpr, tpr, label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC, linewidth=4)
        plt.plot(straight, straight, label="Random Guess", linestyle=':', linewidth=4)
        # plt.legend(loc = 'none')
        # ax.get_legend().remove()
        l = ax.get_xlabel()
        ax.set_xlabel(l, fontsize=24)
        l = ax.get_ylabel()
        ax.set_ylabel(l, fontsize=24)
        plt.legend(loc='lower center', markerscale=5, numpoints=3)
        plt.setp(ax.get_legend().get_texts(), fontsize='20')  # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='20')  # for legend title

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title("AUC="+str(auc))
        # plt.legend(, loc=4)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        # plt.show()
        plt.grid()
        plt.savefig("nl_map_vis/perth_auc_weight_curve.pdf", dpi=300, bbox_inches='tight')

    print("Best model on malaysia dataset:")
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        malaysia_sens, malaysia_spec, malaysia_accuracy_calculated,malaysia_weighted_acc,malaysia_AUC_ROC,malaysia_loss,real_results, pred_results\
            = test_model_thres(best_model, malaysia_loader, thres_val=i,
                                                       trg_label=trg_label)
        print("Threshold: {0:.2f}".format(i),
              "\tmalaysia_Sens: {0:.2f}".format(malaysia_sens),
              "\tmalaysia_Spec: {0:.2f}".format(malaysia_spec),
              "\tmalaysia_Acc: {0:.2f}".format(malaysia_accuracy_calculated),
              '\tmalaysia_Weighted Acc: {0:.2f}'.format(malaysia_weighted_acc),
              "\tmalaysia_AUC: {0:.2f}".format(malaysia_AUC_ROC))
    if plot_map:
        plt.cla()
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        straight = [0, 1]
        ax.text(-0.17, 1, "C", fontsize=20, weight='bold')
        # ax.text(1.13, 1, "B", fontsize=20, weight='bold')
        AUC_ROC = roc_auc_score(real_results, pred_results)
        fpr, tpr, _ = roc_curve(real_results, pred_results)
        # auc = metrics.roc_auc_score(real_res, pred_res)
        plt.plot(fpr, tpr, label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC, linewidth=4)
        plt.plot(straight, straight, label="Random Guess", linestyle=':', linewidth=4)
        l = ax.get_xlabel()
        ax.set_xlabel(l, fontsize=24)
        l = ax.get_ylabel()
        ax.set_ylabel(l, fontsize=24)
        plt.legend(loc='lower center', markerscale=5, numpoints=3)
        plt.setp(ax.get_legend().get_texts(), fontsize='20')  # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='20')  # for legend title

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title("AUC="+str(auc))
        # plt.legend(, loc=4)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        # plt.show()
        plt.grid()
        plt.savefig("nl_map_vis/malaysia_auc_weight_curve.pdf", dpi=300, bbox_inches='tight')

    print("Best model on chongqing dataset:")
    chongqing_weight_arr = []
    chongqing_thres_arr = []
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        chongqing_sens, chongqing_spec, chongqing_accuracy_calculated,chongqing_weighted_acc,\
        chongqing_AUC_ROC,chongqing_loss,real_results, pred_results\
            = test_model_thres(best_model, chongqing_loader, thres_val=i,
                                                       trg_label=trg_label)
        chongqing_weight_arr.append(chongqing_weighted_acc)
        chongqing_thres_arr.append(i)
        print("Threshold: {0:.2f}".format(i),
              "\tchongqing_Sens: {0:.2f}".format(chongqing_sens),
              "\tchongqing_Spec: {0:.2f}".format(chongqing_spec),
              "\tchongqing_Acc: {0:.2f}".format(chongqing_accuracy_calculated),
              '\tchongqing_Weighted Acc: {0:.2f}'.format(chongqing_weighted_acc),
              "\tchongqing_AUC: {0:.2f}".format(chongqing_AUC_ROC))
    if plot_map:
        plt.cla()
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        straight = [0, 1]
        ax.text(-0.17, 1, "D", fontsize=20, weight='bold')
        # ax.text(1.13, 1, "B", fontsize=20, weight='bold')
        AUC_ROC = roc_auc_score(real_results, pred_results)
        fpr, tpr, _ = roc_curve(real_results, pred_results)
        # auc = metrics.roc_auc_score(real_res, pred_res)
        plt.plot(fpr, tpr, label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC, linewidth=4)
        plt.plot(straight, straight, label="Random Guess", linestyle=':', linewidth=4)
        l = ax.get_xlabel()
        ax.set_xlabel(l, fontsize=24)
        l = ax.get_ylabel()
        ax.set_ylabel(l, fontsize=24)
        plt.legend(loc='lower center', markerscale=5, numpoints=3)
        plt.setp(ax.get_legend().get_texts(), fontsize='20')  # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='20')  # for legend title

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title("AUC="+str(auc))
        # plt.legend(, loc=4)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        # plt.show()
        plt.grid()
        plt.savefig("nl_map_vis/chongqing_auc_weight_curve.pdf", dpi=300, bbox_inches='tight')

    print("Best model on GUANGZHOU dataset:")
    guangzhou_weight_arr = []
    guangzhou_thres_arr = []
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        guangzhou_sens, guangzhou_spec, guangzhou_accuracy_calculated,\
        guangzhou_weighted_acc,guangzhou_AUC_ROC,guangzhou_loss,real_results, pred_results\
            = test_model_thres(best_model, china_loader, thres_val=i,
                                                       trg_label=trg_label)
        guangzhou_weight_arr.append(guangzhou_weighted_acc)
        guangzhou_thres_arr.append(i)
        print("Threshold: {0:.2f}".format(i),
              "\tguangzhou_Sens: {0:.2f}".format(guangzhou_sens),
              "\tguangzhou_Spec: {0:.2f}".format(guangzhou_spec),
              "\tguangzhou_Acc: {0:.2f}".format(guangzhou_accuracy_calculated),
              '\tguangzhou_Weighted Acc: {0:.2f}'.format(guangzhou_weighted_acc),
              "\tguangzhou_AUC: {0:.2f}".format(guangzhou_AUC_ROC))
    if plot_map:
        plt.cla()
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        straight = [0, 1]
        ax.text(-0.17, 1, "F", fontsize=20, weight='bold')
        # ax.text(1.13, 1, "B", fontsize=20, weight='bold')
        AUC_ROC = roc_auc_score(real_results, pred_results)
        fpr, tpr, _ = roc_curve(real_results, pred_results)
        # auc = metrics.roc_auc_score(real_res, pred_res)
        plt.plot(fpr, tpr, label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC, linewidth=4)
        plt.plot(straight, straight, label="Random Guess", linestyle=':', linewidth=4)
        # plt.legend(loc = 'none')
        # ax.get_legend().remove()
        l = ax.get_xlabel()
        ax.set_xlabel(l, fontsize=24)
        l = ax.get_ylabel()
        ax.set_ylabel(l, fontsize=24)
        plt.legend(loc='lower center', markerscale=5, numpoints=3)
        plt.setp(ax.get_legend().get_texts(), fontsize='20')  # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='20')  # for legend title

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title("AUC="+str(auc))
        # plt.legend(, loc=4)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        # plt.show()
        plt.grid()
        plt.savefig("nl_map_vis/guangzhou_auc_weight_curve.pdf", dpi=300, bbox_inches='tight')
else:
    print("Best model on training dataset:")
    train_weight_arr = []
    train_thres_arr = []
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        train_sens, train_spec, train_accuracy_calculated,train_weighted_acc,train_AUC_ROC,train_loss,real_results, pred_results\
            = test_model_thres(model, trainloader, thres_val=i,
                                                       trg_label=trg_label)
        train_weight_arr.append(train_weighted_acc)
        train_thres_arr.append(i)
        print("Threshold: {0:.2f}".format(i),
              "\ttrain_Sens: {0:.2f}".format(train_sens),
              "\ttrain_Spec: {0:.2f}".format(train_spec),
              "\ttrain_Acc: {0:.2f}".format(train_accuracy_calculated),
              '\ttrain_Weighted Acc: {0:.2f}'.format(train_weighted_acc),
              "\ttrain_AUC: {0:.2f}".format(train_AUC_ROC))
    from sklearn.metrics import roc_curve
    if plot_map:
        plt.cla()

        plt.figure(figsize=(36, 8))
        ax = plt.subplot(131)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        straight = [0, 1]
        ax.text(-0.17, 1, "A", fontsize=25, weight='bold')
        ax.text(1.13, 1, "B", fontsize=25, weight='bold')
        ax.text(2.48, 1, "C", fontsize=25, weight='bold')
        AUC_ROC = roc_auc_score(real_results, pred_results)
        fpr, tpr, _ = roc_curve(real_results, pred_results)
        # auc = metrics.roc_auc_score(real_res, pred_res)
        plt.plot(fpr, tpr, label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC, linewidth=4)
        # plt.plot([0, 0.225], [0.82, 0.82], label = "Threshold = 0.55",linestyle = ':', color = 'g', linewidth = 4, alpha = 0.7)
        # plt.plot([0.225, 0.225], [0, 0.82], linestyle = ':', color = 'g',linewidth = 4, alpha = 0.7)
        plt.plot(straight, straight, label="Random Guess", linestyle=':', linewidth=4)
        # plt.legend(loc = 'none')
        # ax.get_legend().remove()
        l = ax.get_xlabel()
        ax.set_xlabel(l, fontsize=24)
        l = ax.get_ylabel()
        ax.set_ylabel(l, fontsize=24)
        plt.legend(loc='lower center', markerscale=5, numpoints=3)
        plt.setp(ax.get_legend().get_texts(), fontsize='20')  # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='20')  # for legend title

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title("AUC="+str(auc))
        # plt.legend(, loc=4)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        # plt.show()
        plt.grid()
        # plt.savefig("nl_map_vis/pool_train_auc_weight_curve.pdf", dpi=300, bbox_inches='tight')


        # plt.cla()
        # plt.figure(figsize=(10, 8))
        ax = plt.subplot(132)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        fpr, tpr, _ = roc_curve(real_results, pred_results)
        # auc = metrics.roc_auc_score(real_res, pred_res)
        plt.plot(train_thres_arr[:-2], train_weight_arr[:-2], label='Transformer Model', linewidth=4)
        max_weight = max(train_weight_arr)
        max_thres = train_thres_arr[train_weight_arr.index(max_weight)]
        plt.plot([0.26, max_thres], [max_weight, max_weight + 0.002], label="Threshold = %0.2f" % max_thres, linestyle=':',
                 color='g', linewidth=4, alpha=0.7)
        plt.plot([max_thres, max_thres], [0.41, max_weight], linestyle=':', color='g', linewidth=4, alpha=0.7)
        # ax.text(0.18, 0.74, "B", fontsize=25, weight='bold')
        # plt.legend(loc = 'none')
        # ax.get_legend().remove()
        l = ax.get_xlabel()
        ax.set_xlabel(l, fontsize=24)
        l = ax.get_ylabel()
        ax.set_ylabel(l, fontsize=24)
        plt.legend(loc='lower center', markerscale=5, numpoints=3)
        plt.setp(ax.get_legend().get_texts(), fontsize='20')  # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='20')  # for legend title

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title("AUC="+str(auc))
        # plt.legend(, loc=4)
        plt.xlabel("Threshold values")
        plt.ylabel("Weighted accuracy")
        # plt.show()

        plt.grid()
        ax.set_ylim([0.5, 0.75])
        ax.set_xlim([0.25, 0.75])
        # plt.savefig("nl_map_vis/pool_train_weight_acc.pdf", dpi=300, bbox_inches='tight')

    print("Best model on testing dataset:")
    for i in np.linspace(0, 1, 21):
        # for i in np.linspace(0.4,0.8,9): # Faster for 0.05 intervals, originally 0.01

        test_sens, test_spec, test_accuracy_calculated, test_weighted_acc, test_AUC_ROC, test_loss,real_results, pred_results \
            = test_model_thres(model, test_loader, thres_val=i,
                               trg_label=trg_label)
        print("Threshold: {0:.2f}".format(i),
              "\ttest_Sens: {0:.2f}".format(test_sens),
              "\ttest_Spec: {0:.2f}".format(test_spec),
              "\ttest_Acc: {0:.2f}".format(test_accuracy_calculated),
              '\ttest_Weighted Acc: {0:.2f}'.format(test_weighted_acc),
              "\ttest_AUC: {0:.2f}".format(test_AUC_ROC))
    if plot_map:
        # plt.cla()
        #
        # plt.figure(figsize=(10, 8))
        ax = plt.subplot(133)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        straight = [0, 1]
        # ax.text(-0.17, 1, "C", fontsize=25, weight='bold')
        # ax.text(1.13, 1, "B", fontsize=25, weight='bold')
        # ax.text(2.48, 1, "C", fontsize=25, weight='bold')
        AUC_ROC = roc_auc_score(real_results, pred_results)
        fpr, tpr, _ = roc_curve(real_results, pred_results)
        # auc = metrics.roc_auc_score(real_res, pred_res)
        plt.plot(fpr, tpr, label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC, linewidth=4)
        # plt.plot([0, 0.225], [0.82, 0.82], label = "Threshold = 0.55",linestyle = ':', color = 'g', linewidth = 4, alpha = 0.7)
        # plt.plot([0.225, 0.225], [0, 0.82], linestyle = ':', color = 'g',linewidth = 4, alpha = 0.7)
        plt.plot(straight, straight, label="Random Guess", linestyle=':', linewidth=4)
        # plt.legend(loc = 'none')
        # ax.get_legend().remove()
        l = ax.get_xlabel()
        ax.set_xlabel(l, fontsize=24)
        l = ax.get_ylabel()
        ax.set_ylabel(l, fontsize=24)
        plt.legend(loc='lower center', markerscale=5, numpoints=3)
        plt.setp(ax.get_legend().get_texts(), fontsize='20')  # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='20')  # for legend title

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title("AUC="+str(auc))
        # plt.legend(, loc=4)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        # plt.show()
        plt.grid()
        plt.savefig("nl_map_vis/pool_test_auc_weight_curve.pdf", dpi=300, bbox_inches='tight')
