#!/usr/bin/env python
# coding: utf-8

import datetime as dt
import pandas as pd
import copy
import math

pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)  # Neater

## Useful parameters
debug = True
classifier_data = False
drop_eeg = False
age_category = True  # Trichotomises the age

adults_only = True  # >=18 age

first_regimen_only = True  ## Taking first regimen only

drop_some_regimen = True  ## "ACE" and others

single_regimen_only = False  ## "In terms of only having one regimen in each trial"

april_15 = True  # Big edit to the data on April 15 - hence use this for the latest version

## Outdated parameters

haris_data = False  # Haris edited data before - not relevant anymore since 2021 April

test_no_extra_vars = False  # There were extra variables with Haris' data - not relevant anymore

all_file_name ='/mnt/workdir/fengwei/transformer_master-master/data/validation/dec1/CQ_12 Dec 2021.xls'
               # 'CQ_17 Jan 2022.xls'

all_df = pd.DataFrame(pd.read_excel(all_file_name))
all_df = all_df.reset_index(drop=True)
df = all_df
# if (debug):
#     print(df.dtypes)
    # df.describe() ## Summary of the data
    # print(df)
## Classifier data originally for DRE prediction - not used anymore (project got taken by someone else)

if (debug):
    for i in df.columns:
        print(i)

# ### Some data cleaning utilities
## Self explanatory utilities
def date_to_epoch(x):
    x = pd.to_datetime(x)
    return (x - dt.datetime(1970, 1, 1)).dt.total_seconds()


def normalize(x):
    #     return (x - min(x))/max(x)
    return x


def to_binary(x, dictionary):
    return x.map(dictionary).astype(int)


def to_one_hot(df, x, drop_first=False):
    temp = pd.get_dummies(df[x], prefix=x, drop_first=drop_first)
    cntr = 0

    ## Appending columns
    for i in temp.columns:
        #         print(df)
        df[i] = pd.Series(temp[i], index=df.index)
    return df.drop(columns=[x])


def filter_for(df, col, x):
    return df[df[col] == x]


df_trans = copy.copy(df)  ## Make a copy of the original data set and transform it

## Binarise or categorise most of them
## Some columns are dropped because they have too many NAs, are repeated variables, or are not relevant
# print(df_trans['pid'])

# df_trans['pid'] = df['pid'].str[3:].astype(int)
df_trans['sex'] = to_binary(df['sex'],dict(F=1, M=0))
print("intial num:",len(df_trans))
# df_trans = df_trans.dropna(axis=0,how='any')
print("dropna num:",len(df_trans))
df_trans = df_trans.drop(df_trans[(df_trans['lesion'] == 'NO RECORD') | (df_trans['lesion'].isnull()) |
                                  (df_trans['lesion'] == 'no record')].index)  # 删除的行(2160->2047)

print("drop lesion num:",len(df_trans))
df_trans = df_trans.drop(df_trans[(df_trans['eeg_cat'] == 'no record') | (df_trans['eeg_cat'] == '无记录') |
                                  (df_trans['eeg_cat'] == '需查下记录，2018年有入院检查') | (
                                              df_trans['eeg_cat'] == '治疗10个月后脑电图正常') |
                                  (df_trans['eeg_cat'] == '2017年发病，2021年脑电痫样放电') | (df_trans['eeg_cat'] == '缺') | (
                                      df_trans['eeg_cat'].isnull())].index) ###(2047-->1722)
print("drop eeg cat num:",len(df_trans))
df_trans = to_one_hot(df_trans, 'lesion')
df_trans = df_trans.drop(df_trans[(df_trans['pretrt_sz_5'] == 9)].index)  # 删除的行
df_trans['pretrt_sz_5'] = to_binary(df_trans['pretrt_sz_5'],dict(Y=1, N=0))
df_trans['fam_hx'] = to_binary(df_trans['fam_hx'],dict(Y=1, N=0))
df_trans['febrile'] = to_binary(df_trans['febrile'],dict(Y=1, N=0))
df_trans = df_trans.drop(df_trans[(df_trans['ci'].isnull())].index)  # 删除的行
df_trans['ci'] = to_binary(df_trans['ci'],dict(Y=1, N=0))

df_trans['birth_t'] = to_binary(df_trans['birth_t'],dict(Y=1, N=0))

df_trans = df_trans.drop(df_trans[(df_trans['head'] == 'Unconscious after falling at the age of 4')].index)  # 删除的行
df_trans['head'] = to_binary(df_trans['head'],dict(Y=1, N=0))
df_trans['drug'] = to_binary(df_trans['drug'],dict(Y=1, N=0))
df_trans['alcohol'] = to_binary(df_trans['alcohol'],dict(Y=1, N=0))
df_trans['cvd'] = to_binary(df_trans['cvd'],dict(Y=1, N=0))
df_trans['psy'] = to_binary(df_trans['psy'],dict(Y=1, N=0))
df_trans['ld'] = to_binary(df_trans['ld'],dict(Y=1, N=0))
df_trans['focal'] = to_binary(df_trans['focal'],dict(Y=1, N=0))
df_trans = df_trans.drop(df_trans[(df_trans['focal'].isnull())].index)  # 删除的行
print("drop focal num:",len(df_trans))

if (adults_only):
    if (debug):
        print("Number of patients that are under 18: ", len(df_trans[df_trans['age_init'] < 18]['pid'].unique()))
    print("before remove child:", len(df_trans))
    df_trans = df_trans[df_trans['age_init'] >= 18] ###(2489, 21)-->(2113, 21)
    print("after remove child:",len(df_trans))
# ## OUTDATED
if (drop_eeg):
    df_trans = df_trans.drop(columns=["eeg_cat"])
else:
    df_trans = to_one_hot(df_trans, 'eeg_cat')

## We found a improvement in the performance of the models instead of using continuous variables
## We divided it into 3 categories instead
## Less of the latent space of the model to explore
if (age_category):
    # Setup age values
    age_values = [29, 46]

    age_init_idx = df_trans.columns.get_loc("age_init")  # Find index
    ctr = 1
    for age in age_values:
        df_trans.insert(loc=age_init_idx + ctr, column="age_init_" + str(age), value=[0 for i in range(len(df_trans))])
        ctr += 1
    df_trans.insert(loc=age_init_idx + ctr, column="age_init_>" + str(age_values[-1]),
                    value=[0 for i in range(len(df_trans))])
    # Assign one hot encoding
    for row in df_trans.index:
        if (df_trans["age_init"][row] <= age_values[0]):
            df_trans["age_init_" + str(age_values[0])][row] = 1
        elif (df_trans["age_init"][row] <= age_values[1]):
            df_trans["age_init_" + str(age_values[1])][row] = 1
        else:
            df_trans["age_init_>" + str(age_values[-1])][row] = 1
    # Remove and permute columns
    df_trans = df_trans.drop(columns=["age_init"])
# if (first_regimen_only):
#     df_trans = df_trans.drop_duplicates(subset=["pid"])

unique_drugs = df_trans['asm'].unique()
select_drugs = ['CBZ', 'LEV','LTG', 'OXC', 'PHT', 'TPM', 'VPA']
# print(unique_drugs)
## Drop some regimens
drop_drugs = list(set(unique_drugs)-set(select_drugs))
# print(drop_drugs)
# drop_drugs = [ 'CLB', 'ESL', 'FBM','LCM','PER', 'PGB', 'REM', 'TGB', 'VGB', 'ZNS','RTG','LEVTPM','ACE','RFN']
for i in range(len(drop_drugs)):
    df_trans = df_trans.drop(df_trans[(df_trans['asm'] == drop_drugs[i])].index) ###(2113-->2050)
## Add a column for each drug

## Add a column for each drug
for i in range(len(select_drugs)):
    df_trans[select_drugs[i]] = pd.Series(0,index=df.index).astype(float)
for row in df_trans.index:
    drug_name = df['asm'][row]
    ## If not NAN, allocate
    if (not pd.isna(df['asm'][row])):
        drug_val = 1  # df[ddd_dict[col]][row]
        df_trans[drug_name][row] = drug_val
    ## Else break to next row
    else:
        if debug:
            print('Breaking from for loop')
        break
print(df_trans['outcome_12m'].value_counts())
df_trans['outcome_12m'] = df_trans['outcome_12m'].replace({'Seizure-free':1, 'Not seizure-free':0})
df_trans.rename(columns={'outcome_12m': 'psuedo_outcome'}, inplace=True)
df_trans['psuedo_outcome'] = df_trans['psuedo_outcome'].replace({2: 0})
# ## Add a column for each drug
drugs_to_keep = ['pid', 'cohort','sex', 'age_init_29', 'age_init_46', 'age_init_>46',
                 'pretrt_sz_5', 'focal', 'fam_hx', 'febrile', 'ci', 'birth_t', 'head',
                 'drug', 'alcohol', 'cvd', 'psy', 'ld', 'lesion_A', 'lesion_E',
                 'lesion_N', 'eeg_cat_A', 'eeg_cat_E', 'eeg_cat_N', 'CBZ', 'LEV',
                 'LTG', 'OXC', 'PHT', 'TPM', 'VPA', 'psuedo_outcome']
df_trans = df_trans[[c for c in drugs_to_keep if c in df_trans]]
# df_trans = df_trans.dropna(axis = 0, how='any')

## Add in any missing drugs:
## This is because Perth data set might not have any existing drugs, so need to add it back in to match Glasgow
## perth所包含的药品数量没有glasow多
# df_trans = df_trans.drop_duplicates(subset=["pid"])
print(sum(df_trans['eeg_cat_A']) + sum(df_trans['eeg_cat_E']) + sum(df_trans['eeg_cat_N']))

print(len(df_trans.groupby('age_init_' + str(age_values[0]))['pid'].unique()[1]))
print(len(df_trans.groupby('age_init_' + str(age_values[1]))['pid'].unique()[1]))
print(len(df_trans.groupby('age_init_>' + str(age_values[1]))['pid'].unique()[1]))
print(len(df_trans))
csv_name = '../data/validation/chongqing_single_all_cols_v8_2.csv'
df_trans.to_csv(csv_name, encoding='utf-8', index=False)  # For all the labels


