# -*- coding: utf-8 -*-
"""
Brain-age model is trained on discovery (UCLA CNP + ABIDE I/II + SRPBS
Japan, age >= 18, no BMI available), with feature selection (Pearson
p <= 0.01 on FC edges, computed within each training fold) and a
BayesianRidge regressor, evaluated via 10-fold CV. The fitted fold models
are then applied to an independent held-out set and to the Church cohort.
"""

import pandas as pd
import numpy as np
import scipy.io as sio
from pathlib import Path
from bids.layout import parse_file_entities
import sys
sys.path.append(r'H:\utils_for_all')
from common_utils import read_singal_fmri_ts, get_fc, keep_triangle_half, setup_seed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.linear_model import BayesianRidge, LinearRegression
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from collections import namedtuple
import os

font2 = {'family': 'Tahoma', 'weight': 'bold', 'size': 30}
matplotlib.rc('font', **font2)
setup_seed(6)

Color = namedtuple('RGB', 'red, green, blue')
color1 = Color(0.52, 0.52, 1)
color2 = Color(0., 0.28, 0.87)
color3 = Color(0.4, 0.4, 0.67)

###########################################################
########################################### UCLA CNP
###########################################################
ts_path = Path(r'I:\data\UCLA\ROISignals')
ts_files = ts_path.glob("*\\*_Schaefer100.csv")
ts_files = sorted([str(files) for files in ts_files])
UCLA_CNP_fmri_sub = []
UCLA_CNP_fc = []

for i, file in enumerate(ts_files):
    print(i / len(ts_files))
    file_entities = parse_file_entities(file)
    sub = file_entities['subject']
    UCLA_CNP_fmri_sub.append(sub)

    ts = read_singal_fmri_ts(file)
    fc = get_fc(ts)
    UCLA_CNP_fc.append(fc)
UCLA_CNP_fmri_sub = np.array(UCLA_CNP_fmri_sub)
UCLA_CNP_fc = np.array(UCLA_CNP_fc)
half_fc_UCLA_CNP = keep_triangle_half(UCLA_CNP_fc.shape[1] * (UCLA_CNP_fc.shape[1] - 1) // 2, UCLA_CNP_fc.shape[0], UCLA_CNP_fc)

UCLA_CNP_demo = pd.read_csv(r'I:\data\UCLA\participants.tsv', sep='\t', on_bad_lines='skip')
sub_UCLA_CNP = UCLA_CNP_demo['participant_id'].str.split('-', expand=True).iloc[:, -1]
sex_UCLA_CNP = UCLA_CNP_demo['gender']  # "1": "Male", "2": "Female"
age_UCLA_CNP = UCLA_CNP_demo['age']
dx_UCLA_CNP = UCLA_CNP_demo['diagnosis']

_, fc_idx, demo_idx = np.intersect1d(UCLA_CNP_fmri_sub, sub_UCLA_CNP, return_indices=True)
sex_UCLA_CNP = sex_UCLA_CNP.iloc[demo_idx].values
age_UCLA_CNP = age_UCLA_CNP.iloc[demo_idx].values
sub_UCLA_CNP = sub_UCLA_CNP.iloc[demo_idx].values
dx_UCLA_CNP = dx_UCLA_CNP.iloc[demo_idx].values
fc_UCLA_CNP = half_fc_UCLA_CNP[fc_idx]

dx_UCLA_CNP_cn = dx_UCLA_CNP[dx_UCLA_CNP == 'CONTROL']
sex_UCLA_CNP_cn = sex_UCLA_CNP[dx_UCLA_CNP == 'CONTROL']
age_UCLA_CNP_cn = age_UCLA_CNP[dx_UCLA_CNP == 'CONTROL']
sub_UCLA_CNP_cn = sub_UCLA_CNP[dx_UCLA_CNP == 'CONTROL']
fc_UCLA_CNP_cn = fc_UCLA_CNP[dx_UCLA_CNP == 'CONTROL']

UCLA_CNP_physio = pd.read_csv(r'I:\data\UCLA\phenotype\health.tsv', sep='\t', on_bad_lines='skip')
subject_UCLA_physio = UCLA_CNP_physio['participant_id'].str.split('-', expand=True).iloc[:, -1]
bmi_UCLA_physio = UCLA_CNP_physio['bmi']

_, fc_idx, bmi_idx = np.intersect1d(sub_UCLA_CNP_cn, subject_UCLA_physio, return_indices=True)
sub_UCLA_CNP_bmi = sub_UCLA_CNP_cn[fc_idx]
UCLA_CNP_bmi = bmi_UCLA_physio.iloc[bmi_idx].values

###########################################################
########################################### ABIDE I & II
###########################################################
mat = sio.loadmat(r'H:\PHD\learning\research\dataset\ABIDE\ABIDE_I_rsfMRI_ROIsignals_Schaefer100ROIs.mat')
sub_info_fc_abide1 = np.array([sub[0][0] for sub in mat['subID']])
sub_fc_abide1 = np.array([file.split('_')[0] for file in sub_info_fc_abide1]).astype(int)

ts_abide1 = [sub[0] for sub in mat['ROIsignals']]
FC_abide1 = []
for i, ts in enumerate(ts_abide1):
    print(i / len(ts_abide1))
    FC_abide1.append(get_fc(ts))
FC_abide1 = np.array(FC_abide1)
half_fc_abide1 = keep_triangle_half(FC_abide1.shape[1] * (FC_abide1.shape[1] - 1) // 2, FC_abide1.shape[0], FC_abide1)

demo_abide1 = pd.read_excel(r"H:\PHD\learning\research\dataset\ABIDE\ABIDE_I_Phenotypic.xlsx")
dx_abide1 = demo_abide1['DX_GROUP']
sub_abide1 = demo_abide1['SUB_ID']
age_abide1 = demo_abide1['AGE_AT_SCAN']
sex_abide1 = demo_abide1['SEX']
BMI_abide1 = demo_abide1['BMI']

sub_abide1_cn = sub_abide1[dx_abide1 == 2]
age_abide1_cn = age_abide1[dx_abide1 == 2]
sex_abide1_cn = sex_abide1[dx_abide1 == 2]
BMI_abide1_cn = BMI_abide1[dx_abide1 == 2]

_, fc_idx, demo_idx = np.intersect1d(sub_fc_abide1, sub_abide1_cn, return_indices=True)
sub_abide1_cn = sub_abide1_cn.iloc[demo_idx]
age_abide1_cn = age_abide1_cn.iloc[demo_idx]
sex_abide1_cn = sex_abide1_cn.iloc[demo_idx]
BMI_abide1_cn = BMI_abide1_cn.iloc[demo_idx]
fc_abide1_cn = half_fc_abide1[fc_idx]
sub_abide1_cn_bmi = sub_abide1_cn[(~np.isnan(BMI_abide1_cn)) & (BMI_abide1_cn != -9999)]
abide1_cn_bmi = BMI_abide1_cn[(~np.isnan(BMI_abide1_cn)) & (BMI_abide1_cn != -9999)]

mat = sio.loadmat(r'H:\PHD\learning\research\dataset\ABIDE\ABIDE_II_rsfMRI_ROIsignals_Schaefer100ROIs.mat')
sub_info_fc_abide2 = np.array([sub[0][0] for sub in mat['subID']])
sub_fc_abide2 = np.array([file.split('_')[0] for file in sub_info_fc_abide2]).astype(int)

ts_abide2 = [sub[0] for sub in mat['ROIsignals']]
FC_abide2 = []
for i, ts in enumerate(ts_abide2):
    print(i / len(ts_abide2))
    FC_abide2.append(get_fc(ts))
FC_abide2 = np.array(FC_abide2)
half_fc_abide2 = keep_triangle_half(FC_abide2.shape[1] * (FC_abide2.shape[1] - 1) // 2, FC_abide2.shape[0], FC_abide2)

demo_abide2 = pd.read_excel(r"H:\PHD\learning\research\dataset\ABIDE\ABIDE_II_Phenotypic.xlsx")
dx_abide2 = demo_abide2['DX_GROUP']
sub_abide2 = demo_abide2['SUB_ID']
age_abide2 = demo_abide2['AGE_AT_SCAN']
sex_abide2 = demo_abide2['SEX']
BMI_abide2 = demo_abide2['BMI']

sub_abide2_cn = sub_abide2[dx_abide2 == 2]
age_abide2_cn = age_abide2[dx_abide2 == 2]
sex_abide2_cn = sex_abide2[dx_abide2 == 2]
BMI_abide2_cn = BMI_abide2[dx_abide2 == 2]

_, fc_idx, demo_idx = np.intersect1d(sub_fc_abide2, sub_abide2_cn, return_indices=True)
sub_abide2_cn = sub_abide2_cn.iloc[demo_idx]
age_abide2_cn = age_abide2_cn.iloc[demo_idx]
sex_abide2_cn = sex_abide2_cn.iloc[demo_idx]
BMI_abide2_cn = BMI_abide2_cn.iloc[demo_idx]
fc_abide2_cn = half_fc_abide2[fc_idx]
sub_abide2_cn_bmi = sub_abide2_cn[(~np.isnan(BMI_abide2_cn)) & (BMI_abide2_cn != -9999)]
abide2_cn_bmi = BMI_abide2_cn[(~np.isnan(BMI_abide2_cn)) & (BMI_abide2_cn != -9999)]

###########################################################
########################################### SRPBS Japan
###########################################################
japan_fmri_info = sio.loadmat(r'J:\Alex_data\SPRBS_Japan\fMRIdata_SRPBS_fMRIPrep_struct.mat')
tvar = japan_fmri_info['fMRIdata_SRPBS_fMRIPrep_struct']
japan_fc_info = []
for idx in range(len(tvar)):
    japan_fc_info.append([tvar[idx][0]['subjectID'][0], tvar[idx][0]['site'][0]])
japan_fc_info = np.array(japan_fc_info)

japan_fc = []
for idx in range(len(tvar)):
    value = tvar[idx][0]['ROISignals_Schaefer100']
    japan_fc.append(get_fc(value))
japan_fc = np.array(japan_fc)

japan_pcd_info = sio.loadmat(r'J:\Alex_data\SPRBS_Japan\phenotypeTable_SRPBS_struct.mat')
tvar = japan_pcd_info['phenotypeTable_SRPBS_struct']
japan_PCD_data = []
pcd_keys = ['subjectID', 'diagnosis', 'age', 'sex']
for idx in range(len(tvar)):
    sub_data = []
    for name in pcd_keys:
        value = tvar[idx][0][name][0] if name == 'subjectID' else tvar[idx][0][name][0][0]
        sub_data.append(value)
    japan_PCD_data.append(sub_data)
japan_PCD_data = np.array(japan_PCD_data)

sub_japan_cn = japan_PCD_data[japan_PCD_data[:, 1] == '0', 0]
age_japan_cn = japan_PCD_data[japan_PCD_data[:, 1] == '0', 2]
sex_japan_cn = japan_PCD_data[japan_PCD_data[:, 1] == '0', 3]

_, fc_idx, demo_idx = np.intersect1d(japan_fc_info[:, 0], sub_japan_cn, return_indices=True)
sub_japan_cn = sub_japan_cn[demo_idx]
age_japan_cn = age_japan_cn[demo_idx].astype(float)
sex_japan_cn = sex_japan_cn[demo_idx]
japan_fc_cn = japan_fc[fc_idx]
japan_fc_cn_half = keep_triangle_half(japan_fc_cn.shape[1] * (japan_fc_cn.shape[1] - 1) // 2, japan_fc_cn.shape[0], japan_fc_cn)

###########################################################
########################################### Church data
###########################################################
data_path = r'H:\PHD\learning\research\dataset\church_data\ROISignals_AROMA_updated\ROISignals_AROMA'
all_file = os.listdir(data_path)
subjects = []
fc_all = []

for file in all_file:
    print(file)
    items = file.split('_')
    ID = items[0].split('-')[-1].split('T')[0]
    one_file_name = os.path.join(data_path, file)
    if 'rest' in one_file_name:
        ts = read_singal_fmri_ts(one_file_name)
        fc = get_fc(ts)
        fc_all.append(fc[:100, :100])
        subjects.append(ID)
subjects = np.array(subjects)
fc_all = np.array(fc_all)

church_info = pd.read_excel(r"H:\PHD\learning\research\dataset\church_data\kzhao_bmi_aging_20250807.xlsx", sheet_name="kzhao_bmi_aging")
sub_church = church_info['NDPNum']
age_church = church_info['Age']
sex_church = church_info['Sex']
BMI_church = church_info['BMI']
_, fc_idx, demo_idx = np.intersect1d(subjects, sub_church, return_indices=True)
sub_church_fmri = sub_church[demo_idx]
age_church = age_church[demo_idx].astype(float)
sex_church = sex_church[demo_idx]
BMI_church = BMI_church[demo_idx]
fc_all_church = fc_all[fc_idx]
fc_church_half = keep_triangle_half(fc_all_church.shape[1] * (fc_all_church.shape[1] - 1) // 2, fc_all_church.shape[0], fc_all_church)

###########################################################
########################################### Pool cohorts, build discovery / independent splits
###########################################################
all_sub_sess_cn = np.r_['UCLA_' + sub_UCLA_CNP_cn, sub_abide1_cn, sub_abide2_cn, sub_japan_cn].astype(str)
all_site_cn = np.r_[np.zeros(sub_UCLA_CNP_cn.shape[0]) + 1,
                     np.zeros(sub_abide1_cn.shape[0]) + 2, np.zeros(sub_abide2_cn.shape[0]) + 3, np.zeros(sub_japan_cn.shape[0]) + 4]
all_fc_cn = np.r_[fc_UCLA_CNP_cn, fc_abide1_cn, fc_abide2_cn, japan_fc_cn_half]
scaler = StandardScaler()
all_fc_cn_zscores = scaler.fit_transform(all_fc_cn.T).T
all_sex_cn = np.r_[sex_UCLA_CNP_cn, sex_abide1_cn, sex_abide2_cn, sex_japan_cn]
all_age_cn = np.r_[age_UCLA_CNP_cn, age_abide1_cn, age_abide2_cn, age_japan_cn]
all_sub_sess_bmi = np.r_['UCLA_' + sub_UCLA_CNP_bmi, sub_abide1_cn_bmi, sub_abide2_cn_bmi].astype(str)
all_bmi_cn = np.r_[UCLA_CNP_bmi, abide1_cn_bmi, abide2_cn_bmi].astype(float)

all_age_10 = all_age_cn[all_age_cn >= 18]
all_sex_10 = all_sex_cn[all_age_cn >= 18]
all_fc_zscores_10 = all_fc_cn_zscores[all_age_cn >= 18]
all_site_10 = all_site_cn[all_age_cn >= 18]
all_sub_sess_10 = all_sub_sess_cn[all_age_cn >= 18]

church_age_10 = age_church[age_church >= 18]
church_sub_10 = sub_church_fmri[age_church >= 18]
church_sex_10 = sex_church[age_church >= 18]
fc_church_half_10 = fc_church_half[age_church >= 18]
BMI_church_10 = BMI_church[age_church >= 18]
fc_church_half_zscore = scaler.fit_transform(fc_church_half_10.T).T

all_sub_sess_cn_nobmi = np.setdiff1d(all_sub_sess_cn, all_sub_sess_bmi)
_, no_bmi_idx, _ = np.intersect1d(all_sub_sess_10, all_sub_sess_cn_nobmi, return_indices=True)
_, bmi_idx, bmi_idx2 = np.intersect1d(all_sub_sess_cn, all_sub_sess_bmi, return_indices=True)

all_fc_cn_nobmi = all_fc_zscores_10[no_bmi_idx]
all_sub_sess_cn_nobmi = all_sub_sess_10[no_bmi_idx]
all_sex_nobmi = all_sex_10[no_bmi_idx]
all_age_nobmi = all_age_10[no_bmi_idx]
all_site_cn_nobmi = all_site_10[no_bmi_idx]

all_fc_cn_bmi = all_fc_cn_zscores[bmi_idx]
all_sub_sess_cn_bmi = all_sub_sess_cn[bmi_idx]
all_sex_bmi = all_sex_cn[bmi_idx]
all_age_bmi = all_age_cn[bmi_idx]
all_site_bmi = all_site_cn[bmi_idx]
all_bmi = all_bmi_cn[bmi_idx2]

all_fc_bmi_10 = all_fc_cn_bmi[all_age_bmi >= 18]
all_sub_sess_bmi_10 = all_sub_sess_cn_bmi[all_age_bmi >= 18]
all_sex_bmi_10 = all_sex_bmi[all_age_bmi >= 18]
all_age_bmi_10 = all_age_bmi[all_age_bmi >= 18]
all_site_bmi_10 = all_site_bmi[all_age_bmi >= 18]
all_bmi_10 = all_bmi[all_age_bmi >= 18]

# Discovery / independent split: subjects without BMI are split into
# discovery vs. independent (last 370 of the shuffled no-BMI subjects go to
# independent); all subjects with BMI go to the independent set as well.
num_samples = len(all_sub_sess_cn_nobmi)
seed = 1
indices = np.random.RandomState(seed=seed).permutation(num_samples)
train_indices = indices[:-370]
test_indices = indices[-370:]

discovery_fc_cn_nobmi = all_fc_cn_nobmi[train_indices]
discovery_sub_sess_cn_nobmi = all_sub_sess_cn_nobmi[train_indices]
discovery_sex_nobmi = all_sex_nobmi[train_indices]
discovery_age_nobmi = all_age_nobmi[train_indices]
discovery_site_nobmi = all_site_cn_nobmi[train_indices]

independent_fc_cn = np.r_[all_fc_cn_nobmi[test_indices], all_fc_bmi_10]
independent_sub_sess_cn = np.r_[all_sub_sess_cn_nobmi[test_indices], all_sub_sess_bmi_10]
independent_sex = np.r_[all_sex_nobmi[test_indices], all_sex_bmi_10]
independent_age = np.r_[all_age_nobmi[test_indices], all_age_bmi_10]
independent_site = np.r_[all_site_cn_nobmi[test_indices], all_site_bmi_10]
independent_BMI = np.r_[np.full(test_indices.shape[0], np.nan), all_bmi_10]

###########################################################
########################################### 10-fold CV brain-age model
###########################################################
kf = KFold(n_splits=10, shuffle=True, random_state=66)
r2_scores_discovery = []
r2_scores_ind = []
r2_scores_church = []
y_true_all_discovery = []
test_idx_all_discovery = []
y_pred_all_ind = []
y_pred_all_church = []
y_pred_all_discovery = []
r_p_vector_all = []
model_all = []
model = BayesianRidge()

n_edges = discovery_fc_cn_nobmi.shape[1]

for j, (train_index, test_index) in enumerate(kf.split(discovery_fc_cn_nobmi)):
    print(j)
    X_train, X_test = discovery_fc_cn_nobmi[train_index], discovery_fc_cn_nobmi[test_index]
    y_train, y_test = discovery_age_nobmi[train_index], discovery_age_nobmi[test_index]

    # Feature selection on training data only (no leakage): keep FC edges
    # whose Pearson correlation with age is significant at p <= 0.01.
    r_p_vector = np.zeros((n_edges, 2))
    for i in range(n_edges):
        r_p_vector[i, 0], r_p_vector[i, 1] = pearsonr(X_train[:, i], y_train)
    r_p_vector_all.append(r_p_vector)
    feat_mask = r_p_vector[:, 1] <= 0.01

    X_train = X_train[:, feat_mask]
    X_test = X_test[:, feat_mask]
    independent_fc_cn_ = independent_fc_cn[:, feat_mask]
    fc_church_half_zscore_ = fc_church_half_zscore[:, feat_mask]

    model.fit(X_train, y_train)
    model_all.append(model)

    y_pred = model.predict(X_test)
    r2_scores_discovery.append(r2_score(y_test, y_pred))
    y_true_all_discovery.extend(y_test)
    y_pred_all_discovery.extend(y_pred)
    test_idx_all_discovery.extend(list(test_index))

    y_pred_ind = model.predict(independent_fc_cn_)
    y_pred_all_ind.append(y_pred_ind)
    r2_scores_ind.append(r2_score(independent_age, y_pred_ind))

    y_pred_church = model.predict(fc_church_half_zscore_)
    y_pred_all_church.append(y_pred_church)
    r2_scores_church.append(r2_score(church_age_10, y_pred_church))

print(np.mean(r2_scores_discovery))
print(np.mean(r2_scores_ind))

y_true_all_discovery = np.array(y_true_all_discovery)
y_pred_all_discovery = np.array(y_pred_all_discovery)
y_pred_all_ind = np.array(y_pred_all_ind)
y_pred_all_church = np.array(y_pred_all_church)
test_idx_all_discovery = np.array(test_idx_all_discovery)
r_p_vector_all = np.array(r_p_vector_all)

print(r2_score(y_true_all_discovery[y_true_all_discovery < 65], y_pred_all_discovery[y_true_all_discovery < 65]))
print(pearsonr(y_true_all_discovery[y_true_all_discovery < 65], y_pred_all_discovery[y_true_all_discovery < 65]))
print(r2_score(independent_age[independent_age < 65], y_pred_all_ind.mean(0)[independent_age < 65]))
print(pearsonr(independent_age[independent_age < 65], y_pred_all_ind.mean(0)[independent_age < 65]))
print(r2_score(church_age_10[church_age_10 >= 18], y_pred_all_church.mean(0)[church_age_10 >= 18]))
print(pearsonr(church_age_10[church_age_10 >= 18], y_pred_all_church.mean(0)[church_age_10 >= 18]))

###########################################################
########################################### Scatter plots: true vs. predicted age
###########################################################
def scatter_true_vs_pred(x, y, color, save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.regplot(x=x, y=y, scatter_kws={"s": 150, 'color': color, 'alpha': 0.6},
                line_kws={"color": color, 'lw': 4}, ax=ax)
    ax.set_yticks(ticks=[10, 30, 50, 70])
    ax.set_yticklabels(labels=[10, 30, 50, 70], fontsize=30, fontfamily="Tahoma", fontweight="bold")
    ax.set_xticks(ticks=[10, 30, 50, 70])
    ax.set_xticklabels(labels=[10, 30, 50, 70], fontsize=30, fontfamily="Tahoma", fontweight="bold")
    ax.set_xlabel("Chronological age", **font2)
    ax.set_ylabel("Predicted Brain age", **font2)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.grid(False)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


scatter_true_vs_pred(y_true_all_discovery[y_true_all_discovery < 65], y_pred_all_discovery[y_true_all_discovery < 65], color1)
scatter_true_vs_pred(independent_age[independent_age < 65], y_pred_all_ind.mean(0)[independent_age < 65], color3)
scatter_true_vs_pred(church_age_10, y_pred_all_church.mean(0), color2)

###########################################################
########################################### Brain-age gap (BAG) and age-bias correction
###########################################################
# BAG_raw = true age - predicted age. Age-bias correction follows the
# standard approach (e.g. Le et al. 2018): regress predicted age on true
# age within a cohort, then define BAG_corrected as the residual of that
# fit (predicted - fitted), which removes the regression-to-the-mean
# correlation between BAG_raw and true age.
def age_bias_correct(true_age, pred_age):
    reg = LinearRegression().fit(np.expand_dims(true_age, 1), pred_age)
    bag_raw = true_age - pred_age
    bag_corrected = pred_age - reg.predict(np.expand_dims(true_age, 1))
    return bag_raw, bag_corrected


bag_discovery, bag_discovery_correct = age_bias_correct(y_true_all_discovery, y_pred_all_discovery)
bag_independent, bag_independent_correct = age_bias_correct(independent_age, y_pred_all_ind.mean(0))
bag_church, bag_church_correct = age_bias_correct(church_age_10, y_pred_all_church.mean(0))

print('discovery BAG vs age, raw:', pearsonr(bag_discovery, y_true_all_discovery))
print('discovery BAG vs age, corrected:', pearsonr(bag_discovery_correct, y_true_all_discovery))
print('independent BAG vs age, raw:', pearsonr(bag_independent, independent_age))
print('independent BAG vs age, corrected:', pearsonr(bag_independent_correct, independent_age))
print('church BAG vs age, raw:', pearsonr(bag_church, church_age_10))
print('church BAG vs age, corrected:', pearsonr(bag_church_correct, church_age_10))

# sio.savemat(r'H:\postdoc\UCLA_postdoc\aging_BMI\with_church_data\result\BAG_sub_adult_info_seed{}_withchurch.mat'.format(seed), {
#     'discovery_fc_cn_nobmi': discovery_fc_cn_nobmi, 'discovery_predict_index': test_idx_all_discovery,
#     'discovery_sub_sess_cn_nobmi': discovery_sub_sess_cn_nobmi, 'discovery_sex_nobmi': discovery_sex_nobmi,
#     'discovery_age_nobmi': discovery_age_nobmi, 'discovery_site_nobmi': discovery_site_nobmi,
#     'bag_discovery': bag_discovery, 'bag_discovery_correct': bag_discovery_correct, 'brainage_pred_all_discovery': y_pred_all_discovery,
#     'independent_fc_cn': independent_fc_cn, 'independent_sub_sess_cn': independent_sub_sess_cn,
#     'independent_sex': independent_sex, 'independent_age': independent_age,
#     'independent_site': independent_site, 'independent_BMI': independent_BMI,
#     'bag_independent': bag_independent, 'bag_independent_correct': bag_independent_correct,
#     'brainage_pred_all_ind': y_pred_all_ind, 'r_p_vectors': r_p_vector_all,
#     'bag_church': bag_church, 'bag_church_correct': bag_church_correct,
#     'BMI_church_10': BMI_church_10, 'fc_church_ahlf_10': fc_church_half_zscore,
#     'church_age_10': church_age_10, 'church_sub_10': church_sub_10,
#     'church_sex_10': church_sex_10, 'brainage_pred_all_church': y_pred_all_church})
