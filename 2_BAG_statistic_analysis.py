

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import sys
sys.path.append(r'H:\utils_for_all')
from common_utils import setup_seed

font2 = {'family': 'Tahoma', 'weight': 'bold', 'size': 40}
matplotlib.rc('font', **font2)
setup_seed(6)

RGB = namedtuple('RGB', 'red, green, blue')
color1 = RGB(0.52, 0.52, 1)
color3 = RGB(0.4, 0.4, 0.67)

result_dir = r'H:\postdoc\UCLA_postdoc\aging_BMI\with_church_data\result'
seed = 1

# Clinical variables of interest in each cohort -- these ARE keys_plot,
# key_ind, and key_church from the original script, and drive both the
# trimmed phenotype loading below and the final correlation table/plots.
clinical_discovery = ['FIQ', 'VIQ', 'PIQ', 'ADOS_TOTAL', 'ADOS_SOCIAL', 'ADOS_STEREO_BEHAV',
                      'BDI_II', 'AQ_total_', 'AQ_ss_', 'AQ_as_', 'AQ_atd_', 'AQ_Com_', 'AQ_imag_',
                      'BACSVerbalMemory', 'BACSWorkingMemory_DigitSequencing_', 'BACSMotorSpeed_TokenMotorTask_',
                      'BACSVerbalFluency', 'BACSAttentionAndProcessingSpeed_Symbol_codingTask_',
                      'BACSExecutiveFunction_TowerOfLondon_']
clinical_independent = ['chaphypo_total', 'chapinf_total', 'chapper_total', 'chapphy_total', 'chapsoc_total',
                        'cvlt_totcor', 'cvltz_10', 'cvltz_12', 'cvltz_16', 'cvltz_17', 'cvltz_18', 'cvltz_19']
clinical_church = ['SI_Social_Disconnectedness_Score', 'SI_Lack_Social_Support_Score', 'SI_Perceived_Loneliness_Score',
                   'TRAILA_Time', 'TRAILB_Time', 'HAD_Anxiety', 'HAD_Depression', 'STAI_SAnxiety', 'STAI_TAnxiety']

###########################################################
########################################### UCLA phenotype (clinical_independent only)
###########################################################
ucla_path = r'F:\data\UCLA\phenotype'
ucla_files = {
    'chaphyp.tsv': ['chaphypo_total'],
    'chapinf.tsv': ['chapinf_total'],
    'chapper.tsv': ['chapper_total'],
    'chapphy.tsv': ['chapphy_total'],
    'chapsoc.tsv': ['chapsoc_total'],
    'cvlt.tsv': ['cvlt_totcor', 'cvltz_10', 'cvltz_12', 'cvltz_16', 'cvltz_17', 'cvltz_18', 'cvltz_19'],
}
pcd_UCLA_select = None
for fname, cols in ucla_files.items():
    df = pd.read_csv(os.path.join(ucla_path, fname), sep='\t', on_bad_lines='skip')[['participant_id'] + cols]
    pcd_UCLA_select = df if pcd_UCLA_select is None else pcd_UCLA_select.merge(df, on='participant_id', how='outer')
pcd_UCLA_select['participant_id'] = 'UCLA_' + pcd_UCLA_select['participant_id'].str.split('-', expand=True).iloc[:, -1]

###########################################################
########################################### ABIDE phenotype (FIQ/VIQ/PIQ/ADOS_* only)
###########################################################
abide_path = r'J:\Alex_data\ABIDE'
abide1_info = pd.read_excel(os.path.join(abide_path, 'ABIDE_I_Phenotypic.xlsx'))
abide1_pcd = abide1_info[['SUB_ID', 'FIQ', 'VIQ', 'PIQ', 'ADOS_TOTAL', 'ADOS_SOCIAL', 'ADOS_STEREO_BEHAV']]

abide2_info = pd.read_excel(os.path.join(abide_path, 'ABIDE_II_Phenotypic.xlsx'))
abide2_pcd = abide2_info[['SUB_ID', 'FIQ', 'VIQ', 'PIQ', 'ADOS_G_TOTAL', 'ADOS_G_SOCIAL', 'ADOS_G_STEREO_BEHAV']]
abide2_pcd = abide2_pcd.rename(columns={'ADOS_G_TOTAL': 'ADOS_TOTAL', 'ADOS_G_SOCIAL': 'ADOS_SOCIAL',
                                         'ADOS_G_STEREO_BEHAV': 'ADOS_STEREO_BEHAV'})

abide_pcd = pd.merge(abide1_pcd, abide2_pcd,
                     on=['SUB_ID', 'FIQ', 'VIQ', 'PIQ', 'ADOS_TOTAL', 'ADOS_SOCIAL', 'ADOS_STEREO_BEHAV'],
                     how='outer')
abide_pcd[abide_pcd <= -99] = np.nan
abide_pcd = abide_pcd.rename(columns={'SUB_ID': 'participant_id'})
abide_pcd['participant_id'] = abide_pcd['participant_id'].astype(str)

###########################################################
########################################### SRPBS Japan phenotype (BDI_II/AQ_*/BACS* only)
###########################################################
japan_pcd_info = sio.loadmat(r'J:\Alex_data\SPRBS_Japan\phenotypeTable_SRPBS_struct.mat')
tvar = japan_pcd_info['phenotypeTable_SRPBS_struct']
pcd_keys_japan = ['subjectID', 'FIQ', 'BDI_II', 'AQ_total_', 'AQ_ss_', 'AQ_as_', 'AQ_atd_', 'AQ_Com_', 'AQ_imag_',
                  'BACSVerbalMemory', 'BACSWorkingMemory_DigitSequencing_', 'BACSMotorSpeed_TokenMotorTask_',
                  'BACSVerbalFluency', 'BACSAttentionAndProcessingSpeed_Symbol_codingTask_',
                  'BACSExecutiveFunction_TowerOfLondon_']

japan_PCD_data = []
for idx in range(len(tvar)):
    sub_data = []
    for name in pcd_keys_japan:
        value = tvar[idx][0][name][0] if name == 'subjectID' else tvar[idx][0][name][0][0]
        sub_data.append(value)
    japan_PCD_data.append(sub_data)
japan_PCD_df = pd.DataFrame(np.array(japan_PCD_data), columns=pcd_keys_japan)
japan_PCD_df = japan_PCD_df.rename(columns={'subjectID': 'participant_id'})
japan_PCD_df['FIQ'] = japan_PCD_df['FIQ'].astype(float)

###########################################################
########################################### Church phenotype (clinical_church only)
###########################################################
church_path = r'H:\PHD\learning\research\dataset\church_data'
church_info = pd.read_excel(os.path.join(church_path, 'kzhao_bmi_aging_20250807.xlsx'), sheet_name="kzhao_bmi_aging")
church_pcd = church_info[['NDPNum'] + clinical_church].copy()
church_pcd = church_pcd.rename(columns={'NDPNum': 'participant_id'})

###########################################################
########################################### Merge phenotypes, load BAG info
###########################################################
pcd_all = pd.merge(pcd_UCLA_select, abide_pcd, on='participant_id', how='outer')
pcd_all = pd.merge(pcd_all, japan_PCD_df, on=['participant_id', 'FIQ'], how='outer')
pcd_all = pd.merge(pcd_all, church_pcd, on=['participant_id'], how='outer')

bag_info = sio.loadmat(os.path.join(result_dir, 'BAG_sub_adult_info_seed{}_withchurch.mat'.format(seed)))

discovery_index_nobmi = bag_info['discovery_predict_index'].squeeze()
discovery_fc_cn_nobmi = bag_info['discovery_fc_cn_nobmi']
discovery_sub_sess_cn_nobmi = pd.Series(bag_info['discovery_sub_sess_cn_nobmi']).str.split(' ', expand=True).iloc[:, 0]

discovery_age_nobmi = bag_info['discovery_age_nobmi'].squeeze()
discovery_brainage_nobmi = bag_info['brainage_pred_all_discovery'].squeeze()
discovery_age_nobmi_resort = discovery_age_nobmi[discovery_index_nobmi]
discovery_sub_sess_cn_nobmi_resort = discovery_sub_sess_cn_nobmi[discovery_index_nobmi]

bag_discovery_correct = bag_info['bag_discovery_correct'].squeeze()
bag_discovery_correct_adult = bag_discovery_correct[discovery_age_nobmi_resort >= 18]
discovery_sub_sess_cn_nobmi_adult = discovery_sub_sess_cn_nobmi_resort[discovery_age_nobmi_resort >= 18]

independent_sub_sess_cn = pd.Series(bag_info['independent_sub_sess_cn']).str.split(' ', expand=True).iloc[:, 0]
independent_age = bag_info['independent_age'].squeeze()
bag_independent_correct = bag_info['bag_independent_correct'].squeeze()

bag_independent_correct_adult = bag_independent_correct[independent_age >= 18]
independent_sub_sess_cn_adult = independent_sub_sess_cn[independent_age >= 18]

church_sub_cn = np.array([sub[0] for sub in bag_info['church_sub_10'].squeeze()])
bag_church_correct = bag_info['bag_church_correct'].squeeze()

pcd_all_values = pcd_all.iloc[:, 1:].astype(float)
pheno_keys = np.array(pcd_all_values.keys())

_, pcd_disc_idx, fc_disc_idx = np.intersect1d(pcd_all['participant_id'], discovery_sub_sess_cn_nobmi_adult, return_indices=True)
_, pcd_ind_idx, fc_ind_idx = np.intersect1d(pcd_all['participant_id'], independent_sub_sess_cn_adult, return_indices=True)
_, pcd_church_idx, fc_church_idx = np.intersect1d(pcd_all['participant_id'], church_sub_cn, return_indices=True)

# ===========================================================================
# CORRELATION ANALYSIS FOR THE CLINICAL VARIABLES OF INTEREST (per cohort)
# ===========================================================================

scaler = StandardScaler()
bag_discovery_correct_adult_zscore = scaler.fit_transform(np.expand_dims(bag_discovery_correct_adult, 1)).squeeze()
bag_independent_correct_adult_zscore = scaler.fit_transform(np.expand_dims(bag_independent_correct_adult, 1)).squeeze()

bag_disc_correct = bag_discovery_correct_adult_zscore[fc_disc_idx]
bag_ind_correct = bag_independent_correct_adult_zscore[fc_ind_idx]
bag_church_correct_ = bag_church_correct[fc_church_idx]


def clinical_corr(cohort, var_list, pcd_idx, bag_correct):
    """Pearson correlation between each clinical variable and the corrected BAG.

    Returns a DataFrame with the correlation, raw p-value, FDR-corrected p-value
    (within this cohort's clinical set) and the n of non-missing subjects.
    """
    rows = []
    for var in var_list:
        loc = np.where(pheno_keys == var)[0]
        if len(loc) == 0:
            rows.append([var, np.nan, np.nan, 0])
            continue
        value = pcd_all_values.iloc[pcd_idx, loc[0]]
        nonan = ~np.isnan(value)
        if nonan.sum() >= 20:
            r, p = pearsonr(value[nonan], bag_correct[nonan])
            rows.append([var, r, p, int(nonan.sum())])
        else:
            rows.append([var, np.nan, np.nan, int(nonan.sum())])
    df = pd.DataFrame(rows, columns=['variable', 'r', 'p', 'n'])
    df.insert(0, 'cohort', cohort)
    valid = df['p'].notna()
    df['p_fdr'] = np.nan
    if valid.sum() > 0:
        df.loc[valid, 'p_fdr'] = multipletests(df.loc[valid, 'p'], method='fdr_bh')[1]
    return df


res_disc = clinical_corr('discovery', clinical_discovery, pcd_disc_idx, bag_disc_correct)
res_ind = clinical_corr('independent', clinical_independent, pcd_ind_idx, bag_ind_correct)
res_church = clinical_corr('church', clinical_church, pcd_church_idx, bag_church_correct_)

clinical_results = pd.concat([res_disc, res_ind, res_church], ignore_index=True)

print('\n===== Clinical BAG correlation results =====')
print(clinical_results.to_string(index=False))

clinical_csv_path = os.path.join(result_dir, 'BAG_clinical_correlation_results_seed{}.csv'.format(seed))
clinical_results.to_csv(clinical_csv_path, index=False)
print('\nSaved clinical correlation results to {}'.format(clinical_csv_path))

print('\nDiscovery model fit:')
print(pearsonr(discovery_age_nobmi_resort, discovery_brainage_nobmi))

# ===========================================================================
# SCATTER PLOTS: BAG vs. each clinical variable (per cohort)
# ===========================================================================
def scatter_bag_vs_measure(bag_x, value_y, label, color, save_path):
    nonan = ~np.isnan(value_y)
    if nonan.sum() < 20:
        return
    x = bag_x[nonan]
    y = value_y[nonan]

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.regplot(x=x, y=y, scatter_kws={"s": 300, 'color': color, 'alpha': 0.6},
                line_kws={"color": color, 'lw': 4}, ax=ax)
    ax.set_xticks([-3.0, -1.5, 0, 1.5, 3.0])
    plt.xlim(-3.5, 3.5)

    unity = (np.max(y) - np.min(y)) // 2
    if abs(unity) < 1:
        unity = round((np.max(y) - np.min(y)) / 2, 2)
        ax.set_yticks(np.arange(np.min(y) - unity, np.max(y) + unity, unity))
    else:
        ax.set_yticks(np.arange(np.min(y) - unity, np.max(y) + unity, unity).astype(int))

    ax.set_xlabel("BAG", **font2)
    ax.set_ylabel(label, **font2)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.grid(False)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


for var in clinical_discovery:
    loc = np.where(pheno_keys == var)[0]
    if len(loc) == 0:
        continue
    value_disc = pcd_all_values.iloc[pcd_disc_idx, loc[0]].values
    scatter_bag_vs_measure(
        bag_disc_correct, value_disc, var, color1,
        os.path.join(result_dir, 'bag_correct_discovery_adultmodel_seed{}_scatter_{}.svg'.format(seed, var)))

for var in clinical_independent:
    loc = np.where(pheno_keys == var)[0]
    if len(loc) == 0:
        continue
    value_ind = pcd_all_values.iloc[pcd_ind_idx, loc[0]].values
    scatter_bag_vs_measure(
        bag_ind_correct, value_ind, var, color3,
        os.path.join(result_dir, 'bag_correct_replication_adultmodel_seed{}_scatter_{}.svg'.format(seed, var)))

for var in clinical_church:
    loc = np.where(pheno_keys == var)[0]
    if len(loc) == 0:
        continue
    value_church = pcd_all_values.iloc[pcd_church_idx, loc[0]].values
    scatter_bag_vs_measure(
        bag_church_correct_, value_church, var, color3,
        os.path.join(result_dir, 'bag_correct_church_adultmodel_seed{}_scatter_{}.svg'.format(seed, var)))

# ===========================================================================
# SUMMARY BAR PLOTS: r-value per clinical variable (per cohort)
# ===========================================================================
def bar_summary(res, title, color, save_path):
    fig, ax = plt.subplots(figsize=(max(9, len(res)), 9))
    ax.bar(range(len(res)), res['r'], color=color, alpha=0.7)
    ax.set_xticks(range(len(res)))
    ax.set_xticklabels(res['variable'], rotation=45, ha='right')
    ax.set_yticks([-0.50, -0.25, 0, 0.25, 0.50])
    ax.set_yticklabels([-0.50, -0.25, 0, 0.25, 0.50], fontsize=30, fontfamily="Tahoma", fontweight="bold")
    ax.grid(axis='y', linestyle='--', color='gray', alpha=1)
    ax.set_ylabel('Correlation Coefficients (r)', fontsize=30)
    ax.set_title(title)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


bar_summary(res_disc, 'Discovery', color1,
            os.path.join(result_dir, 'bag_correct_discovery_adultmodel_seed{}_bar.svg'.format(seed)))
bar_summary(res_ind, 'Independent', color3,
            os.path.join(result_dir, 'bag_correct_replication_adultmodel_seed{}_bar.svg'.format(seed)))
bar_summary(res_church, 'Church', color3,
            os.path.join(result_dir, 'bag_correct_church_adultmodel_seed{}_bar.svg'.format(seed)))
