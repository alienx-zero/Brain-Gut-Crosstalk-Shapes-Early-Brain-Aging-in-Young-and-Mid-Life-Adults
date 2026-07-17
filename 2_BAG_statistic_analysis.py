

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import sys
sys.path.append(r'H:\utils_for_all')
from common_utils import setup_seed

setup_seed(6)

result_dir = r'H:\postdoc\UCLA_postdoc\aging_BMI\with_church_data\result'
seed = 1

###########################################################
########################################### UCLA phenotype
###########################################################
ucla_path = r'F:\data\UCLA\phenotype'

UCLA_adhd = pd.read_csv(os.path.join(ucla_path, 'adhd.tsv'), sep='\t', on_bad_lines='skip')
UCLA_asrs = pd.read_csv(os.path.join(ucla_path, 'asrs.tsv'), sep='\t', on_bad_lines='skip')
UCLA_barratt = pd.read_csv(os.path.join(ucla_path, 'barratt.tsv'), sep='\t', on_bad_lines='skip')
UCLA_bipolar_ii = pd.read_csv(os.path.join(ucla_path, 'bipolar_ii.tsv'), sep='\t', on_bad_lines='skip')
UCLA_chaphyp = pd.read_csv(os.path.join(ucla_path, 'chaphyp.tsv'), sep='\t', on_bad_lines='skip')
UCLA_chapinf = pd.read_csv(os.path.join(ucla_path, 'chapinf.tsv'), sep='\t', on_bad_lines='skip')
UCLA_chapper = pd.read_csv(os.path.join(ucla_path, 'chapper.tsv'), sep='\t', on_bad_lines='skip')
UCLA_chapphy = pd.read_csv(os.path.join(ucla_path, 'chapphy.tsv'), sep='\t', on_bad_lines='skip')
UCLA_chapsoc = pd.read_csv(os.path.join(ucla_path, 'chapsoc.tsv'), sep='\t', on_bad_lines='skip')
UCLA_colortrails = pd.read_csv(os.path.join(ucla_path, 'colortrails.tsv'), sep='\t', on_bad_lines='skip')
UCLA_cvlt = pd.read_csv(os.path.join(ucla_path, 'cvlt.tsv'), sep='\t', on_bad_lines='skip')
UCLA_dickman = pd.read_csv(os.path.join(ucla_path, 'dickman.tsv'), sep='\t', on_bad_lines='skip')
UCLA_golden = pd.read_csv(os.path.join(ucla_path, 'golden.tsv'), sep='\t', on_bad_lines='skip')
UCLA_eysenck = pd.read_csv(os.path.join(ucla_path, 'eysenck.tsv'), sep='\t', on_bad_lines='skip')
UCLA_hamilton = pd.read_csv(os.path.join(ucla_path, 'hamilton.tsv'), sep='\t', on_bad_lines='skip')
UCLA_health = pd.read_csv(os.path.join(ucla_path, 'health.tsv'), sep='\t', on_bad_lines='skip')
UCLA_hopkins = pd.read_csv(os.path.join(ucla_path, 'hopkins.tsv'), sep='\t', on_bad_lines='skip')
UCLA_mpq = pd.read_csv(os.path.join(ucla_path, 'mpq.tsv'), sep='\t', on_bad_lines='skip')
UCLA_rk = pd.read_csv(os.path.join(ucla_path, 'rk.tsv'), sep='\t', on_bad_lines='skip')
UCLA_sans = pd.read_csv(os.path.join(ucla_path, 'sans.tsv'), sep='\t', on_bad_lines='skip')
UCLA_saps = pd.read_csv(os.path.join(ucla_path, 'saps.tsv'), sep='\t', on_bad_lines='skip')
UCLA_wms = pd.read_csv(os.path.join(ucla_path, 'wms.tsv'), sep='\t', on_bad_lines='skip')
UCLA_wais = pd.read_csv(os.path.join(ucla_path, 'wais.tsv'), sep='\t', on_bad_lines='skip')
UCLA_vcap = pd.read_csv(os.path.join(ucla_path, 'vcap.tsv'), sep='\t', on_bad_lines='skip')
UCLA_demo = pd.read_csv(os.path.join(ucla_path, 'demographics.tsv'), sep='\t', on_bad_lines='skip')
UCLA_hand = pd.read_csv(os.path.join(ucla_path, 'handedness.tsv'), sep='\t', on_bad_lines='skip')

df_merged = UCLA_adhd.merge(UCLA_asrs, on='participant_id', how='outer') \
                      .merge(UCLA_barratt, on='participant_id', how='outer') \
                      .merge(UCLA_bipolar_ii, on='participant_id', how='outer') \
                      .merge(UCLA_chaphyp, on='participant_id', how='outer') \
                      .merge(UCLA_chapinf, on='participant_id', how='outer') \
                      .merge(UCLA_chapper, on='participant_id', how='outer') \
                      .merge(UCLA_chapphy, on='participant_id', how='outer') \
                      .merge(UCLA_chapsoc, on='participant_id', how='outer') \
                      .merge(UCLA_colortrails, on='participant_id', how='outer') \
                      .merge(UCLA_cvlt, on='participant_id', how='outer') \
                      .merge(UCLA_dickman, on='participant_id', how='outer') \
                      .merge(UCLA_golden, on='participant_id', how='outer') \
                      .merge(UCLA_eysenck, on='participant_id', how='outer') \
                      .merge(UCLA_hamilton, on='participant_id', how='outer') \
                      .merge(UCLA_health, on='participant_id', how='outer') \
                      .merge(UCLA_hopkins, on='participant_id', how='outer') \
                      .merge(UCLA_mpq, on='participant_id', how='outer') \
                      .merge(UCLA_rk, on='participant_id', how='outer') \
                      .merge(UCLA_sans, on='participant_id', how='outer') \
                      .merge(UCLA_saps, on='participant_id', how='outer') \
                      .merge(UCLA_wms, on='participant_id', how='outer') \
                      .merge(UCLA_wais, on='participant_id', how='outer') \
                      .merge(UCLA_demo, on='participant_id', how='outer') \
                      .merge(UCLA_vcap, on='participant_id', how='outer') \
                      .merge(UCLA_hand, on='participant_id', how='outer')

pcd_UCLA_select = df_merged[['participant_id', 'adhd5', 'adhd4', 'adhd7', 'adhd2', 'adhd8', 'adhd6', 'adhd1', 'adhd3',
                             'asrs_score',
                             'bis_2attimp', 'bis_2motimp', 'bis_2npimp',
                            'bipollarii_mood', 'bipollarii_daydreaming', 'bipollarii_sumscore', 'bipollarii_energy', 'bipollarii_anxiety',
                            'chaphypo_total', 'chapinf_total', 'chapper_total', 'chapphy_total', 'chapsoc_total',
                            'crt_index',
                            'cvlt_totcor', 'cvltz_10', 'cvltz_12', 'cvltz_16', 'cvltz_17', 'cvltz_18', 'cvltz_19',
                            'func_neg', 'func_pos', 'func_total', 'dysfunc_neg', 'dysfunc_pos', 'dysfunc_total',
                            'golden_sumscore',
                            'scorev', 'scoree', 'scorei',
                            'hamd_28', 'hamd_21', 'hamd_17',
                            'la2khealth1', 'la2khealth2', 'la2khealth3', 'la2khealth4', 'la2khealth5', 'la2khealth6', 'la2khealth7',
                            'la2khealth8', 'la2khealth9', 'la2khealth10', 'la2khealth11', 'la2khealth13', 'la2khealth14', 'la2khealth15', 'la2khealth16',
                            'la2khealth17', 'la2khealth18', 'la2khealth19', 'la2khealth20', 'la2khealth21',
                            'hopkins_obscomp', 'hopkins_anxiety', 'hopkins_somatization', 'hopkins_intsensitivity',
                            'hopkins_globalseverity', 'hopkins_depression',
                            'mpq_score', 'rk_percentktwocorrect',
                            'global_bluntaffect', 'global_anhedonia', 'global_alogia', 'factor_attention', 'global_avolition',
                            'factor_bluntaffect', 'factor_anhedonia', 'factor_alogia', 'factor_attention', 'factor_avolition',
                            'factor_bizarrebehav', 'factor_posformalthought', 'global_bizarrebehav', 'global_inappaffect',
                            'factor_delusions', 'global_delusions', 'factor_hallucinations', 'factor_inappaffect', 'global_posformalthought', 'global_hallucinations',
                            'vr2r_totalraw', 'ds_ftrs', 'ssp_totalraw', 'vr2dr_totalraw', 'vr1ir_totalraw', 'ds_totalraw', 'ds_strs', 'ds_btrs',
                            'voc_totalraw', 'lns_totalraw', 'mr_totalraw',
                            'cigs'
                            ]].copy()

pcd_UCLA_select['bis_sum'] = df_merged[['barratt1', 'barratt2', 'barratt3', 'barratt4', 'barratt5', 'barratt6', 'barratt7',
                                       'barratt8', 'barratt9', 'barratt10', 'barratt11', 'barratt13', 'barratt14', 'barratt15', 'barratt16',
                                       'barratt17', 'barratt18', 'barratt19', 'barratt20', 'barratt21', 'barratt22', 'barratt23', 'barratt24',
                                       'barratt25', 'barratt26', 'barratt27', 'barratt28', 'barratt29', 'barratt30']].sum(1)
pcd_UCLA_select['total_reccapacity_vcap'] = df_merged[['vcap_capacity3', 'vcap_capacity5', 'vcap_capacity7', 'vcap_capacity9']].mean(1)
pcd_UCLA_select['total_correct_vcap'] = df_merged[['vcap3_correct_sum', 'vcap5_correct_sum', 'vcap7_correct_sum', 'vcap9_correct_sum']].mean(1)
pcd_UCLA_select['participant_id'] = 'UCLA_' + pcd_UCLA_select['participant_id'].str.split('-', expand=True).iloc[:, -1]

###########################################################
########################################### ABIDE phenotype
###########################################################
abide_path = r'J:\Alex_data\ABIDE'
abide1_info = pd.read_excel(os.path.join(abide_path, 'ABIDE_I_Phenotypic.xlsx'))
abide1_pcd = abide1_info[['SUB_ID', 'FIQ', 'VIQ', 'PIQ', 'ADOS_TOTAL', 'ADOS_COMM', 'ADOS_SOCIAL', 'ADOS_STEREO_BEHAV', 'ADOS_GOTHAM_SOCAFFECT',
                         'ADOS_GOTHAM_RRB', 'ADOS_GOTHAM_TOTAL', 'ADOS_GOTHAM_SEVERITY', 'SRS_RAW_TOTAL', 'SRS_AWARENESS', 'SRS_COGNITION',
                         'SRS_COMMUNICATION', 'SRS_MOTIVATION', 'SRS_MANNERISMS', 'SCQ_TOTAL', 'AQ_TOTAL', 'VINELAND_RECEPTIVE_V_SCALED', 'VINELAND_EXPRESSIVE_V_SCALED',
                         'VINELAND_WRITTEN_V_SCALED', 'VINELAND_COMMUNICATION_STANDARD', 'VINELAND_PERSONAL_V_SCALED', 'VINELAND_DOMESTIC_V_SCALED',
                         'VINELAND_COMMUNITY_V_SCALED', 'VINELAND_DAILYLVNG_STANDARD', 'VINELAND_INTERPERSONAL_V_SCALED', 'VINELAND_PLAY_V_SCALED',
                         'VINELAND_COPING_V_SCALED', 'VINELAND_SOCIAL_STANDARD', 'VINELAND_SUM_SCORES', 'VINELAND_ABC_STANDARD',
                         'WISC_IV_VCI', 'WISC_IV_PRI',
                         'WISC_IV_WMI', 'WISC_IV_PSI', 'WISC_IV_SIM_SCALED', 'WISC_IV_VOCAB_SCALED', 'WISC_IV_INFO_SCALED', 'WISC_IV_BLK_DSN_SCALED', 'WISC_IV_PIC_CON_SCALED',
                         'WISC_IV_MATRIX_SCALED', 'WISC_IV_DIGIT_SPAN_SCALED', 'WISC_IV_LET_NUM_SCALED', 'WISC_IV_CODING_SCALED', 'WISC_IV_SYM_SCALED']]

abide2_info = pd.read_excel(os.path.join(abide_path, 'ABIDE_II_Phenotypic.xlsx'))
abide2_pcd = abide2_info[['SUB_ID', 'FIQ', 'VIQ', 'PIQ', 'ADOS_G_TOTAL', 'ADOS_G_COMM', 'ADOS_G_SOCIAL', 'ADOS_G_STEREO_BEHAV', 'ADOS_2_SOCAFFECT',
                         'ADOS_2_RRB', 'ADOS_2_TOTAL', 'ADOS_2_SEVERITY_TOTAL', 'SRS_TOTAL_RAW', 'SRS_AWARENESS_RAW', 'SRS_COGNITION_RAW',
                         'SRS_COMMUNICATION_RAW', 'SRS_MOTIVATION_RAW', 'SRS_MANNERISMS_RAW', 'SCQ_TOTAL', 'AQ_TOTAL', 'VINELAND_RECEPTIVE_V_SCALED', 'VINELAND_EXPRESSIVE_V_SCALED',
                         'VINELAND_WRITTEN_V_SCALED', 'VINELAND_COMMUNICATION_STANDARD', 'VINELAND_PERSONAL_V_SCALED', 'VINELAND_DOMESTIC_V_SCALED',
                         'VINELAND_COMMUNITY_V_SCALED', 'VINELAND_DAILYLIVING_STANDARD', 'VINELAND_INTERPERSONAL_V_SCALED', 'VINELAND_PLAY_V_SCALED',
                         'VINELAND_COPING_V_SCALED', 'VINELAND_SOCIAL_STANDARD', 'VINELAND_SUM_SCORES', 'VINELAND_ABC_Standard',
                         'MASC_TOTAL_T', 'MASC_T/R_T', 'MASC_S/A_T', 'MASC_PHYSICAL_TOTAL_T', 'MASC_PER_T', 'MASC_AC_T', 'MASC_HARM_TOTAL_T', 'MASC_H/R_T', 'MASC_PP_T',
                         'MASC_SOCIAL_TOTAL_T', 'MASC_SEP_T', 'MASC_ADI_T', 'MASC_INCONSISTENCY_SCORE', 'BRIEF_INHIBIT_T', 'BRIEF_SHIFT_T', 'BRIEF_EMOTIONAL_T', 'BRIEF_BRI_T',
                         'BRIEF_INITIATE_T', 'BRIEF_WORKING_T', 'BRIEF_PLAN_T', 'BRIEF_ORGANIZATION_T', 'BRIEF_MONITOR_T', 'BRIEF_MI_T', 'BRIEF_GEC_T', 'BRIEF_INCONSISTENCY_SCORE',
                         'BRIEF_NEGATIVITY_SCORE', 'CBCL_6-18_ACTIVITIES_T', 'CBCL_6-18_SOCIAL_T', 'CBCL_6-18_SCHOOL_T', 'CBCL_6-18_TOTAL_COMPETENCE_T', 'CBCL_6-18_ANXIOUS_T',
                         'CBCL_6-18_WITHDRAWN_T', 'CBCL_6-18_SOMATIC_COMPAINT_T', 'CBCL_6-18_SOCIAL_PROBLEM_T', 'CBCL_6-18_THOUGHT_T', 'CBCL_6-18_ATTENTION_T',
                         'CBCL_6-18_RULE_T', 'CBCL_6-18_AGGRESSIVE_T', 'CBCL_6-18_INTERNAL_T', 'CBCL_6-18_EXTERNAL_T', 'CBCL_6-18_TOTAL_PROBLEM_T', 'CBCL_6-18_AFFECTIVE_T',
                         'CBCL_6-18_ANXIETY_T', 'CBCL_6-18_SOMATIC_PROBLEM_T', 'CBCL_6-18_ATTENTION_DEFICIT_T', 'CBCL_6-18_OPPOSITIONAL_T', 'CBCL_6-18_CONDUCT_T', 'CBCL_6-18_SLUGGISH_T',
                         'CBCL_6-18_OBSESSIVE_T', 'CBCL_6-18_POST_TRAUMATIC_T']]

abide2_pcd = abide2_pcd.rename(columns={'ADOS_G_TOTAL': 'ADOS_TOTAL', 'ADOS_G_COMM': 'ADOS_COMM', 'ADOS_G_SOCIAL': 'ADOS_SOCIAL', 'ADOS_G_STEREO_BEHAV': 'ADOS_STEREO_BEHAV',
                           'ADOS_2_SOCAFFECT': 'ADOS_GOTHAM_SOCAFFECT', 'ADOS_2_RRB': 'ADOS_GOTHAM_RRB', 'ADOS_2_TOTAL': 'ADOS_GOTHAM_TOTAL',
                           'ADOS_2_SEVERITY_TOTAL': 'ADOS_GOTHAM_SEVERITY', 'SRS_TOTAL_RAW': 'SRS_RAW_TOTAL', 'SRS_AWARENESS_RAW': 'SRS_AWARENESS',
                           'SRS_COGNITION_RAW': 'SRS_COGNITION', 'SRS_COMMUNICATION_RAW': 'SRS_COMMUNICATION', 'SRS_MOTIVATION_RAW': 'SRS_MOTIVATION',
                           'SRS_MANNERISMS_RAW': 'SRS_MANNERISMS', 'VINELAND_DAILYLIVING_STANDARD': 'VINELAND_DAILYLVNG_STANDARD', 'VINELAND_ABC_Standard': 'VINELAND_ABC_STANDARD',
                           })

abide_pcd = pd.merge(abide1_pcd, abide2_pcd, on=['SUB_ID', 'FIQ', 'VIQ', 'PIQ', 'ADOS_TOTAL', 'ADOS_COMM', 'ADOS_SOCIAL', 'ADOS_STEREO_BEHAV', 'ADOS_GOTHAM_SOCAFFECT',
                         'ADOS_GOTHAM_RRB', 'ADOS_GOTHAM_TOTAL', 'ADOS_GOTHAM_SEVERITY', 'SRS_RAW_TOTAL', 'SRS_AWARENESS', 'SRS_COGNITION',
                         'SRS_COMMUNICATION', 'SRS_MOTIVATION', 'SRS_MANNERISMS', 'SCQ_TOTAL', 'AQ_TOTAL', 'VINELAND_RECEPTIVE_V_SCALED', 'VINELAND_EXPRESSIVE_V_SCALED',
                         'VINELAND_WRITTEN_V_SCALED', 'VINELAND_COMMUNICATION_STANDARD', 'VINELAND_PERSONAL_V_SCALED', 'VINELAND_DOMESTIC_V_SCALED',
                         'VINELAND_COMMUNITY_V_SCALED', 'VINELAND_DAILYLVNG_STANDARD', 'VINELAND_INTERPERSONAL_V_SCALED', 'VINELAND_PLAY_V_SCALED',
                         'VINELAND_COPING_V_SCALED', 'VINELAND_SOCIAL_STANDARD', 'VINELAND_SUM_SCORES', 'VINELAND_ABC_STANDARD'], how='outer')

abide_pcd[abide_pcd <= -99] = np.nan
abide_pcd = abide_pcd.rename(columns={'SUB_ID': 'participant_id'})
abide_pcd['participant_id'] = abide_pcd['participant_id'].astype(str)

###########################################################
########################################### SRPBS Japan phenotype
###########################################################
japan_pcd_info = sio.loadmat(r'J:\Alex_data\SPRBS_Japan\phenotypeTable_SRPBS_struct.mat')
tvar = japan_pcd_info['phenotypeTable_SRPBS_struct']
japan_PCD_data = []
pcd_keys_japan = ['subjectID', 'diagnosis', 'age', 'sex', 'EstimatedIQ', 'FIQ', 'BDI_II', 'AQ_total_', 'AQ_ss_', 'AQ_as_', 'AQ_atd_', 'AQ_Com_', 'AQ_imag_', 'PDIYes_noScore', 'PDIDistressScore',
            'PDIPreoccupationScore', 'PDIConvictionScore', 'DurationOfIllness_year_', 'Medication_Antipsychotics_ChlorpromazineEquivalentsMg_day_',
            'Medication_Anticholinergic_1_NoUse_2_Use_', 'Medication_Benzodiazepine_1_NoUse_2_Use_',
            'Medication_MoodStabiliser_1_NoUse_2_Use_', 'Medication_AntiDepression_1_NoUse_2_Use_', 'Smoking_CurrentUse_1_NoUse_2_Use_', 'BACSVerbalMemory',
            'BACSWorkingMemory_DigitSequencing_', 'BACSMotorSpeed_TokenMotorTask_', 'BACSVerbalFluency', 'BACSAttentionAndProcessingSpeed_Symbol_codingTask_',
            'BACSExecutiveFunction_TowerOfLondon_', 'PANSSTotal_Positive_Negative_General_', 'PANSSPositiveTotal', 'PANSSNegativeTotal', 'PANSSGeneralTotal', 'PANSSPosi_1Delusion',
            'PANSSPosi_2ConceptualDisorganization', 'PANSSPosi_3HallucinatoryBehavior', 'PANSSPosi_4Excitement', 'PANSSPosi_5Grandiosity', 'PANSSPosi_6Suspiciousness', 'PANSSPosi_7Hostility',
            'PANSSNega_1BluntedAffect', 'PANSSNega_2EmotionalWithdrawal', 'PANSSNega_3PoorRapport', 'PANSSNega_4PassiveApatheticSocialWithdrawal', 'PANSSNega_5DifficultyInAbstractThinking',
            'PANSSNega_6LuckOfSpontaneityAndFlowOfConversation', 'PANSSNega_7StereotypedThinking', 'PANSSGene_1SomaticConcern', 'PANSSGene_2Anxiety', 'PANSSGene_3GuiltFeeling',
            'PANSSGene_4Tension', 'PANSSGene_5MannerismsAndPosturing', 'PANSSGene_6Depression', 'PANSSGene_7MotorRetardation', 'PANSSGene_8Uncooperativeness', 'PANSSGene_9UnusualThoughtContent',
            'PANSSGene_10Disorientation', 'PANSSGene_11PoorAttention', 'PANSSGene_12LackOfJudgementAndInsight', 'PANSSGene_13DisturbanceOfVolition', 'PANSSGene_14PoorImpulseControl',
            'PANSSGene_15Preoccupation', 'PANSSGene_16ActiveSocialAvoidance', 'JART_25_FullScaleIntelligenceQuotient', 'JART_25_VerbalIntelligenceQuotient', 'JART_25_PerformanceIntelligenceQuotient',
            'Medication_AntiDepression_ImipramineEquivalentsMg_day_', 'CES_D', 'PANSS_positive', 'PANSS_negative', 'PANSS_generalPsychopathologyScales', 'StanfordSleepinessScale', 'PainDuration', 'VAS']

for idx in range(len(tvar)):
    sub_data = []
    for name in pcd_keys_japan:
        value = tvar[idx][0][name][0] if name == 'subjectID' else tvar[idx][0][name][0][0]
        sub_data.append(value)
    japan_PCD_data.append(sub_data)
japan_PCD_data = np.array(japan_PCD_data)
japan_PCD_df = pd.DataFrame(japan_PCD_data, columns=pcd_keys_japan)
japan_PCD_df = japan_PCD_df.rename(columns={'subjectID': 'participant_id'})
japan_PCD_df['FIQ'] = japan_PCD_df['FIQ'].astype(float)

###########################################################
########################################### Church phenotype
###########################################################
church_path = r'H:\PHD\learning\research\dataset\church_data'
church_info = pd.read_excel(os.path.join(church_path, 'kzhao_bmi_aging_20250807.xlsx'), sheet_name="kzhao_bmi_aging")
pcd_keys_church = ['NDPNum', 'BMI', 'Demog_Education', 'Demog_Income',
                'ACE_Emotional_Abuse', 'ACE_Physical_Abuse', 'ACE_Sexual_Abuse',
                'ACE_Substance_Abuse', 'ACE_Parental_DivorceSep', 'ACE_Household_Mental_Illness',
                  'ACE_Incarcerated_Household_Member', 'ACE_Parents_Treated_Violently', 'ACE_Score',
                  'ETI_General_Score', 'ETI_Physical_Score', 'ETI_Emotional_Score', 'ETI_Sexual_Score',
                  'ETI_Total_Score', 'SF12_PCS', 'SF12_MCS', 'SI_Social_Disconnectedness_Score',
                  'SI_Lack_Social_Support_Score', 'SI_Perceived_Loneliness_Score', 'MASQ_Language',
                  'MASQ_Visual_Perception', 'MASQ_Verbal_Memory', 'MASQ_Visual_Spatial', 'TRAILA_Time',
                  'TRAILB_Time', 'CAMSR_Score', 'STAI_TAnxiety_raw', 'STAI_TAnxiety', 'HAD_Anxiety',
                  'HAD_Depression', 'BISBAS_BAS_Drive', 'BISBAS_BAS_Fun_Seeking', 'BISBAS_BAS_Reward_Response',
                  'BISBAS_BIS', 'MHCSF_Hedonic', 'MHCSF_Eudaimonic_Social',
                  'MHCSF_Eudaimonic_Psych', 'MHCSF_Overall', 'FFM_Observe', 'FFM_Describe', 'FFM_ActAwareness',
                  'FFM_Nonjudge', 'FFM_Nonreact', 'FFM_Total_Score', 'CDRISC_Score', 'CDRISC_Persistence_Score',
                  'CDRISC_Adaptability_Score', 'CDRISC_Control_Meaning_Score', 'CDRISC_Meaning_Score',
                  'IPAQ_Work_Total_MET', 'IPAQ_Transport_Total_MET', 'IPAQ_Domestic_Total_MET',
                  'IPAQ_Total_Walking_MET', 'IPAQ_Total_Moderate_MET', 'IPAQ_Total_Vigorous_MET', 'IPAQ_Total_PA_MET',
                  'YFAS_SymptomCount', 'IPAQ_Sitting_Total', 'VSI_Score', 'GFCQT_Total', 'SSR_Arousal', 'SSR_Stress', 'SSR_Anxiety', 'SSR_Anger', 'SSR_Fatigue',
                  'SSR_Attention', 'STAI_SAnxiety_raw', 'STAI_SAnxiety', 'PANAS_PosAffect', 'PANAS_NegAffect',
                  'GFCQS_Total', 'PROMIS_Sleep_Score_R', 'PROMIS_Sleep_Score',
                  'MINI_Major_Depressive_c', 'MINI_Dysthymia_c', 'MINI_Major_Depressive_p',
                  'MINI_Dysthymia_p', 'MINI_Suicidality_c', 'MINI_Manic_p', 'MINI_Manic_c', 'MINI_Panic_c',
                  'MINI_Agoraphobia_l', 'MINI_Agoraphobia_c', 'MINI_Social_Phobia_c', 'MINI_Specific_Phobia_c',
                  'MINI_OCD_c', 'MINI_Alcohol_Dependence_l', 'MINI_Alcohol_Dependence_c', 'MINI_Substance_Dependence_l',
                  'MINI_Substance_Dependence_c', 'MINI_Anorexia_c', 'MINI_Bulimia_c', 'MINI_Generalized_Anxiety_c',
                  'MINI_Body_Dysmorphic_c', 'MINI_Premenstrual_Dysmorphic_c', 'IBS', 'IBS_c', 'GERD',
                  'GERD_c', 'Gastroparesis', 'Gastroparesis_c', 'FDyspepsia', 'FDyspepsia_c', 'UDyspepsia',
                  'UDyspepsia_c', 'CVS', 'CVS_c', 'VLVD', 'VLVD_c', 'UC', 'UC_c', 'Crohns', 'Crohns_c', 'ICIBPPS',
                  'ICIBPPS_c', 'Prostatitis', 'Prostatitis_c', 'Endomet', 'Endomet_c', 'TMJ_TMD', 'TMJ_TMD_c',
                  'CFS', 'CFS_c', 'FM', 'FM_c', 'Migraine', 'Migraine_c', 'Chest_Pain', 'Chest_Pain_c', 'Back_Neck',
                  'Back_Neck_c', 'Anxiety', 'Anxiety_c', 'Depression', 'Depression_c', 'Bipolar', 'Bipolar_c',
                  'PTSD', 'PTSD_c', 'Schizo', 'Schizo_c', 'Eating', 'Eating_c', 'Substance', 'Substance_c', 'IBD',
                  'IBD_c', 'Other_Pain', 'Other_Pain_c', 'Other_Condition', 'Other_Condition_c',
                  'OCD', 'OCD_c', 'UCPPS', 'UCPPS_c', 'Tension_HA', 'Tension_HA_c', 'Low_Back', 'Low_Back_c']
church_pcd = church_info[pcd_keys_church].copy()
church_pcd.iloc[:, 79:][church_pcd.iloc[:, 79:] >= 2] = np.nan
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

discovery_sex_nobmi = np.array([s[0][0][0] for s in bag_info['discovery_sex_nobmi'].T]).squeeze()
discovery_age_nobmi = bag_info['discovery_age_nobmi'].squeeze()
discovery_site_nobmi = bag_info['discovery_site_nobmi'].squeeze()
discovery_brainage_nobmi = bag_info['brainage_pred_all_discovery'].squeeze()
discovery_age_nobmi_resort = discovery_age_nobmi[discovery_index_nobmi]
discovery_site_nobmi_resort = discovery_site_nobmi[discovery_index_nobmi]
discovery_sex_nobmi_resort = discovery_sex_nobmi[discovery_index_nobmi]
discovery_sub_sess_cn_nobmi_resort = discovery_sub_sess_cn_nobmi[discovery_index_nobmi]
discovery_fc_cn_nobmi_resort = discovery_fc_cn_nobmi[discovery_index_nobmi]

bag_discovery = bag_info['bag_discovery'].squeeze()
bag_discovery_correct = bag_info['bag_discovery_correct'].squeeze()
bag_discovery_adult = bag_discovery[discovery_age_nobmi_resort >= 18]
bag_discovery_correct_adult = bag_discovery_correct[discovery_age_nobmi_resort >= 18]
discovery_sub_sess_cn_nobmi_adult = discovery_sub_sess_cn_nobmi_resort[discovery_age_nobmi_resort >= 18]
discovery_age_nobmi_resort_adult = discovery_age_nobmi_resort[discovery_age_nobmi_resort >= 18]

independent_fc_cn = bag_info['independent_fc_cn']
independent_sub_sess_cn = pd.Series(bag_info['independent_sub_sess_cn']).str.split(' ', expand=True).iloc[:, 0]
independent_sex = np.array([s[0][0][0] for s in bag_info['independent_sex'].T]).squeeze()
independent_age = bag_info['independent_age'].squeeze()
independent_site = bag_info['independent_site'].squeeze()
independent_BMI = bag_info['independent_BMI'].squeeze().astype(float)
bag_independent = bag_info['bag_independent'].squeeze()
bag_independent_correct = bag_info['bag_independent_correct'].squeeze()
ind_brainage_nobmi = bag_info['brainage_pred_all_ind'].squeeze().mean(0)

bag_independent_adult = bag_independent[independent_age >= 18]
bag_independent_correct_adult = bag_independent_correct[independent_age >= 18]
independent_sub_sess_cn_adult = independent_sub_sess_cn[independent_age >= 18]
independent_age_adult = independent_age[independent_age >= 18]

church_fc_cn = bag_info['fc_church_ahlf_10']
church_sub_cn = np.array([sub[0] for sub in bag_info['church_sub_10'].squeeze()])
church_sex = np.array([s[0][0][0] for s in bag_info['church_sex_10'].T]).squeeze()
church_age = bag_info['church_age_10'].squeeze()
bag_church = bag_info['bag_church'].squeeze()
bmi_church = bag_info['BMI_church_10'].squeeze()
bag_church_correct = bag_info['bag_church_correct'].squeeze()
brain_age_church = bag_info['brainage_pred_all_church'].squeeze().mean(0)

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

# Clinical variables of interest in each cohort
clinical_discovery = ['FIQ', 'VIQ', 'PIQ', 'ADOS_TOTAL', 'ADOS_SOCIAL', 'ADOS_STEREO_BEHAV',
                      'BDI_II', 'AQ_total_', 'AQ_ss_', 'AQ_as_', 'AQ_atd_', 'AQ_Com_', 'AQ_imag_',
                      'BACSVerbalMemory', 'BACSWorkingMemory_DigitSequencing_', 'BACSMotorSpeed_TokenMotorTask_',
                      'BACSVerbalFluency', 'BACSAttentionAndProcessingSpeed_Symbol_codingTask_',
                      'BACSExecutiveFunction_TowerOfLondon_']
clinical_independent = ['chaphypo_total', 'chapinf_total', 'chapper_total', 'chapphy_total', 'chapsoc_total',
                        'cvlt_totcor', 'cvltz_10', 'cvltz_12', 'cvltz_16', 'cvltz_17', 'cvltz_18', 'cvltz_19']
clinical_church = ['SI_Social_Disconnectedness_Score', 'SI_Lack_Social_Support_Score', 'SI_Perceived_Loneliness_Score',
                   'TRAILA_Time', 'TRAILB_Time', 'HAD_Anxiety', 'HAD_Depression', 'STAI_SAnxiety', 'STAI_TAnxiety']


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
