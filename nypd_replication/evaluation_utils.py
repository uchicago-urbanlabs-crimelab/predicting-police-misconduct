import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

RBC_SAMPLE_RANDOM_STATE = 42

def calc_auc(preds_w_outcomes, pred_col, outcome):
    '''
    Calculates AUC

    Parameters:
        preds_w_outcomes: (df) with risk scores and outcomes
        pred_col: (str) name of column with risk scores produced by model
                  of interest
        outcome: (str) name of outcome
    Returns:
        auc: (float)
    '''
    df = preds_w_outcomes.copy()
    df["indicator_outcome"] = df[outcome].clip(upper=1)
    auc = roc_auc_score(df["indicator_outcome"], df[pred_col])
    return auc

def calc_base_rate_across_years(y, outcome, indicator=True):
    df = y.copy()
    if indicator:
            df["outcome"] = df[outcome] >= 1
    else:
        df["outcome"] = df[outcome].copy()
    base_rate_across_years = df.groupby("pred_year")["outcome"].mean().to_frame()
    base_rate_across_years.rename(columns={"outcome": "base_rate"}, inplace=True)
    
    return base_rate_across_years


def calc_base_rate(y, outcome, indicator=True):
    '''
    Calculates the indicator base rate of a given outcome

    Parameters:
        y: (df) with outcomes
        outcome: (str) name of outcome
        indicator: (bool) True if we want to calculate the indicator base rate
    Returns:
        base_rate: (float) proportion of officers with outcome in outcome period
    '''
    if indicator:
        base_rates = y.groupby("pred_year").apply(lambda x: x[outcome].clip(upper=1).mean()).to_frame()
    else:
        base_rates = y.groupby("pred_year").apply(lambda x: x[outcome].mean()).to_frame()
    base_rates.rename(columns={0: "Base Rate"}, inplace=True)
    base_rate = base_rates["Base Rate"].mean()
    
    return base_rate


def get_flags(preds, pred_col, threshold):
    '''
    Determines which officers are flagged

    Parameters:
        preds: (df) with risk scores produced by models
        pred_col: (str) name of column with risk scores produced by model
                  of interest
        threshold: (float) minimum percentile of risk score at which 
                    officers will be flagged. Range [0, 1]
    Returns:
        preds_w_flags: (df) same as preds but with an extra columns called
                       "flagged" that is True if officer is flagged
    '''
    preds_w_flags = preds.copy()
    preds_w_flags["flagged"] = (preds_w_flags.groupby('pred_year')[pred_col].rank(pct=True,ascending=True)) >= threshold
    return preds_w_flags


def average_stat_across_periods(stat_df, stat_col_name):
    '''
    Calculates the average of some eval statistic accross all years

    Parameters:
        stat_df: (df) with the statist of interest for each year/period
        stat_col_name: (str) name of stat column in stat_df
    Returns:
        overall_stat: (float)
    '''
    overall_stat = stat_df[stat_col_name].mean()
    return overall_stat


def calc_precision_across_years(preds_w_outcomes, pred_col, outcome, threshold, indicator=True):
    '''
    Calculates the precision of model for each period/year
    
    Parameters:
        preds_w_outcomes: (df) with risk scores and outcomes
        pred_col: (str) name of column with risk scores produced by model
                  of interest
        outcome: (str) name of outcome
        threshold: (float) minimum percentile of risk score at which 
                    officers will be flagged. Range [0, 1]
        indicator: (bool) True if we want to calculate the indicator precision
    Returns:
        (df) where each row contains the period start date (as index) and the
        precision
    '''
    preds_w_flags = get_flags(preds_w_outcomes, pred_col, threshold)
    if indicator:
        preds_w_flags["outcome"] = preds_w_flags[outcome] >= 1
    else:
        preds_w_flags["outcome"] = preds_w_flags[outcome].copy()
    flagged_df = preds_w_flags[preds_w_flags["flagged"] == True]
    precision_across_years = flagged_df.groupby("pred_year")["outcome"].mean().to_frame()
    precision_across_years.rename(columns={"outcome": "precision"}, inplace=True)
    
    return precision_across_years


def calc_precision(preds_w_outcomes, pred_col, outcome, threshold, indicator=True):
    '''
    Calculates the precision of model
    
    Parameters:
        preds_w_outcomes: (df) with risk scores and outcomes
        pred_col: (str) name of column with risk scores produced by model
                  of interest
        outcome: (str) name of outcome
        threshold: (float) minimum percentile of risk score at which 
                    officers will be flagged. Range [0, 1]
        indicator: (bool) True if we want to calculate the indicator precision
    Returns:
        precision: (float)
    '''
    precision_across_years = calc_precision_across_years(preds_w_outcomes, pred_col, outcome, 
                                                         threshold, indicator=indicator)
    precision = average_stat_across_periods(precision_across_years, "precision")
    return precision


def calc_recall_across_years(preds_w_outcomes, pred_col, outcome, threshold, indicator=True):
    '''
    Calculates the recall of model for each period/year
    
    Parameters:
        preds_w_outcomes: (df) with risk scores and outcomes
        pred_col: (str) name of column with risk scores produced by model
                  of interest
        outcome: (str) name of outcome
        threshold: (float) minimum percentile of risk score at which 
                    officers will be flagged. Range [0, 1]
        indicator: (bool) True if we want to calculate the indicator recall 
    Returns:
        recall_across_years: (df) where each row has the period start date as
                             the index and the recall as the column value
    '''
    preds_w_flags = get_flags(preds_w_outcomes, pred_col, threshold)
    if indicator:
        preds_w_flags["outcome"] = preds_w_flags[outcome] >= 1
    else:
        preds_w_flags["outcome"] = preds_w_flags[outcome].copy()
    recall_across_years = preds_w_flags.groupby('pred_year').apply(lambda x: x[x["flagged"] == True]["outcome"].sum() / x["outcome"].sum()).to_frame()
    recall_across_years.rename(columns={0: "recall"}, inplace=True)
    
    return recall_across_years


def calc_recall(preds_w_outcomes, pred_col, outcome, threshold, indicator=True):
    '''
    Calculates the recall of the model

    Parameters:
        preds_w_outcomes: (df) with risk scores and outcomes
        pred_col: (str) name of column with risk scores produced by model
                  of interest
        outcome: (str) name of outcome
        threshold: (float) minimum percentile of risk score at which 
                    officers will be flagged. Range [0, 1]
        indicator: (bool) True if we want to calculate the indicator recall 
    Returns:
        recall: (float)
    '''
    recall_across_years = calc_recall_across_years(preds_w_outcomes, pred_col, outcome, 
                                                   threshold, indicator=indicator)
    recall = average_stat_across_periods(recall_across_years, "recall")
    return recall


def calc_num_true_positives_across_years(preds_w_outcomes, pred_col, outcome, threshold):
    '''
    Calculates the number of true positives for a model for each period/year

    Parameters:
        preds_w_outcomes: (df) with risk scores and outcomes
        pred_col: (str) name of column with risk scores produced by model
                  of interest
        outcome: (str) name of outcome
        threshold: (float) minimum percentile of risk score at which 
                    officers will be flagged. Range [0, 1]
    Returns:
        (df) where each row contains the period start date (as index) and the
        number of correctly flagged officers

    '''
    preds_w_flags = get_flags(preds_w_outcomes, pred_col, threshold)
    preds_w_flags["indicator_outcome"] = preds_w_flags[outcome] >= 1
    true_positives_across_years = preds_w_flags.groupby('pred_year').apply(lambda x: x[x["flagged"] == True]["indicator_outcome"].sum()).to_frame()
    true_positives_across_years.rename(columns={0: "true_positives"}, inplace=True)
    
    return true_positives_across_years


def calc_num_true_postives(preds_w_outcomes, pred_col, outcome, threshold):
    '''
    Calculates the number of true positives for a model

    Parameters:
        preds_w_outcomes: (df) with risk scores and outcomes
        pred_col: (str) name of column with risk scores produced by model
                  of interest
        outcome: (str) name of outcome
        threshold: (float) minimum percentile of risk score at which 
                    officers will be flagged. Range [0, 1]
    Returns: 
        true_positives: (int)
    '''
    true_positives_across_years = calc_num_true_positives_across_years(preds_w_outcomes, pred_col, outcome, threshold)
    true_positives = average_stat_across_periods(true_positives_across_years, "true_positives")
    return true_positives


def get_flags_rbc(preds, rbc_col, threshold, random_state):
    '''
    Determines which officers are flagged for Rank by Complaints.
    We want to flag exacty (1 - threshold)% of officers but this us more
    challenging with with rbc because of ties. Therefore, here we randomly
    break ties to determine who is flagged.

    Parameters:
        preds: (df) with rank by complaints column
        rbc_col: (str) name of rank by complaint column
        threshold: (float) minimum percentile of risk score at which 
                    officers will be flagged. Range [0, 1]
        random_state: (int) used as seed to shuffle preds which in turn
                      breaks ties
    Returns:
        preds_w_flags: (df) same as preds but with an extra columns called
                       "flagged" that is True if officer is flagged
    '''
    preds_w_flags = preds.copy()
    preds_w_flags = preds_w_flags.sample(frac=1, random_state=random_state)
    preds_w_flags["flagged"] = (preds_w_flags.groupby('pred_year')[rbc_col].rank(pct=True,ascending=True,method="first")) >= threshold
    return preds_w_flags


def calc_precision_across_years_rbc(preds_w_outcomes, rbc_col, outcome, threshold, random_state, indicator=True):
    '''
    Calculates the precision for rank by complaints for each period/year

    Parameters:
        preds_w_outcomes: (df) with the rank by complaints and outcome columns
        rbc_col: (str) name of rank by complaint column
        outcome: (str) name of outcome
        threshold: (float) minimum percentile of risk score at which 
                    officers will be flagged. Range [0, 1]
        random_start: (int) random seed for breaking ties when flagging
        indicator: (bool) True if we want to calculate the indicator precision
    Returns:
        (df) where each row contains the period start date (as index) and the
        precision
    '''
    preds_w_flags = get_flags_rbc(preds_w_outcomes, rbc_col, threshold, random_state)
    if indicator:
        preds_w_flags["outcome"] = preds_w_flags[outcome] >= 1
    else:
        preds_w_flags["outcome"] = preds_w_flags[outcome].copy()
    flagged_df = preds_w_flags[preds_w_flags["flagged"] == True]
    precision_across_years = flagged_df.groupby("pred_year")["outcome"].mean().to_frame()
    precision_across_years.rename(columns={"outcome": "precision"}, inplace=True)
    
    return precision_across_years


def calc_recall_across_years_rbc(preds_w_outcomes, rbc_col, outcome, threshold, random_state, indicator=True):
    '''
    Calculates the recall for rank by complaints for each period/year

    Parameters:
        preds_w_outcomes: (df) with the rank by complaints and outcome columns
        rbc_col: (str) name of rank by complaint column
        outcome: (str) name of outcome
        threshold: (float) minimum percentile of risk score at which 
                    officers will be flagged. Range [0, 1]
        random_start: (int) random seed for breaking ties when flagging
        indicator: (bool) True if we want to calculate the indicator recall
    Returns:
        (df) where each row contains the period start date (as index) and the
        recall
    '''
    preds_w_flags = get_flags_rbc(preds_w_outcomes, rbc_col, threshold, random_state)
    if indicator:
        preds_w_flags["outcome"] = preds_w_flags[outcome] >= 1
    else:
        preds_w_flags["outcome"] = preds_w_flags[outcome].copy()
    recall_across_years = preds_w_flags.groupby('pred_year').apply(lambda x: x[x["flagged"] == True]["outcome"].sum() / x["outcome"].sum()).to_frame()
    recall_across_years.rename(columns={0: "recall"}, inplace=True)
    
    return recall_across_years


def calc_num_true_positives_across_years_rbc(preds_w_outcomes, rbc_col, outcome, threshold, random_state):
    '''
    Calculates the number of true positives for rank by complaints for
    each period/year

    Parameters:
        preds_w_outcomes: (df) with the rank by complaints and outcome columns
        rbc_col: (str) name of rank by complaint column
        outcome: (str) name of outcome
        threshold: (float) minimum percentile of risk score at which 
                    officers will be flagged. Range [0, 1]
        random_start: (int) random seed for breaking ties when flagging
    Returns:
        (df) where each row contains the period start date (as index) and the
        number of correctly flagged officers
    '''
    preds_w_flags = get_flags_rbc(preds_w_outcomes, rbc_col, threshold, random_state)
    preds_w_flags["indicator_outcome"] = preds_w_flags[outcome] >= 1
    true_positives_across_years = preds_w_flags.groupby('pred_year').apply(lambda x: x[x["flagged"] == True]["indicator_outcome"].sum()).to_frame()
    true_positives_across_years.rename(columns={0: "true_positives"}, inplace=True)
    
    return true_positives_across_years


def calc_precision_rbc(preds_w_outcomes, rbc_col, outcome, threshold,
                       n_iterations=10, indicator=True):
    '''
    Calculates precision for rank by complaints.
    Because there are ties between officers and we want to flag exactly
    (1-threshold)% of officers per year, we use a scheme where we randomly
    break ties between officers over many iterations and then average the eval 
    statistic across the iterations.

    Parameters:
        preds_w_outcomes: (df) with the rank by complaints and outcome columns
        rbc_col: (str) name of rank by complaint column
        outcome: (str) name of outcome
        threshold: (float) minimum percentile of risk score at which 
                    officers will be flagged. Range [0, 1]
        n_iterations: (int) number of iterations of calculating statistcs where
                      we are randomly breaking ties between officers
        indicator: (bool) True if we want to calculate the indicator precision
    Returns:
        precision: (float)
    '''
    random_state = RBC_SAMPLE_RANDOM_STATE
    precisions = []
    for i in range(n_iterations):
        current_random_state = random_state + i
        
        precision_across_years = calc_precision_across_years_rbc(preds_w_outcomes, rbc_col, outcome, 
                                                           threshold, current_random_state, indicator=indicator)
        iteration_precision = average_stat_across_periods(precision_across_years, "precision")
        precisions.append(iteration_precision)
        
    precision = np.mean(precisions)
    
    return precision


def calc_recall_and_num_true_positives_rbc(preds_w_outcomes, rbc_col, outcome, threshold,
                                           n_iterations=10, indicator=True):
    '''
    Calculates recall and number of true positives for rank by complaints.
    Because there are ties between officers and we want to flag exactly
    (1-threshold)% of officers per year, we use a scheme where we randomly
    break ties between officers over many iterations and then average the eval 
    statistic across the iterations.

    Parameters:
        preds_w_outcomes: (df) with the rank by complaints and outcome columns
        rbc_col: (str) name of rank by complaint column
        outcome: (str) name of outcome
        threshold: (float) minimum percentile of risk score at which 
                    officers will be flagged. Range [0, 1]
        n_iterations: (int) number of iterations of calculating statistcs where
                      we are randomly breaking ties between officers
        indicator: (bool) True if we want to calculate the indicator recall
    Returns:
        recall: (float)
        true_positive: (int)
    '''
    random_state = RBC_SAMPLE_RANDOM_STATE
    recalls = []
    true_positives = []
    for i in range(n_iterations):
        current_random_state = random_state + i
        
        recall_across_years = calc_recall_across_years_rbc(preds_w_outcomes, rbc_col, outcome, 
                                                           threshold, current_random_state, indicator=indicator)
        iteration_recall = average_stat_across_periods(recall_across_years, "recall")
        recalls.append(iteration_recall)
        
        true_positives_across_years = calc_num_true_positives_across_years_rbc(preds_w_outcomes, rbc_col, outcome, threshold, current_random_state)
        iteration_true_positives = average_stat_across_periods(true_positives_across_years, "true_positives")
        true_positives.append(iteration_true_positives)
        
    recall = np.mean(recalls)
    true_positive = np.mean(true_positives)
    
    return recall, true_positive
