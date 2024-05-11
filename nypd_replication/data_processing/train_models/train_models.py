import pandas as pd
import numpy as np
import os
from datetime import datetime
from ml_utils import *
from ml_utils import _get_pseudo_id
import argparse


def get_substantiated_complaint_features(df):

    total_count_cols = (
        df.filter(like="past").filter(like="complaints.disposition_substantiated").columns.tolist()
    )
    substantiated_allegation_cols = (
        df.filter(like="past")
        .filter(like="allegations")
        .filter(like=".substantiated")
        .columns.tolist()
    )

    return total_count_cols + substantiated_allegation_cols


def get_all_complaint_cols(df):

    total_count_cols = df.filter(like="past").filter(like="complaints").columns.tolist()
    allegation_cols = df.filter(like="past").filter(like="allegations").columns.tolist()

    return total_count_cols + allegation_cols


def train_model(active_officer_df, target, feature_list=None, mc_iters=1, n_jobs=1):

    if feature_list is None:
        feature_list = active_officer_df.filter(like="past_").columns.tolist()

    random_state = 0

    all_predictions = []
    for i in np.arange(mc_iters):
        print(f"iteration {i}")
        # temp_est = clone(best_estimator)
        temp_data = active_officer_df.copy()

        # This is a hack to get around the fact that GroupKFold is deterministic function of the `group` labels
        temp_data["pseudo_ID"] = _get_pseudo_id(temp_data, random_state + i)

        gkf = GroupKFold(n_splits=3)

        j = 0
        for train_ix, test_ix in gkf.split(temp_data, groups=temp_data["pseudo_ID"]):
            print(f"group {j}")
            temp_train = temp_data.iloc[train_ix].reset_index()

            temp_test = temp_data.iloc[test_ix].reset_index()

            temp_est = get_model_search_clf("HistGBM", feature_list, [], n_jobs=n_jobs)
            temp_est.fit(temp_train, (temp_train[target] >= 1) * 1.0, groups=temp_train["tax_id"])

            if hasattr(temp_est, "predict_proba"):
                temp_test["phat"] = temp_est.predict_proba(temp_test)[:, 1]
            else:
                temp_test["phat"] = temp_est.predict(temp_test)

            temp_test["cv_group"] = j
            temp_test["iteration"] = i
            j += 1

            record_cols = ["tax_id", "observation_date", "phat", "iteration", "cv_group"]
            all_predictions.append(temp_test[record_cols])

    all_preds = pd.concat(all_predictions)

    all_preds = all_preds.groupby(["tax_id", "observation_date"])["phat"].mean().reset_index()

    return all_preds


def limit_observations_to_active_officers(_main_table):

    """
        Only keeps observations where the observation date is between career start and career end date. 
        Implicilty drops any rows where career start date is null
    """
    career_dates = pd.read_parquet("../create_career_start_end_dates/output/career_dates.parquet")

    main_table = pd.merge(
        _main_table, career_dates, left_on="tax_id", right_on="tax_id", how="left"
    )

    main_table["officer_presumed_active"] = main_table["observation_date"].between(
        main_table["career_start_date"], main_table["career_end_date"]
    )
    main_table["year"] = main_table["observation_date"].dt.year

    active_officer_df = main_table[main_table["officer_presumed_active"] == True].reset_index(
        drop=True
    )
    return active_officer_df


def train_model_and_write_predictions(
    df, target, target_short_name, output_dir="output", mc_iters=1, n_jobs=1
):

    print("training on", target)
    all_feature_predictions = train_model(df, target=target, mc_iters=mc_iters, n_jobs=n_jobs)

    only_substantiated_complaint_predictions = train_model(
        df,
        target=target,
        feature_list=get_substantiated_complaint_features(df),
        mc_iters=mc_iters,
        n_jobs=n_jobs,
    )
    only_substantiated_complaint_predictions.rename(
        columns={"phat": "phat__only_sus_complaints"}, inplace=True
    )

    only_complaint_predictions = train_model(
        df, target=target, feature_list=get_all_complaint_cols(df), mc_iters=mc_iters, n_jobs=n_jobs
    )
    only_complaint_predictions.rename(columns={"phat": "phat__only_complaints"}, inplace=True)

    df_w_preds = pd.merge(
        df,
        all_feature_predictions,
        how="inner",
        left_on=["tax_id", "observation_date"],
        right_on=["tax_id", "observation_date"],
    )
    df_w_preds = pd.merge(
        df_w_preds,
        only_substantiated_complaint_predictions,
        how="inner",
        left_on=["tax_id", "observation_date"],
        right_on=["tax_id", "observation_date"],
    )
    df_w_preds = pd.merge(
        df_w_preds,
        only_complaint_predictions,
        how="inner",
        left_on=["tax_id", "observation_date"],
        right_on=["tax_id", "observation_date"],
    )

    output_path = f"{output_dir}/{target_short_name}"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df_w_preds.to_parquet(f"{output_path}/observations_with_predictions.parquet")


if __name__ == "__main__":

    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add the argument with a default value
    parser.add_argument(
        "--mc_iters", type=int, default=5, help="number of Monte Carlo iterations (default: 5)"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=5, help="number of concurrently running workers (default: 5)"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Get the argument values
    mc_iters = args.mc_iters
    n_jobs = args.n_jobs
    print(f"the specified number of jobs is {n_jobs}")
    print(f"the specified number of iterations is {mc_iters}")

    main_table = pd.read_parquet(
        "../create_observations_main_table/output/observation_table.parquet"
    )

    features = pd.read_parquet("../create_features_and_outcomes/output/features.parquet")
    features.set_index(["tax_id", "observation_date"], inplace=True)
    outcomes = pd.read_parquet("../create_features_and_outcomes/output/outcomes.parquet")
    outcomes.set_index(["tax_id", "observation_date"], inplace=True)

    main_table = pd.merge(
        main_table, features, how="left", left_on=["tax_id", "observation_date"], right_index=True
    )

    main_table = pd.merge(
        main_table, outcomes, how="left", left_on=["tax_id", "observation_date"], right_index=True
    )

    active_officer_df = limit_observations_to_active_officers(main_table)

    PREDICTION_START = datetime(2014, 12, 31)
    PREDICTION_END = datetime(2019, 1, 2)
    active_officer_df = active_officer_df[
        active_officer_df.observation_date.between(PREDICTION_START, PREDICTION_END)
    ].reset_index()

    train_model_and_write_predictions(
        active_officer_df,
        "future_two_years.complaints.disposition_substantiated",
        "sustained_complaints",
        mc_iters=mc_iters,
        n_jobs=n_jobs,
    )

    train_model_and_write_predictions(
        active_officer_df,
        "future_two_years.lawsuits.high_payout_suit",
        "expensive_lawsuit",
        mc_iters=mc_iters,
        n_jobs=n_jobs,
    )
