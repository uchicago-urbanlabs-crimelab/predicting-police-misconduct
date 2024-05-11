import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
import os

dispo_map = {"substantiated": 4, "not_substantiated": 3, "truncated": 2, "pending": 1}

dispo_map__reverse = {dispo_map[k]: k for k in dispo_map}


def get_time_period_name(x):

    temp_dict = {1: "past_year", 2: "past_two_years", 5: "past_five_years"}
    return temp_dict[x]


def limit_allegations_to_time_period(_allegations, start_date, end_date, omniscient=False):

    allegations = _allegations.copy()

    # First limit to events that took place during the time period.
    allegations = allegations[allegations["incident_date"].between(start_date, end_date)]

    if omniscient == False:

        # Now limit to allegations that were created before the end of the time period
        allegations = allegations[allegations["received_date"] <= end_date]

        # If the complaint hasn't reached disposition by the end of the time period, change the status to pending.
        allegations["ccrb_disposition__collapsed"] = np.where(
            allegations["close_date"] <= end_date,
            allegations["ccrb_disposition__collapsed"],
            "pending",
        )
        allegations["ccrb_disposition__collapsed"] = np.where(
            allegations["close_date"] <= end_date,
            allegations["ccrb_disposition__collapsed"],
            "pending",
        )

    return allegations


def agg_allegation_types_complaint_level(temp_allegations):

    fado_types = [
        "FADO_abuse_of_authority",
        "FADO_discourtesy",
        "FADO_force",
        "FADO_offensive_language",
        "FADO_untruthful_statement",
    ]

    complaint_level_data = temp_allegations.groupby(["complaint_id", "tax_id"])[fado_types].max()

    return complaint_level_data.reset_index()


def agg_dispositions_complaint_level(temp_allegations):

    complaint_level_dispo = (
        temp_allegations.groupby(["complaint_id", "tax_id"])["dispo_code"].max().reset_index()
    )
    complaint_level_dispo["disposition"] = complaint_level_dispo["dispo_code"].map(
        dispo_map__reverse
    )
    complaint_level_dispo = pd.get_dummies(
        complaint_level_dispo, columns=["disposition"], prefix="disposition"
    )

    complaint_level_counts = (
        complaint_level_dispo.drop(columns=["complaint_id", "dispo_code"]).groupby("tax_id").sum()
    )

    complaint_level_counts["total"] = complaint_level_dispo["tax_id"].value_counts()

    return complaint_level_counts


def aggregate_officer_allegations(temp_allegations):

    temp_complaints = agg_allegation_types_complaint_level(temp_allegations)

    complaint_counts = temp_complaints.drop(columns=["complaint_id"]).groupby("tax_id").sum()
    # complaint_counts['total'] = temp_complaints['tax_id'].value_counts()

    return complaint_counts


def summarize_complaints_and_allegations(temp_allegations):

    all_allegation_counts = aggregate_officer_allegations(temp_allegations)
    all_allegation_counts = all_allegation_counts = all_allegation_counts.add_prefix(
        "all_allegations."
    )

    substantiated_allegation_counts = aggregate_officer_allegations(
        temp_allegations[temp_allegations["ccrb_disposition__collapsed"] == "substantiated"]
    )
    substantiated_allegation_counts = substantiated_allegation_counts.add_prefix(
        "substantiated_allegations."
    )

    truncated_allegation_counts = aggregate_officer_allegations(
        temp_allegations[temp_allegations["ccrb_disposition__collapsed"] == "truncated"]
    )
    truncated_allegation_counts = truncated_allegation_counts.add_prefix("truncated_allegations.")

    not_substantiated_allegation_counts = aggregate_officer_allegations(
        temp_allegations[temp_allegations["ccrb_disposition__collapsed"] == "not_substantiated"]
    )
    not_substantiated_allegation_counts = not_substantiated_allegation_counts.add_prefix(
        "notsubstantiated_allegations."
    )

    pending_allegation_counts = aggregate_officer_allegations(
        temp_allegations[temp_allegations["ccrb_disposition__collapsed"] == "pending"]
    )
    pending_allegation_counts = pending_allegation_counts.add_prefix("pending.")

    officer_complaint_counts = agg_dispositions_complaint_level(temp_allegations)
    officer_complaint_counts = officer_complaint_counts.add_prefix("complaints.",)

    officer_summary = pd.concat(
        [
            officer_complaint_counts,
            all_allegation_counts,
            substantiated_allegation_counts,
            truncated_allegation_counts,
            not_substantiated_allegation_counts,
            pending_allegation_counts,
        ],
        axis=1,
    )

    return officer_summary


def limit_lawsuits_to_time_period(_lawsuits, start_date, end_date, omniscient=False):

    lawsuits = _lawsuits.copy()

    lawsuits = lawsuits[lawsuits["lit_start"].between(start_date, end_date)]

    lawsuits["pending"] = lawsuits["disp_date"].isna()

    if omniscient == False:

        # If the lawsuits hasn't reached disposition by end of time period, change the payout to 0 (i.e. essentially consider it pending)
        lawsuits["officer_payout"] = np.where(
            lawsuits["disp_date"] > end_date, 0, lawsuits["officer_payout"]
        )
        lawsuits["pending"] = np.where(lawsuits["disp_date"] > end_date, True, lawsuits["pending"])
        lawsuits["high_payout_suit"] = np.where(
            lawsuits["disp_date"] > end_date, False, lawsuits["high_payout_suit"]
        )

    lawsuits["closed"] = lawsuits["pending"] == False

    return lawsuits


def aggregate_lawsuits(_temp_lawsuits):

    temp_lawsuits = _temp_lawsuits.copy()

    agg_cols = [
        "officer_payout",
        "use_of_force_allegation",
        "assault_battery_allegation",
        "malicious_prosecution_allegation",
        "false_arrest_imprison_allegation",
        "pending",
        "closed",
        "high_payout_suit",
    ]

    aggregated_lawsuits = temp_lawsuits.groupby("tax_id")[agg_cols].sum()
    aggregated_lawsuits["total"] = temp_lawsuits["tax_id"].value_counts()

    return aggregated_lawsuits


def summarize_lawsuits(temp_lawsuits):

    agg_suits = aggregate_lawsuits(temp_lawsuits)
    agg_suits = agg_suits.add_prefix("lawsuits.")
    return agg_suits


def get_outcome_time_period_name(x):

    if x == 1:
        return "future_one_year"
    if x == 2:
        return "future_two_years"


def create_outcomes(
    observation_table,
    allegations,
    lawsuits,
    outcome_period_list=[1, 2],
    use_lawsuit_offset=False,
    lawsuit_offset_months=6,
):

    observation_date_list = observation_table["observation_date"].unique()

    all_outcome_list = []

    lawsuit_offset = DateOffset(months=lawsuit_offset_months)

    for observation_date in observation_date_list:

        outcome_df_list = []
        for y in outcome_period_list:
            end_date = pd.to_datetime(observation_date) + DateOffset(years=y)
            start_date = observation_date
            print(start_date, end_date)

            temp_allegations = limit_allegations_to_time_period(
                allegations, start_date, end_date, omniscient=True
            )
            temp_allegations["dispo_code"] = temp_allegations["ccrb_disposition__collapsed"].map(
                dispo_map
            )
            temp_complaint_summary = summarize_complaints_and_allegations(temp_allegations)

            if use_lawsuit_offset == True:
                temp_lawsuits = limit_lawsuits_to_time_period(
                    lawsuits,
                    pd.to_datetime(start_date) + lawsuit_offset,
                    end_date + lawsuit_offset,
                    omniscient=True,
                )
            else:
                temp_lawsuits = limit_lawsuits_to_time_period(
                    lawsuits, start_date, end_date, omniscient=True
                )

            temp_lawsuit_summary = summarize_lawsuits(temp_lawsuits)

            time_period_name = get_outcome_time_period_name(y)
            temp_complaint_summary = temp_complaint_summary.add_prefix(f"{time_period_name}.")
            temp_lawsuit_summary = temp_lawsuit_summary.add_prefix(f"{time_period_name}.")

            outcome_df_list.append(temp_complaint_summary)
            outcome_df_list.append(temp_lawsuit_summary)

        all_period_outcomes = pd.concat(outcome_df_list, axis=1, join="outer")
        all_period_outcomes["observation_date"] = observation_date

        all_period_outcomes.fillna(0, inplace=True)

        all_outcome_list.append(all_period_outcomes)

    all_outcomes = pd.concat(all_outcome_list)
    all_outcomes.reset_index(inplace=True)
    all_outcomes.set_index(["tax_id", "observation_date"], inplace=True)

    keep_col_regex_list = [
        "complaints.total",
        "complaints.disposition_substantiated",
        "lawsuits.total",
        "lawsuits.officer_payout",
        "lawsuits.high_payout_suit",
    ]

    keep_cols = []
    for col_regex in keep_col_regex_list:
        temp_cols = all_outcomes.filter(like=col_regex).columns.tolist()
        keep_cols.extend(temp_cols)

    keep_cols = list(set(keep_cols))
    observation_table_w_outcomes = pd.merge(
        observation_table,
        all_outcomes[keep_cols],
        how="left",
        left_on=["tax_id", "observation_date"],
        right_index=True,
    )

    observation_table_w_outcomes.fillna(0, inplace=True)

    return observation_table_w_outcomes


def create_features(observation_table, allegations, lawsuits, past_year_list=[1, 2, 5]):

    observation_date_list = observation_table["observation_date"].unique()
    past_year_list = [1, 2, 5]
    all_observation_list = []

    for observation_date in observation_date_list:

        feature_df_list = []
        for y in past_year_list:
            start_date = pd.to_datetime(observation_date) - DateOffset(years=y)
            end_date = observation_date
            print(start_date, end_date)

            temp_allegations = limit_allegations_to_time_period(allegations, start_date, end_date)
            temp_allegations["dispo_code"] = temp_allegations["ccrb_disposition__collapsed"].map(
                dispo_map
            )
            temp_complaint_summary = summarize_complaints_and_allegations(temp_allegations)

            temp_lawsuits = limit_lawsuits_to_time_period(lawsuits, start_date, end_date)
            temp_lawsuit_summary = summarize_lawsuits(temp_lawsuits)

            time_period_name = get_time_period_name(y)
            temp_complaint_summary = temp_complaint_summary.add_prefix(f"{time_period_name}.")
            temp_lawsuit_summary = temp_lawsuit_summary.add_prefix(f"{time_period_name}.")

            feature_df_list.append(temp_complaint_summary)
            feature_df_list.append(temp_lawsuit_summary)

        all_period_features = pd.concat(feature_df_list, axis=1, join="outer")
        all_period_features["observation_date"] = observation_date

        all_period_features.fillna(0, inplace=True)

        all_observation_list.append(all_period_features)

    all_features = pd.concat(all_observation_list)
    all_features.reset_index(inplace=True)
    all_features.set_index(["tax_id", "observation_date"], inplace=True)

    observation_table_w_features = pd.merge(
        observation_table,
        all_features,
        how="left",
        left_on=["tax_id", "observation_date"],
        right_index=True,
    )

    observation_table_w_features.fillna(0, inplace=True)

    return observation_table_w_features


if __name__ == "__main__":

    main_table = pd.read_parquet(
        "../create_observations_main_table/output/observation_table.parquet"
    )
    allegations = pd.read_parquet(
        "../clean_complaints_and_allegations/output/clean_allegations.parquet"
    )
    lawsuits = pd.read_parquet("../clean_lawsuits/output/clean_lawsuits.parquet")

    features = create_features(main_table, allegations, lawsuits)
    outcomes = create_outcomes(main_table, allegations, lawsuits, use_lawsuit_offset=True)

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    features.to_parquet(f"{output_dir}/features.parquet")
    outcomes.to_parquet(f"{output_dir}/outcomes.parquet")
