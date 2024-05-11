import pandas as pd
import os


def deduplicate_lawsuits(df):

    # We want to keep the last record in the case of duplicates since it will have the most up to date info

    id_cols = ["docket_number", "tax_id"]
    df.sort_values(by="export_year", inplace=True)

    return df.drop_duplicates(subset=id_cols, keep="last")


if __name__ == "__main__":

    # A dictionary that maps export date to file name
    lawsuit_list = {
        2018: "NYPD Alleged Misconduct Matters Commenced in CY 2014-2018.xls",
        2019: "NYPD Alleged Misconduct Matters Commenced in CY 2015-2019.xls",
        2020: "NYPD Alleged Misconduct Matters Commenced in CY 2016-2020.xls",
        2021: "NYPD Alleged Misconduct Matters Commenced in CY 2017-2021.xls",
        2022: "NYPD Alleged Misconduct Matters commenced in CY 2018-2022.xls",
    }

    input_dir = "../raw_data/"

    df_list = []
    for export_year in lawsuit_list:

        temp_df = pd.read_excel(f"{input_dir}/{lawsuit_list[export_year]}")
        temp_df["export_year"] = export_year
        df_list.append(temp_df)

    lawsuit_df = pd.concat(df_list)

    rename_cols = {
        "Docket/\nIndex#": "docket_number",
        "Tax #": "tax_id",
        "Lit Start": "lit_start",
        "Disp Date": "disp_date",
        "Total City Payout AMT": "total_city_payout",
        "Use of Force Alleged?": "use_of_force_allegation",
        "Assault/ Battery Alleged?": "assault_battery_allegation",
        "Malicious Prosecution Alleged?": "malicious_prosecution_allegation",
        "False Arrest/Imprisonment Alleged?": "false_arrest_imprison_allegation",
    }

    lawsuit_df.rename(columns=rename_cols, inplace=True)

    # Bad values
    lawsuit_df["tax_id"].replace("939185)", "939185", inplace=True)
    lawsuit_df["tax_id"].replace("*67642", None, inplace=True)

    lawsuit_df["lit_start"] = pd.to_datetime(lawsuit_df["lit_start"])
    lawsuit_df["disp_date"] = pd.to_datetime(lawsuit_df["disp_date"])

    convert_cols = [
        "use_of_force_allegation",
        "assault_battery_allegation",
        "malicious_prosecution_allegation",
        "false_arrest_imprison_allegation",
    ]

    for c in convert_cols:
        lawsuit_df[c] = (lawsuit_df[c] == "Y") * 1.0

    # Drop rows without tax_id since they aren't useful to us
    print("before dropping null tax_id", lawsuit_df.shape)
    lawsuit_df.dropna(subset=["tax_id"], inplace=True)
    print("after dropping null tax_id", lawsuit_df.shape)

    print("before removing duplicates", lawsuit_df.shape)
    lawsuit_df = deduplicate_lawsuits(lawsuit_df)
    print("after removing duplicates", lawsuit_df.shape)

    HIGH_PAYOUT_CUTPOINT = 50000
    lawsuit_df["high_payout_suit"] = 1.0 * (lawsuit_df["total_city_payout"] >= HIGH_PAYOUT_CUTPOINT)

    NORMALIZE_PAYOUTS = True
    if NORMALIZE_PAYOUTS == True:

        # If multiple officers are named on a suit, spread the cost equally over all of them to avoid double counting later
        lawsuit_df["num_officers_in_suit"] = lawsuit_df.groupby("docket_number")[
            "tax_id"
        ].transform("count")
        lawsuit_df["officer_payout"] = (
            lawsuit_df["total_city_payout"] / lawsuit_df["num_officers_in_suit"]
        )
        lawsuit_df.drop(columns=["num_officers_in_suit"], inplace=True)
    else:
        lawsuit_df["officer_payout"] = lawsuit_df["total_city_payout"]

    drop_cols = ["Matter Name", "Plaintiff & Firm", "Individual Defendants", "Represented by"]

    lawsuit_df.drop(columns=drop_cols, inplace=True)

    # Make sure this is an int for better matching later
    lawsuit_df["tax_id"] = lawsuit_df["tax_id"].astype(int)

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    lawsuit_df.to_parquet(f"{output_dir}/clean_lawsuits.parquet", index=False)
