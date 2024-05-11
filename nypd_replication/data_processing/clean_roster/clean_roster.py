import pandas as pd

if __name__ == "__main__":

    raw_roster = pd.read_csv(
        "../raw_data/Civilian_Complaint_Review_Board__Police_Officers_20231126.csv",
        parse_dates=["As Of Date", "Last Reported Active Date"],
    )

    roster = raw_roster.copy()

    rename_cols = {
        "Tax ID": "tax_id",
        "As Of Date": "as_of_date",
        "Last Reported Active Date": "last_reported_active_date",
    }

    roster.rename(columns=rename_cols, inplace=True)

    output_dir = "output"

    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    roster.to_parquet(f"{output_dir}/clean_roster.parquet", index=False)
