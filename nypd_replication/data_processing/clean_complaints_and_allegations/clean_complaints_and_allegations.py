import pandas as pd
import os


def clean_fado_type(x):

    return x.lower().replace(" ", "_")


def recode_allegation_types(x):
    """
        We can see lots of duplicates like = Refusal to provide name and Refusal to provide name/shield number
        In the future, we might want to combine these into similar levels. 
        For now, do nothing
    """

    return x


def recode_ethnicity(x):

    """
        To-do
    """

    return x


recode_dispositions = {
    "Unsubstantiated": "not_substantiated",
    "Exonerated": "not_substantiated",
    "Complainant Uncooperative": "truncated",
    "Complaint Withdrawn": "truncated",
    "Complainant Unavailable": "truncated",
    "Unfounded": "not_substantiated",
    "Closed - Pending Litigation": "truncated",
    "Miscellaneous - Subject Resigned": "substantiated",
    "Within NYPD Guidelines": "not_substantiated",
    "Substantiated (Formalized Training)": "substantiated",
    "Alleged Victim Uncooperative": "truncated",
    "Unable to Determine": "not_substantiated",
    "Substantiated (Command Discipline B)": "substantiated",
    "Alleged Victim Unavailable": "truncated",
    "Substantiated (Charges)": "substantiated",
    "Miscellaneous - Subject Terminated": "substantiated",
    "Miscellaneous": "not_substantiated",
    "Substantiated (Command Discipline)": "substantiated",
    "Miscellaneous - Subject Retired": "substantiated",
    "Victim Unidentified": "truncated",
    "Substantiated (Command Discipline A)": "substantiated",
    "Substantiated (Command Lvl Instructions)": "substantiated",
    "Substantiated (Instructions)": "substantiated",
    "Substantiated (No Recommendations)": "substantiated",
    "Witness Uncooperative": "truncated",
    "Witness Unavailable": "truncated",
    "Officer(s) Unidentified": "truncated",
    "none": "none",
    "Formalized Training": "substantiated",
    "Mediation Attempted": "truncated",
    "Mediated": "substantiated",
    "Administratively Closed": "truncated",
}


def recode_ccrb_disposition(x):

    return recode_dispositions[x]


def recode_nypd_disposition(x):

    return recode_dispositions[x]


if __name__ == "__main__":

    dates = ["CCRB Received Date", "Incident Date", "Close Date"]

    raw_complaints = pd.read_csv(
        "../raw_data/Civilian_Complaint_Review_Board__Complaints_Against_Police_Officers_20231126.csv",
        parse_dates=dates,
    )
    complaints = raw_complaints.copy()

    rename_cols = {
        "Incident Date": "incident_date",
        "CCRB Received Date": "received_date",
        "Close Date": "close_date",
        "Complaint Id": "complaint_id",
    }

    complaints.rename(columns=rename_cols, inplace=True)

    complaints["incident_date"] = pd.to_datetime(complaints["incident_date"], errors="coerce")
    complaints["received_date"] = pd.to_datetime(complaints["received_date"], errors="coerce")
    complaints["close_date"] = pd.to_datetime(complaints["close_date"], errors="coerce")

    raw_allegations = pd.read_csv(
        "../raw_data/Civilian_Complaint_Review_Board__Allegations_Against_Police_Officers_20231126.csv"
    )

    allegations = raw_allegations.copy()

    rename_cols = {"Tax ID": "tax_id", "Complaint Id": "complaint_id"}
    allegations.rename(columns=rename_cols, inplace=True)

    temp_cols = ["incident_date", "received_date", "close_date"]

    allegations = pd.merge(
        allegations,
        complaints[["complaint_id"] + temp_cols],
        left_on="complaint_id",
        right_on="complaint_id",
        how="left",
    )

    allegations["allegation"] = allegations["Allegation"].apply(recode_allegation_types)
    allegations["FADO Type"] = allegations["FADO Type"].apply(clean_fado_type)

    allegations["vic_ethnicity"] = allegations["Victim / Alleged Victim Race / Ethnicity"].fillna(
        allegations["Victim / Alleged Victim Race (Legacy)"]
    )
    allegations["vic_ethnicity"] = allegations["vic_ethnicity"].apply(recode_ethnicity)

    allegations["ccrb_disposition__collapsed"] = (
        allegations["CCRB Allegation Disposition"].fillna("none").map(recode_ccrb_disposition)
    )
    # allegations['nypd_disposition__collapsed'] = allegations['NYPD Allegation Disposition'].fillna('none').map(recode_nypd_disposition)

    complaints["ccrb_disposition__collapsed"] = (
        complaints["CCRB Complaint Disposition"].fillna("none").map(recode_nypd_disposition)
    )

    # Dropping allegations for which there's no officer indetified.
    allegations.dropna(subset=["tax_id"], inplace=True)

    drop_cols = [
        "Victim / Alleged Victim Race / Ethnicity",
        "Victim / Alleged Victim Race (Legacy)",
        "Officer Rank At Incident",
        "CCRB Allegation Disposition",
        "NYPD Allegation Disposition",
    ]
    allegations.drop(columns=drop_cols, inplace=True)
    allegations = pd.get_dummies(allegations, prefix=["FADO"], columns=["FADO Type"])

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    allegations.to_parquet(f"{output_dir}/clean_allegations.parquet", index=False)
    complaints.to_parquet(f"{output_dir}/clean_complaints.parquet", index=False)
