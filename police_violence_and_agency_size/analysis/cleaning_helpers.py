import pandas as pd
import numpy as np


def clean_name_field(df: pd.DataFrame, name_field: str,) -> pd.DataFrame:
    """
    Cleans the agency name text field
    
    Inputs:
        df (pandas dataframe object): the dataframe
        name_field (string): agency name column
        addl_cleaning (boolean): if True, do a little more cleaning than in V1 of analysis
        
    Returns:
        output_field (pandas dataframe object): the cleaned field 
    """

    field_to_clean = df[name_field]
    output_field = field_to_clean.copy()

    # remove special symbols
    output_field = output_field.str.replace("â€™", "")
    output_field = output_field.str.replace("�", "")

    # strip white space
    output_field = output_field.str.strip()

    # convert to lower case
    output_field = output_field.str.lower()

    # strip quotation marks
    output_field = output_field.str.replace("'", "")
    output_field = output_field.str.replace('"', "")

    # replace sheriff dept text
    output_field = output_field.str.replace("sheriffs office", "sheriffs department")

    # replace metro text
    output_field = output_field.str.replace("metropolitan", "metro")

    # replace st with saint
    output_field = output_field.str.replace("st\\.", "saint")

    # remove hyphen
    output_field = output_field.str.replace("-", " ")

    return output_field


def clean_mpv_data(
    df_mpv: pd.DataFrame,
    mpv_name_col: str = "agency_responsible",
    start_date: str = "2012-12-31",
    end_date: str = "2023-04-03",
) -> pd.DataFrame:
    """
    Cleans mpv data and collapses at the agency level
    
    Inputs: 
        df_mpv (dataframe): the raw mpv df
        start_date (string): the exclusive start date to slice the data
        end_date (string): the exclusive end date to slice the data
    
    Returns: df_mpv_collapsed (dataframe): the cleaned mpv df
    """
    # Drop data where agency_responsible is NaN
    df_mpv = df_mpv.dropna(subset=[mpv_name_col]).assign(
        mpv_agency_name=clean_name_field(df_mpv, mpv_name_col), date=pd.to_datetime(df_mpv["date"]),
    )

    # Limit the data to the start and end dates
    df_mpv_filtered = df_mpv[
        (df_mpv["date"] > pd.to_datetime(start_date))
        & ((df_mpv["date"] < pd.to_datetime(end_date)))
    ]

    # collapse the dataset by department
    df_mpv_collapsed = (
        df_mpv_filtered.groupby(["mpv_agency_name", "state"], as_index=False)
        .size()
        .reset_index()
        .rename(columns={0: "num_killings"})
    )

    return df_mpv_collapsed


def clean_roster_data(
    df: pd.DataFrame, agency_name_col: str = "NAME", manual_rename: bool = True
) -> pd.DataFrame:
    """
    Cleans roster data
    
    Inputs: 
        df (dataframe): the raw roster df
        manual_rename (boolean): if True, do manual renaming of the largest 
            agencies so that they match mpv data
  
    Returns: cleaned_df (dataframe): the cleaned roster df
    """
    cleaned_df = df.copy().assign(
        roster_agency_name=clean_name_field(df, agency_name_col),
        PE14_TOTAL_EMPLOYEES=lambda df: df["PE14_TOTAL_EMPLOYEES"]
        .replace(" ", np.nan, regex=True)
        .astype("float"),
        PE14_MALE_OFFICERS=lambda df: df["PE14_MALE_OFFICERS"]
        .replace(" ", np.nan, regex=True)
        .astype("float"),
        PE14_FEMALE_OFFICERS=lambda df: df["PE14_FEMALE_OFFICERS"]
        .replace(" ", np.nan, regex=True)
        .astype("float"),
        total_officers=lambda df: df["PE14_MALE_OFFICERS"] + df["PE14_FEMALE_OFFICERS"],
    )

    # Manually correct some of the misaligned names for the bigger departments
    if manual_rename:
        depts_to_rename_dict = {
            "new york city police department": "new york police department",
            "miamidade police department": "miami dade police department",
            "columbus police department": "columbus division of police",
            "philadelphia city police department": "philadelphia police department",
            "baltimore city police": "baltimore police department",
            "indianapolis police": "indianapolis metro police department",
            "metro police department, dc": "dc metro police department",
            "charlottemecklenburg police department": "charlotte mecklenburg police department",
            "greenville sheriffs department county": "greenville county sheriffs department",
            "baton rouge city police": "baton rouge police department",
            "cleveland police department": "cleveland division of police",
            "ogden city police department": "ogden police department",
            "pittsburgh city police department": "pittsburgh bureau of police",
            "e baton rouge parish sheriffs department": "east baton rouge sheriffs department",
            "syracuse city police department": "syracuse police department",
            "rochester city police department": "rochester police department",
            "st tammany parish sheriffs department": "saint tammany parish sheriffs department",
            "savannah chatham metro police department": "savannah police department",
            "sunnyvale department of public safety": "sunnyvale police department",
            "buffalo city police department": "buffalo police department",
            "kern county sd coroner": "kern county sheriffs department",
        }

        # replace original name with revision if in dict
        cleaned_df.loc[:, "roster_agency_name"] = np.where(
            cleaned_df["roster_agency_name"].isin(depts_to_rename_dict.keys()),
            cleaned_df["roster_agency_name"].map(depts_to_rename_dict),
            cleaned_df["roster_agency_name"],
        )

    return cleaned_df
