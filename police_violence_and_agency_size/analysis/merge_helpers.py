import pandas as pd


def merge_on_names(
    df_mpv: pd.DataFrame,
    df_roster: pd.DataFrame,
    mpv_agency_col: str = "mpv_agency_name",
    roster_agency_col: str = "roster_agency_name",
    multi_agency_rule: str = "all",
) -> pd.DataFrame:
    """
    Merges roster and mpv data. To deal with messiness between datasets, we do a 3-step merge. In the MPV 
    data, sometimes multiple agencies are included in the agency name column. After merging once on 
    agency name and state, we take the unmatched agencies in the MPV data and convert the agency name column (
    which could contain multiple agencies) into a separate row (or only take the first agency) 
    before merging again with the roster df. We then take the remaining MPV non-matches, replace the substring 
    'police department' with 'police' in the agency_namecolumn, and do a final merge. We concatenate these three sets 
    of matches and aggregate to the agency level to get our final df with roster sizes and counts of killings. 
    
    Inputs:
        df_mpv_collapsed (dataframe): the mpv dataframe collapsed by agency
        df_roster (dataframe): the roster dataframe
        multi_agency_rule (string): rule input for how to handle when multiple agencies
            are listed in mpv agency field
    
    Returns:
        collapsed_merged_df (dataframe): the dataframe of merged data, collapsed at the agency level
    """
    assert multi_agency_rule in ("first", "all"), "invalid value for 'multi_agency_rule'"

    mpv_merge_cols = [mpv_agency_col, "state"]
    roster_merge_cols = [roster_agency_col, "STATE"]

    # merge left on name and state
    initial_merge = df_mpv.merge(
        df_roster,
        how="left",
        left_on=mpv_merge_cols,
        right_on=roster_merge_cols,
        indicator="merge_1",
    )

    # separate the matches and non-matches
    first_merge_matches = initial_merge[initial_merge["merge_1"] == "both"]
    mpv_nonmatch_data = initial_merge[initial_merge["merge_1"] == "left_only"][df_mpv.columns]

    # handle multiple agencies in the agency field
    if multi_agency_rule == "first":
        # truncate the agency field ahead of the first comma (if it exists). If no comma, field doesn't change
        mpv_nonmatch_data.loc[:, mpv_agency_col] = (
            mpv_nonmatch_data.loc[:, mpv_agency_col].str.split(",").str[0]
        )

    elif multi_agency_rule == "all":
        # create column with list of agencies, split by comma, then transform each agency in list into separate row
        mpv_nonmatch_data.loc[:, mpv_agency_col] = mpv_nonmatch_data.loc[
            :, mpv_agency_col
        ].str.split(",")
        mpv_nonmatch_data = mpv_nonmatch_data.explode(column=mpv_agency_col)

    # Merge on multi-agencies
    second_merge = mpv_nonmatch_data.merge(
        df_roster,
        how="left",
        left_on=[mpv_agency_col, "state"],
        right_on=[roster_agency_col, "STATE"],
        indicator="merge_2",
    )

    second_merge_matches = second_merge[second_merge["merge_2"] == "both"]
    mpv_nonmatch_multi = second_merge[second_merge["merge_2"] == "left_only"][df_mpv.columns]

    # replace instances of 'police department' with police in mpv data before final merge
    mpv_nonmatch_multi.loc[:, mpv_agency_col] = mpv_nonmatch_multi.loc[
        :, mpv_agency_col
    ].str.replace("police department", "police")

    third_merge_matches = mpv_nonmatch_multi.merge(
        df_roster,
        how="inner",
        left_on=[mpv_agency_col, "state"],
        right_on=[roster_agency_col, "STATE"],
        indicator="merge_3",
    )

    # Concat the merged dfs together
    merged_df = pd.concat([first_merge_matches, second_merge_matches, third_merge_matches])

    # Collapse again given we have duplicates in the agency field; drop agencies with no officers
    collapsed_merged_df = (
        merged_df.groupby([mpv_agency_col, "state"], as_index=False)
        .agg(
            {
                "LEAR_ID": "first",
                roster_agency_col: "first",
                "CITY": "first",
                "num_killings": "sum",
                "PE14_TOTAL_EMPLOYEES": "first",
                "total_officers": "first",
            }
        )
        .dropna(subset=["total_officers"])
    )

    return collapsed_merged_df
