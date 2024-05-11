import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from typing import List

sns.set_palette(("#193063", "#f0b660", "#56a4ba", "#f28157", "#6079B3", "#7d1b2a"))


def add_roster_size_cat(
    df: pd.DataFrame,
    roster_col: str,
    roster_bins: List[str],
    roster_labels: List[str],
    right: bool = False,
) -> pd.Series:
    """
    Add column that segments police department sizes into groups of roster sizes 
    """
    rost_size_cat = pd.cut(df[roster_col], bins=roster_bins, right=right, labels=roster_labels)
    return rost_size_cat


def make_cumulative_table(
    merged_df: pd.DataFrame,
    df_roster: pd.DataFrame,
    mpv_field: str = "num_killings",
    roster_field: str = "total_officers",
    bins: List[float] = [1, 100, 200, 500, 1000, float("inf")],
    labels: List[str] = ["1-99", "100-199", "200-499", "500-999", "1000+"],
) -> pd.DataFrame:
    """
    Generate table with cumulative shares of officers and police killings per agency roster size category
    """
    merged_df.loc[:, "rost_size_cat"] = add_roster_size_cat(
        merged_df, roster_field, bins, labels, right=False
    )
    df_roster.loc[:, "rost_size_cat"] = add_roster_size_cat(
        df_roster, roster_field, bins, labels, right=False
    )

    # calc summed percentages by rost_size_cat and merge into one df
    summed_employees_by_cat = df_roster.groupby("rost_size_cat")[roster_field].sum().reset_index()
    summed_killings_by_cat = merged_df.groupby("rost_size_cat")[mpv_field].sum().reset_index()

    cumulative_table = (
        pd.merge(summed_employees_by_cat, summed_killings_by_cat, on="rost_size_cat")
        .assign(
            cumulative_total_officers=lambda df: df["total_officers"].cumsum(),
            cumulative_num_killings=lambda df: df["num_killings"].cumsum(),
            cumulative_share_officers=lambda df: df["cumulative_total_officers"]
            / df["total_officers"].sum(),
            cumulative_share_killings=lambda df: df["cumulative_num_killings"]
            / df["num_killings"].sum()
        )
    )
    return cumulative_table


def make_roster_frequency_graph(
    df: pd.DataFrame,
    roster_field: str = "total_officers",
    row_field: str = "LEAR_ID",
    bins: List[float] = [1, 50, 100, 200, 500, 1000, float("inf")],
    labels: List[str] = ["1-49", "50-99", "100-199", "200-499", "500-999", "1000+"],
) -> None:
    """
    Generates bar graph with proportion of police departments by roster size
    """
    # generate roster size group variable
    df.loc[:, "rost_size_cat"] = add_roster_size_cat(df, roster_field, bins, labels, right=False)

    # reshape the data
    summed_rosters_by_cat = df.groupby("rost_size_cat")[row_field].count()
    tot_rosters = summed_rosters_by_cat.sum()
    summed_rosters_by_cat = summed_rosters_by_cat / tot_rosters

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=summed_rosters_by_cat.index, y=summed_rosters_by_cat.values)

    ax.set_title("Proportion of Departments by Size", fontdict={"size": 16})
    ax.set_xlabel("Police Department Size (Total Number of Officers)", fontdict={"size": 14})
    ax.set_ylabel("% of All Departments", fontdict={"size": 14})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    bar_labels = [f"{int(np.round(100*val))}%" for val in summed_rosters_by_cat]
    ax.bar_label(ax.containers[0], label_type="edge", labels=bar_labels)
    ax.axes.get_yaxis().set_ticks([])

    plt.show()


# helper function to make key bar graph
def make_bar_graph(
    merged_df: pd.DataFrame,
    df_roster: pd.DataFrame,
    mpv_field: str = "num_killings",
    roster_field: str = "total_officers",
    bins: List[float] = [1, 50, 100, 200, 500, 1000, float("inf")],
    labels: List[str] = ["1-49", "50-99", "100-199", "200-499", "500-999", "1000+"],
) -> None:
    """
    Generates double-bar chart comparing agency sizes and share of police killings
    """
    # generate roster size group variable
    merged_df.loc[:, "rost_size_cat"] = add_roster_size_cat(
        merged_df, roster_field, bins, labels, right=False
    )
    df_roster.loc[:, "rost_size_cat"] = add_roster_size_cat(
        df_roster, roster_field, bins, labels, right=False
    )

    # calc summed percentages by rost_size_cat and merge into one df
    summed_employees_by_cat = df_roster.groupby("rost_size_cat")[roster_field].sum().reset_index()
    summed_employees_by_cat["perc_of_tot_officers"] = (
        summed_employees_by_cat[roster_field] / summed_employees_by_cat[roster_field].sum()
    )

    summed_killings_by_cat = merged_df.groupby("rost_size_cat")[mpv_field].sum().reset_index()
    summed_killings_by_cat["perc_of_tot_killings"] = (
        summed_killings_by_cat[mpv_field] / summed_killings_by_cat[mpv_field].sum()
    )

    graph_df = pd.merge(summed_employees_by_cat, summed_killings_by_cat, on="rost_size_cat")
    graph_df = graph_df[["rost_size_cat", "perc_of_tot_killings", "perc_of_tot_officers"]]

    ax = graph_df.plot(kind="bar", x="rost_size_cat", width=0.8, rot=0, figsize=(10, 6))
    ax.axes.get_yaxis().set_ticks([])

    ax.set_xlabel("Police Department Size (total number of officers)", fontdict={"size": 14})
    ax.set_ylabel("% of Total", fontdict={"size": 14})

    ax.set_title("Police Department Size and Number of Killings", fontdict={"size": 16})
    ax.legend(["Killings", "Officers"], bbox_to_anchor=(1.05, 1), loc="upper left")

    bar_labels_killings = [f"{int(np.round(100*val))}%" for val in graph_df["perc_of_tot_killings"]]
    bar_labels_officers = [f"{int(np.round(100*val))}%" for val in graph_df["perc_of_tot_officers"]]

    ax.bar_label(ax.containers[0], label_type="edge", labels=bar_labels_killings)
    ax.bar_label(ax.containers[1], label_type="edge", labels=bar_labels_officers)

    # Show graph
    plt.show()
