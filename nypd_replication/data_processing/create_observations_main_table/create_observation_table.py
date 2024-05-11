import pandas as pd
import numpy as np
from itertools import product
import os

from datetime import datetime

if __name__ == "__main__":

    clean_roster = pd.read_parquet("../clean_roster/output/clean_roster.parquet")
    start_year = 2013
    end_year = 2020

    observation_years = [datetime(x, 1, 1) for x in np.arange(start_year, end_year + 1)]
    observations = list(product(clean_roster.tax_id.values, observation_years))

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    observation_df = pd.DataFrame(observations, columns=["tax_id", "observation_date"])
    observation_df.to_parquet(f"{output_dir}/observation_table.parquet")
