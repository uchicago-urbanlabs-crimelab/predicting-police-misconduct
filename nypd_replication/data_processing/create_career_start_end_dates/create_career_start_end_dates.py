import pandas as pd
import numpy as np
import os
from pandas.tseries.offsets import DateOffset


def load_raw_payroll():

    payroll_list = [
        
        '../raw_data/payroll.csv',
        '../raw_data/payroll_2000_2009.csv',
        '../raw_data/payroll_2010_2019.csv'
    ]

    all_payroll = []
    for p in payroll_list:
        temp_df = pd.read_csv(p)
        temp_df['source'] = p
        all_payroll.append(temp_df)

    raw_payroll = pd.concat(all_payroll)
    return raw_payroll


def get_start_dates_from_payroll():
    
    payroll = load_raw_payroll()
    
    payroll = payroll[payroll['Agency']=='POLICE DEPARTMENT'].copy()
    payroll['start_date'] = pd.to_datetime(payroll['Start.date'],utc=True, errors='coerce').dt.date
    payroll['name'] = payroll['First.name'].str.lower() + '__' + payroll['Last.name'].str.lower()

    print('before dedup',payroll.shape)
    payroll.drop_duplicates(subset=['name','start_date'],inplace=True)
    print('after dedup',payroll.shape)
    
    
    payroll_start_dates = payroll.groupby('name').agg({
        'start_date':['first','count'],
    })

    payroll_start_dates.columns = ['payroll_start_date','payroll_name_count']
    payroll_start_dates.reset_index(inplace=True)
    
    payroll_start_dates['payroll_start_date'] = pd.to_datetime(payroll_start_dates['payroll_start_date'])

    return payroll_start_dates

def merge_payroll_and_roster():

    payroll_start_dates = get_start_dates_from_payroll()
    
    roster = pd.read_parquet('../clean_roster/output/clean_roster.parquet')

    roster['name'] = roster['Officer First Name'].str.lower() + '__' + roster['Officer Last Name'].str.lower()

    roster = pd.merge(
        roster,
        payroll_start_dates,
        how='left',
        left_on='name',
        right_on='name',indicator=True)

    roster['matched_to_payroll'] = roster['_merge']=='both'
    roster['payroll_name_conflict'] = (roster['payroll_name_count'] > 1)

    roster['payroll_start_date__valid'] = (roster['payroll_name_conflict']==False) & (roster['matched_to_payroll']==True)
    return roster

def get_career_starts_from_allegations():
    
    allegations = pd.read_parquet('../clean_complaints_and_allegations/output/clean_allegations.parquet')
    allegations['allegations_implied_career_start'] = allegations['incident_date'] - pd.to_timedelta(allegations['Officer Days On Force At Incident'], unit='D')
    allegation_career_starts = allegations.groupby('tax_id')['allegations_implied_career_start'].min()
    
    return allegation_career_starts


if __name__=='__main__':
    
    roster_w_payroll_starts = merge_payroll_and_roster()
    allegation_career_starts = get_career_starts_from_allegations()
    roster_w_payroll_starts = pd.merge(roster_w_payroll_starts,allegation_career_starts, how='left',left_on='tax_id',right_index=True)
    roster_w_payroll_starts['career_start_date'] = np.where(roster_w_payroll_starts['payroll_start_date__valid']==True, 
                                                       roster_w_payroll_starts['payroll_start_date'],
                                                       roster_w_payroll_starts['allegations_implied_career_start'])

    cols = ['tax_id','career_start_date','last_reported_active_date']
    roster_career_dates = roster_w_payroll_starts[cols].copy()

    roster_career_dates['career_end_date'] = roster_career_dates['last_reported_active_date'].dt.date + DateOffset(years=1)
    career_date_cols = ['tax_id','career_start_date','career_end_date']

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    roster_career_dates[career_date_cols].to_parquet(f'{output_dir}/career_dates.parquet')

