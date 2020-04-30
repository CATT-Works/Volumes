import pickle
import sys

import pandas as pd
import numpy as np
import tensorflow as tf

def delete_weird_counts(df, stddev_th = 3.0):
    base_sh = df.shape[0]
    meanstddf = pd.DataFrame(columns=['tmc', 'hour', 'hour_mean', 'hour_stddev'])
    tmcs = df.tmc.unique()
    idx = -1
    for tmcnr, tmc in enumerate(tmcs):
        sys.stdout.write('Processing {}. {:.1f}% ({} / {}) done.      \r'.format(tmc, 100 * tmcnr / len(tmcs), tmcnr, len(tmcs)))
        tmcdf = df.loc[df.tmc == tmc, :]
        hours = tmcdf.hour.unique()
        for hour in hours:
            idx += 1
            tmp = tmcdf[tmcdf.hour == hour]
            mean = tmp.count_total.mean()
            stddev = np.std(tmp.count_total)
            meanstddf.loc[idx, :] = [tmc, hour, mean, stddev]

    df = df.merge(meanstddf, on = ['tmc', 'hour'])
    df = df[np.abs(df.count_total - df.hour_mean) < stddev_th * df.hour_stddev]
    df = df.drop(["hour_mean", "hour_stddev"], axis=1)
    delnr = base_sh - df.shape[0]
    print ('{:.2f}% ({} / {}) deleted                             '.format(100*delnr / base_sh, delnr, base_sh))
    return df

def add_hourly_averaged_gps_counts(df):
    tmpdf = pd.DataFrame(columns=['tmc', 'hour', 
                                  'h_mean_wc1', 'h_mean_wc2', 'h_mean_wc3',
                                  'h_std_wc1', 'h_std_wc2', 'h_std_wc3'])
    tmcs = df.tmc.unique()
    idx = -1
    for tmcnr, tmc in enumerate(tmcs):
        sys.stdout.write('Processing {}. {:.1f}% ({} / {}) done.      \r'.format(tmc, 100 * tmcnr / len(tmcs), tmcnr, len(tmcs)))
        tmcdf = df.loc[df.tmc == tmc, :]
        hours = tmcdf.hour.unique()
        for hour in hours:
            idx += 1
            tmp = tmcdf[tmcdf.hour == hour]
            mean_wc1 = tmp.gps_wc1.mean()
            mean_wc2 = tmp.gps_wc2.mean()
            mean_wc3 = tmp.gps_wc3.mean()
            std_wc1 = np.std(tmp.gps_wc1)
            std_wc2 = np.std(tmp.gps_wc2)
            std_wc3 = np.std(tmp.gps_wc3)
            
            stddev = np.std(tmp.count_total)
            tmpdf.loc[idx, :] = [tmc, hour, mean_wc1, mean_wc2, mean_wc3, std_wc1, std_wc2, std_wc3]

    df = df.merge(tmpdf, on = ['tmc', 'hour'])
    return df


def get_tmcs_to_delete():
    tmcs = ['110+05562', '110-05561'] # No data for this TMC
    #tmcs = [-1] ## Just to be sure that it works
    return tmcs

def get_stations_to_delete():
    
    stations = [-1] ## Just to be sure that it works

    if False: # Florida
        stations = [126060, 126061] # Misassigned stations, the correct one is only 120184
        stations += [870382] # Misassigned. The correct one is 870031
        stations += [126053] # Misassigned. The correct one is 126010
        stations += [860381] # Weird station with 4 TMCs, it is easier to delete :)
        stations += [134007] # Misassigned. This station is located on the local road (no TMC)
        stations += [870187] # There is something weird with this station
        stations += [930217, 340116] #Duplicates, there are 2 stations for the same TMCs
    
    return stations

def add_timemask(df):
    mask = (df['datetime'] >= '2018-01-01') & (df['datetime'] < '2019-12-31 18:00')
    return df[mask]

def add_holidays(df, holidays = ['MD2018', 'MD2019']):
    """
    Adds information about holidays to the DataFrame
    Arguments:
        df        - Dataframe to be processed
        holidays  - Information about holidays. Could be string or a list of strings. Possible values:
                    MD2015 - maryland 2015 for Feb, May, Jun, Jul and Oct
                    FL2016 - Florida for 2016 Q4
                    NH2017 - New Hampshire for 2017 Q2
                    MD2018 - Maryland 2018 (Entire year) - Default
    """
        
    def add_holiday(df, colname, h_date):
        colname = 'day_' + colname
        if colname not in list(df.columns):
            df[colname] = 0
        df.loc[days == h_date, colname] = 1    
        
    days = df['datetime'].map(lambda x: x.strftime('%Y-%m-%d'))
    

    if isinstance(holidays, str):
        holidays = [holidays]
        
    if 'MD2015' in holidays:
        add_holiday(df, 'Washington_Birthday', '2015-02-16')
        add_holiday(df, 'Memorial', '2015-05-25')
        add_holiday(df, 'Independence', '2015-07-04')
        
    if 'FL2016' in holidays:
        add_holiday(df, 'Columbus', '2016-10-12') # Columbus Day
        add_holiday(df, 'Veterans', '2016-11-11') # Veterans Day
        add_holiday(df, 'Thanksgiving', '2016-11-24') # Thanksgiving Day
        add_holiday(df, 'Chrismas1', '2016-12-25') # Chrismas 1
        add_holiday(df, 'Chrismas2', '2016-12-26') # Chrismas 2
        
    if 'NH2017' in holidays:
        add_holiday(df, 'Independence', '2017-07-04') # Independence Day
        add_holiday(df, 'Labour', '2017-09-04') # Labour Day
    
    if 'MD2018' in holidays:
        add_holiday(df, 'New Year', '2018-01-01')
        add_holiday(df, 'MLK_Birthday', '2018-01-15')
        add_holiday(df, 'Presidents', '2018-02-19')    
        add_holiday(df, 'Memorial', '2018-05-28')
        add_holiday(df, 'Independence', '2018-07-04')
        add_holiday(df, 'Labor', '2018-09-03')
        add_holiday(df, 'Columbus', '2018-10-08') # Columbus Day
        add_holiday(df, 'Election', '2018-11-06') 
        add_holiday(df, 'Veterans', '2018-11-11') # Veterans Day
        add_holiday(df, 'Thanksgiving', '2018-11-22') # Thanksgiving Day
        add_holiday(df, 'Indian_Heritage', '2018-11-23') # Thanksgiving Day
        add_holiday(df, 'Chrismas1', '2018-12-25') # Chrismas 1
        add_holiday(df, 'Chrismas2', '2018-12-26') # Chrismas 2

    if 'MD2019' in holidays:
        add_holiday(df, 'New Year', '2019-01-01')
        add_holiday(df, 'MLK_Birthday', '2019-01-21')
        add_holiday(df, 'Presidents', '2019-02-18')    
        add_holiday(df, 'Memorial', '2019-05-27')
        add_holiday(df, 'Independence', '2019-07-04')
        add_holiday(df, 'Labor', '2019-09-02')
        add_holiday(df, 'Columbus', '2019-10-14') # Columbus Day
        add_holiday(df, 'Veterans', '2019-11-11') # Veterans Day
        add_holiday(df, 'Thanksgiving', '2019-11-28') # Thanksgiving Day
        add_holiday(df, 'Indian_Heritage', '2019-11-29') # Thanksgiving Day
        add_holiday(df, 'Chrismas1', '2019-12-25') # Chrismas 1
        add_holiday(df, 'Chrismas2', '2019-12-26') # Chrismas 2
        
        
    return df

def merge_columns(df, col1, col2, merged_name):
    df[merged_name] = df[col1]
    df.loc[df[merged_name].isnull(), merged_name] = df.loc[df[merged_name].isnull(), col2]
    df = df.drop([col1, col2], axis=1)
    return df

def merge_speeds(df):
    df = merge_columns(df, 'here_speed', 'npmrds_speed', 'speed')
    df = merge_columns(df, 'here_ref_speed', 'npmrds_ref_speed', 'ref_speed')
    df = merge_columns(df, 'here_tt_sec', 'npmrds_tt_sec', 'tt_sec')
    return df

def split_train_test_fixed(df):
    test_tmcs = pickle.load(open("../Data/test_tmcs.p", "rb"))
    df_train = df[~df.tmc.isin(test_tmcs)]    
    df_test = df[df.tmc.isin(test_tmcs)]
    return df_train, df_test

def add_null_indicator(df, columns = None):
    
    if columns is None:
        columns = list(df.columns)
    
    if type(columns) is str:
        columns = [columns]
        
    testcols = []
    for column in columns:
        colisnull = df[column].isnull()
        if colisnull.any():
            testcols += [column]
            isnull_colname = column + "_is_null"
            df[isnull_colname] = 0
            df.loc[colisnull, isnull_colname] = 1
        
    print ("Columns with null indicator: ", ", ".join(testcols))
    return df

def clear_monovalent_columns(df):
    cols = df.columns
    for c in cols:
        val = pd.unique(df[c])
        if len(val) == 1:
            df = df.drop([c], axis=1)
    
    print('Deleted monovalent columns: ', set(cols) - set(df.columns))
    return df


def change_values(df):    
    df.loc[df.osm_highway=='tertiary', 'osm_highway'] = 'secondary'
    df.loc[df.frc > 3, 'frc'] = 3
    df.loc[df.f_system==4, 'f_system'] = 3
    return df

def prepare_df(df, clear_monovalent = True, drop_zero_counts=True):    
    """
    The main function that prepares a dataframe for machine learning. This is a wrapper for other
    functions in prepare_data.py library. It works as follows:
    - deletes unwanted stations and tmcs
    - fills NaN values with zeros
    - drops the rows where ATR counts is 0 or NaN (optional)
    - filters data
    - adds temporal information (holidays etc.)
    - performs one-hot encoding
    - drops monovalent columns (optional)
    Arguments:
        df               - dataframe that shall be preprocessed
        clear_monovalent - if True (default) the monovalent columns are being dropped
        drop_zero_counts - if True (default) drops ATR counts that are zero or None
    """
    
    stations_to_delete = get_stations_to_delete()
    df = df[~df.count_location.isin(stations_to_delete)]
    
    tmc_to_delete = get_tmcs_to_delete()
    df = df[~df.tmc.isin(tmc_to_delete)]
    
    # Merging speed data - do not use with MD 2018 (only one speed set)
    #df = merge_speeds(df)

    
    fillnacols = [
        'gps_wc1_INRIX', 'gps_wc2_INRIX', 'gps_wc3_INRIX',
        'gps_wc1_CW', 'gps_wc2_CW', 'gps_wc3_CW',
        'gps_pt1_wc1_INRIX', 'gps_pt1_wc2_INRIX', 'gps_pt1_wc3_INRIX', 
        'gps_pt2_wc1_INRIX', 'gps_pt2_wc2_INRIX', 'gps_pt2_wc3_INRIX',
    ]
    for col in fillnacols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    if drop_zero_counts:
        df = df[df.count_total > 0]
    
    df = add_null_indicator(df)

    df = df.fillna(0)
    
    df['datetime'] = pd.to_datetime(df.date + ' ' + df.time, utc=None)  
    #df.datetime = df.datetime.dt.tz_localize('US/Eastern')        
    
    df = add_timemask(df)            
    df = add_holidays(df)
    
    df['weekday'] = df['datetime'].dt.dayofweek
    df = pd.get_dummies(df, columns=['hour', 'weekday', 'frc', 'f_system', 'osm_highway'])

    if clear_monovalent:
        df = clear_monovalent_columns(df)
    
    return df

def get_XY(df):   
    drop_cols = [
        'tmc', 'tmc_linear', 'tmc_dir', 'unix_ts', 'date', 'time', 'dow',
        'count_type', 'count_location',
        'datetime', # Not original column, added in code
    ]
    drop_cols = [x for x in drop_cols if x in df.columns]

    X = df.drop(drop_cols + ["count_total"], axis=1)
    y = df["count_total"]
    
    return X, y

