# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:42:59 2017

@author: hum094
"""
from dateutil.parser import parse
import datetime as dt
import pandas as pd
import os
import shutil 
import glob
import matplotlib.pyplot as plt
import numpy as np



startdate = '2016-04-25'
enddate = '2016-06-30'

source_drive_pic = 'j'
local_path_pic = 'r:\\RV_Investigator\\GHGs\\Picarro\\'

source_drive_aer = 'k'
local_path_aer = 'r:\\RV_Investigator\\GHGs\\Aerodyne\\'



def main():
    #transfer_aerodyne_co_n2o_raw()
    #df = read_aerodyne_data(local_path_aer,startdate,enddate)
    #os.chdir(local_path_aer)
    #df.to_hdf('concat_aerodyne.h5',key='ghg')
    
    #transfer_picarro_co2_ch4_raw()
    #df = read_picarro_data(local_path_pic,startdate,enddate)
    #os.chdir(pic_local_path)
    #df.to_hdf('concat_picarro.h5',key='ghg')
    
    os.chdir(local_path_pic)
    dfp = pd.read_hdf('concat_picarro.h5',key='ghg')
    dfp0 = dfp['2016-05-06 00:06:50':'2016-05-06 06:00:00'].copy()
    dfp = resample_interpolation(dfp0)
    
    os.chdir(local_path_aer)
    dfa = pd.read_hdf('concat_aerodyne.h5',key='ghg')
    dfa0 = dfa['2016-05-06 00:06:50':'2016-05-06 06:00:00'].copy()
    dfa = resample_interpolation(dfa0)
    
    df = dfa.join(dfp,how='outer')
    
    plot_me(df)
    return



def plot_me(df):
    
    
    df = exhaust_flag_co2(df)
    
    df['CO2_filt'] = df['CO2_dry']
    df['CO2_filt'].loc[df['exhaust_filt']] = np.nan
 
    
    plt.plot(df['CO2_dry'],'.',
             df['CO2_filt'],'x', 
             df['CO2_dry_std']+395,'.',
             df['CO2_dry_stdminute']+395,'.',
             df['CO']+350,'o')
    plt.show()
    
    return

#==============================================================================
#  Useful aligning functions
#==============================================================================

def resample_interpolation(df):
    '''
    Aligns the timestamp to the rounded second interval by interpolating 
    between sample points.
    '''
    t = df.index
    ai = pd.date_range(start=t.min().date(), end=t.max(), freq='1S')
    df_interp = df.reindex(t.union(ai)).interpolate().ix[ai].dropna()
    return df_interp

#==============================================================================
#  Create exhaust flags
#==============================================================================
def exhaust_flag_co(df):
    
    return

def exhaust_flag_co2(df):
    df['CO2_dry_std'] = df['CO2_dry'].rolling(window=9,
                                              center=True).std()
    df['CO2_dry_stdminute'] = df['CO2_dry'].rolling(window=181,
                                                    center=True).std()
    
    # Filter for min std over threshold
    exhaust_rows0 = df['CO2_dry_stdminute'] > 0.03
    # Filter for sec std over threshold
    exhaust_rows0.loc[(df['CO2_dry_std'] > 0.05)] = True
    
    # Filter data around identified periods
    exhaust_rows = exhaust_rows0.rolling(window=300,
                                         center=True).apply(exhaust_window)
    exhaust_rows.fillna(True,inplace=True)
    exhaust_rows = exhaust_rows.astype(bool)
    
    # Initialise
    
    df['exhaust_filt'] = False
    df.loc[exhaust_rows,'exhaust_filt'] = True

    return df

def exhaust_window(x):
    '''
    if any value within the passed window satisfies the value, then return true
    
    This is used as a moving window to remove data either side of a filter 
    event
    '''
    if any(x):
        return True
    else:
        return False
    



def exhaust_flag_OutlierIteration(df, column):
    '''
    Takes the chosen column and fits harmonics, polynomials, and performs 
    fourier transforms to enable outlier identification.
    
    Iterates through this process multiple times.
    '''
    return

#==============================================================================
#  Read data
#==============================================================================


def read_aerodyne_data(local_path_aer= 'r:\\RV_Investigator\\GHGs\\Aerodyne\\', 
                      startdate = '2015-01-30', 
                      enddate = '2015-02-02', 
                      abridged = True):
    '''
    reads aerodyne data from str files, reading only the FIRST columns of 
    N2O and CO 
    '''
    assert parse(startdate), "Cannot recognise startdate format, please check"
    assert parse(enddate), "Cannot recognise enddate format, please check"    
    assert os.path.exists(local_path_aer), 'check local aerodyne path'
    
    s_date = parse(startdate)
    e_date = parse(enddate)
    
    years = list(range(s_date.year, e_date.year+1))
    
    df = []
    for yr in years:
        try:
            os.chdir(local_path_aer+str(yr))
        except:
            continue
        filelist = glob.glob('*.str')
        filelist.sort()
        for file in filelist:
            file_date = dt.datetime.strptime(file[0:6], '%y%m%d')
            if (file_date >= s_date) & (file_date <= e_date):
                if abridged:
                    try:
                        d = pd.read_csv(file,
                                delim_whitespace=True,
                                skiprows=2, 
                                header=None,
                                usecols=[0,1,3])
                    except:
                        print('No data in file - skipping')
                        continue
                else:
                    try:
                        d = pd.read_csv(file,
                                delim_whitespace=True,
                                skiprows=2, 
                                header=None)
                    except:
                        print('No data in file - skipping')
                        continue
                df.append(d)
    # Concatenate each df in the dictionary
    try:
        df = pd.concat(df)
    except:
        print('No data to load! Check your input dates')
        return None
    
    if abridged:
        df.columns = ['seconds_since','N2O','CO']
    else:
        df.columns = ['seconds_since','N2O','N2O_1','CO','CO_1','H2O']
    
    # Format timestamp and set as index
    timezero = dt.datetime(1904,1,1,0,0)
    df['Timestamp'] = [timezero + dt.timedelta(seconds=x) \
                      for x in df['seconds_since']]
    df = df.set_index('Timestamp')
    
    del df['seconds_since']
    df.sort_index()
    
    
    return df


def read_picarro_data(pic_local_path= 'r:\\RV_Investigator\\GHGs\\Picarro\\', 
                      startdate = '2014-01-01', 
                      enddate = '2014-01-02', 
                      abridged = True):
    '''
    Returns a dataframe after reading & concatenating data from raw ASCII files
    '''
    assert parse(startdate), "Cannot recognise startdate format, please check"
    assert parse(enddate), "Cannot recognise enddate format, please check"    
    assert os.path.exists(pic_local_path), 'check local picarro path'
    
    s_date = parse(startdate)
    e_date = parse(enddate)
    
    years = list(range(s_date.year, e_date.year+1))
    
    df = []
    for yr in years:
        try:
            os.chdir(pic_local_path+str(yr))
        except:
            continue
        filelist = glob.glob('*.dat')
        filelist.sort()
        for file in filelist:
            file_date = dt.datetime.strptime(file.split('-')[1],'%Y%m%d')
            if (file_date >= s_date) & (file_date <= e_date):
                d = pd.read_csv(file, delim_whitespace=True)
                df.append(d)
    # Concatenate each df in the dictionary
    try:
        df = pd.concat(df)
    except:
        print('No data to load! Check your input dates')
        return None
    
    df.sort_index()
    
    # Format timestamp and set as index
    df['Timestamp'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
    df = df.set_index('Timestamp')
    
    # Select only a few columns if requested
    if abridged:
        df_abridged = df[['CH4','CH4_dry', 'CO2', 'CO2_dry', 'H2O']]
        return df_abridged
    else:
        return df


#==============================================================================
# Transfer from remote server
#==============================================================================

def transfer_picarro_co2_ch4_raw(driveletter='j', 
                            dest_dir='r:\\RV_Investigator\\GHGs\\Picarro\\', 
                            startdate = '2014-01-01', 
                            enddate = '2050-01-01'):
    
    
    assert parse(startdate), "Cannot recognise startdate format, please check"
    assert parse(enddate), "Cannot recognise enddate format, please check"    
    assert os.path.exists(driveletter+':\\'), 'check driveletter'
    
    s_date = parse(startdate)
    e_date = parse(enddate)
    
        
    subdirs = [x[0]+'\\' \
               for x in os.walk(driveletter + ':\\') if len(x[0])==13]
    subdirs.sort()
    years = [x.split('\\')[1] for x in subdirs]
    yrs = list(set(years)) # get unique years
    month = [int(x.split('\\')[2]) for x in subdirs]
    day = [int(x.split('\\')[3]) for x in subdirs]
    
    for yr in yrs:
        for i in range(0, len(subdirs)):
            
            # Only transfer files between the selected dates
            if (years[i] == yr) & \
               (dt.datetime(int(years[i]),month[i],day[i]) >= s_date) &\
               (dt.datetime(int(years[i]),month[i],day[i]) <= e_date):
                   # Make the new folder
                   if not os.path.exists(dest_dir + yr):
                       os.makedirs(dest_dir + yr)
                   
                   # Get a list of files and transfer
                   os.chdir(subdirs[i])
                   filelist = glob.glob('*.dat')
                   for file in filelist:
                       src = subdirs[i]+file
                       dst = dest_dir + yr + '\\' + file
                       shutil.copy(src,dst)
                       
    return

def transfer_aerodyne_co_n2o_raw(driveletter='k', 
                             dest_dir='r:\\RV_Investigator\\GHGs\\Aerodyne\\', 
                             startdate = '2014-01-01', 
                             enddate = '2050-01-01'):
    '''
    Pulls str files from the remote server (those not in folders)
    
    '''
    assert parse(startdate), "Cannot recognise startdate format, please check"
    assert parse(enddate), "Cannot recognise enddate format, please check"    
    assert os.path.exists(driveletter+':\\'), 'check driveletter'
    
    s_date = parse(startdate)
    e_date = parse(enddate)
    
    os.chdir(driveletter+':\\')
    filelist = glob.glob('*str')
    filelist.sort()
    
    years = [x[0:2] for x in filelist]
    months = [int(x[2:4]) for x in filelist]
    days = [int(x[4:6]) for x in filelist]
    
    yrs_unique = list(set(years)) # get unique years
    yrs_unique.sort()
    
    for yr in yrs_unique:
        for i in range(0,len(filelist)):
            if (years[i] == yr) & \
               (dt.datetime(2000+int(years[i]),months[i],days[i]) >= s_date) &\
               (dt.datetime(2000+int(years[i]),months[i],days[i]) <= e_date):
                   # Make the new folder
                   if not os.path.exists(dest_dir + str(2000+int(yr))):
                       os.makedirs(dest_dir + str(2000+int(yr)))
                   
                   # Get a list of files and transfer
                   src = driveletter + ':\\' + filelist[i]
                   dst = dest_dir + str(2000+int(yr)) + '\\' + filelist[i]
                   shutil.copy(src,dst)
                   print('Copying ' + src + ' to ' + dst)
                
    return

main()