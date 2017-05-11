# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:42:59 2017

@author: hum094
"""
from dateutil.parser import parse
import datetime
import pandas as pd
import os
import shutil 
import glob
import matplotlib.pyplot as plt
import numpy as np



startdate = '2016-04-25'
enddate = '2016-06-30'

source_pic_drive = 'j'
pic_local_path = 'r:\\RV_Investigator\\GHGs\\Picarro\\'



def main():
    #transfer_picarro_co2_ch4_raw(source_pic_drive,dest_dir = pic_local_path,startdate,enddate)
    #transfer_picarro_co2_ch4_raw()
    #df = read_picarro_data(pic_local_path,startdate,enddate)
    #os.chdir(pic_local_path)
    #df.to_hdf('concat_picarro.h5',key='ghg')
    
    os.chdir(pic_local_path)
    df = pd.read_hdf('concat_picarro.h5',key='ghg')
    
    plot_me(df)
    return


def plot_me(df):
    dfa = df['2016-05-06 00:06:50':'2016-05-06 06:00:00'].copy()
    
    dfa = exhaust_flag_co2(dfa)
    
    dfa['CO2_filt'] = dfa['CO2_dry']
    dfa['CO2_filt'].loc[dfa['exhaust_filt']] = np.nan
    '''
    exhaust_rows = dfa['CO2_dry_std'] > 0.05
    for i in range(0, len(exhaust_rows)):
        #if (dfa['CO2_dry_std'][i] > 0.05) or \
        if (dfa['CO2_dry_stdminute'][i] > 0.03):
            # remove 1 minute around each exhaust identified time
            exhaust_rows.loc[dfa.index[i]-datetime.timedelta(minutes=0):dfa.index[i]+datetime.timedelta(minutes=0)] = True
    '''      
    
    plt.plot(dfa['CO2_dry'],'.',
             dfa['CO2_filt'],'x', 
             dfa['CO2_dry_std']+395,'.',
             dfa['CO2_dry_stdminute']+395,'.')
    plt.show()
    
    return

#==============================================================================
# 
#==============================================================================
def exhaust_flag_co2(df):
    df['CO2_dry_std'] = df['CO2_dry'].rolling(window=9,center=True).std()
    df['CO2_dry_stdminute'] = df['CO2_dry'].rolling(window=181,center=True).std()
    
    # Filter for min std over threshold
    exhaust_rows0 = df['CO2_dry_stdminute'] > 0.03
    # Filter for sec std over threshold
    exhaust_rows0.loc[(df['CO2_dry_std'] > 0.05)] = True
    
    # Filter data around identified periods
    exhaust_rows = exhaust_rows0.rolling(window=300,center=True).apply(exhaust_window)
    exhaust_rows.fillna(True,inplace=True)
    exhaust_rows = exhaust_rows.astype(bool)
    
    # Initialise
    
    df['exhaust_filt'] = False
    df.loc[exhaust_rows,'exhaust_filt'] = True

    return df

def exhaust_window(x):
    '''
    if any value within the passed window satisfies the value, then return true
    
    This is used as a moving window to remove data either side of a filter event
    '''
    if any(x):
        return True
    else:
        return False
    







#==============================================================================
# 
#==============================================================================

def exhaust_flag_OutlierIteration(df, column):
    '''
    Takes the chosen column and fits harmonics, polynomials, and performs 
    fourier transforms to enable outlier identification.
    
    Iterates through this process multiple times.
    '''
    return

def transfer_aerodyne_co_raw(drive, startdate, enddate):
    '''
    Pulls str files from the remote server (those not in folders)
    
    '''
    return

def read_aerodyne_data():
    '''
    reads aerodyne data from str files, reading only the FIRST columns of 
    N2O and CO 
    '''
    return df

#==============================================================================
# 
#==============================================================================







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
        os.chdir(pic_local_path+str(yr))
        filelist = glob.glob('*.dat')
        filelist.sort()
        for file in filelist:
            file_date = datetime.datetime.strptime(file.split('-')[1], '%Y%m%d')
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



def transfer_picarro_co2_ch4_raw(driveletter='j', 
                                 dest_dir='r:\\RV_Investigator\\GHGs\\Picarro\\', 
                                 startdate = '2014-01-01', 
                                 enddate = '2050-01-01'):
    
    
    assert parse(startdate), "Cannot recognise startdate format, please check"
    assert parse(enddate), "Cannot recognise enddate format, please check"    
    assert os.path.exists(driveletter+':\\'), 'check driveletter'
    
    s_date = parse(startdate)
    e_date = parse(enddate)
    
        
    subdirs = [x[0]+'\\' for x in os.walk(driveletter + ':\\') if len(x[0]) == 13]
    subdirs.sort()
    years = [x.split('\\')[1] for x in subdirs]
    yrs = list(set(years)) # get unique years
    month = [int(x.split('\\')[2]) for x in subdirs]
    day = [int(x.split('\\')[3]) for x in subdirs]
    
    for yr in yrs:
        for i in range(0, len(subdirs)):
            
            # Only transfer files between the selected dates
            if (years[i] == yr) & \
               (datetime.datetime(int(years[i]),month[i],day[i]) >= s_date) &\
               (datetime.datetime(int(years[i]),month[i],day[i]) <= e_date):
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



main()