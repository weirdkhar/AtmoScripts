# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:11:38 2017

@author: hum094
"""
import sys
sys.path.append('h:\\code\\atmoscripts\\')
import os
import RVI_GHGs as ghg
import RVI_Underway
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

startdate = '2016-05-06 00:06:50'
enddate = '2016-05-06 06:00:00'

local_path_cn = 's:\\in2016_v03\\cpc\\'
local_path_uwy = 's:\\in2016_v03\\uwy\\'
local_path_pic = 'r:\\RV_Investigator\\GHGs\\Picarro\\'
local_path_aer = 'r:\\RV_Investigator\\GHGs\\Aerodyne\\'

master_path = 'h:\\code\\AtmoScripts\\'

def main():
    os.chdir(master_path)
    #os.remove('exhaust_filter_test_data.h5')
    if os.path.isfile('exhaust_filter_test_data.h5'):
        df = pd.read_hdf('exhaust_filter_test_data.h5', key='data')
    else:
        #======================================================================
        # Load GHG data
        #======================================================================
        os.chdir(local_path_pic)
        dfp = pd.read_hdf('concat_picarro.h5',key='ghg')
        dfp = dfp[startdate:enddate].copy()
        dfp = ghg.resample_interpolation(dfp)
        
        os.chdir(local_path_aer)
        dfa = pd.read_hdf('concat_aerodyne.h5',key='ghg')
        dfa = dfa[startdate:enddate].copy()
        dfa = ghg.resample_interpolation(dfa)
        
        #======================================================================
        # Load cn data
        #======================================================================
        os.chdir(local_path_cn)
        dfc = pd.read_hdf('CN3_raw_2016_wk18_flowCal_logFilt.h5',key='cn')
        dfc = dfc[startdate:enddate].copy()
        dfc.rename(columns={'Concentration':'cn10'},inplace=True)
        
        #======================================================================
        # Load uwy data
        #======================================================================
        os.chdir(local_path_uwy)
        dfu = pd.read_hdf('uwy_filt.h5',key='uwy')
        dfu = dfu[startdate:enddate].copy()
        
        #======================================================================
        # Merge datasets
        #======================================================================
        df = dfa.join(dfp,how='outer')
        df = df.join(dfc,how='outer')
        df = df.join(dfu,how='outer')
        
        #======================================================================
        # Save data for quick reuse
        #======================================================================
        os.chdir(master_path)
        df.to_hdf('exhaust_filter_test_data.h5',key='data')
        
    #==========================================================================
    # Create exhaust masks and plot after each step
    #==========================================================================
    
    #df1 = ghg.exhaust_flag_co(df.copy())
    
    #df2 = ghg.exhaust_flag_co2(df.copy())
    
    #df3 = exhaust_flag_bc(df.copy())
    
    df4 = exhaust_flag_rolling_std(df.copy(),'cn10')
    
    
    plOT(df4)
    
    plot_me(df, df1, df2, df3, df4)
    return


def exhaust_flag_bc(df):    
    RVI_Underway.create_exhaust_mask(df, 
                                 Filter4WindDir = False, 
                                 Filter4BC = True,
                                 Filter4O3 = False,
                                 Filter4CNstd = False,
                                 BC_lim = 0.05                                 
                                 )
    df.rename(columns={'exhaust_mask_L1':'exhaust'}, inplace=True)
    
    return df



def exhaust_flag_rolling_std(df,
                             column='cn10', 
                             stat_window=10,
                             filt_around_window=150):
    df[column+'_std'] = df[column].rolling(window=stat_window,center=True).std()
    #df['cn_stdminute'] = df[column].rolling(window=181,center=True).std()
    
    # Filter for min std over threshold
    exhaust_rows0 = df[column+'_std'] > 16.2
    # Filter for sec std over threshold
    #exhaust_rows0.loc[(df['cn_stdminute'] > 10.5)] = True
    
    # Filter data around identified periods
    exhaust_rows = exhaust_rows0.rolling(window=filt_around_window,
                                         center=True).apply(exhaust_window)
    exhaust_rows.fillna(True,inplace=True)
    exhaust_rows = exhaust_rows.astype(bool)
    
    if 'exhaust' not in df:
        # Initialise
        df['exhaust'] = False
    
    df.loc[exhaust_rows,'exhaust'] = True
    
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

def exhaust_flag_rolling_std(df,
                             column='cn10', 
                             stat_window=10,
                             stat_threshold = 0.03,
                             filt_around_window=150):
    df[column+'_std'] = df[column].rolling(window=stat_window,center=True).std()
    #df['cn_stdminute'] = df[column].rolling(window=181,center=True).std()
    
    # Filter for min std over threshold
    exhaust_rows0 = df[column+'_std'] > 16.2
    # Filter for sec std over threshold
    #exhaust_rows0.loc[(df['cn_stdminute'] > 10.5)] = True
    
    # Filter data around identified periods
    exhaust_rows = exhaust_rows0.rolling(window=filt_around_window,
                                         center=True).apply(exhaust_window)
    exhaust_rows.fillna(True,inplace=True)
    exhaust_rows = exhaust_rows.astype(bool)
    
    if 'exhaust' not in df:
        # Initialise
        df['exhaust'] = False
    
    df.loc[exhaust_rows,'exhaust'] = True
    
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
def plOT(df):
    df['cn10_filt'] = df['cn10']
    df['cn10_filt'].loc[df['exhaust']] = np.nan
    plt.plot(df['cn10_std'],'.g',df['cn10'],'.b',df['cn10_filt'],'or',)#,df['cn_stdminute'],'.')
    plt.ylim(0,1000)
    plt.show()
    return

def plot_me(df):
    
    
    
    df['CO_filt'] = df['CO']
    df['CO_filt'].loc[df['exhaust']] = np.nan
    
    '''
    plt.plot(df['CO'],'.')#,
             #df['CO_std']+45,'.',
             #df['CO_stdminute']+45,'.',
             df['CO_filt'],'.',
             #)
    plt.show()
    
    '''
    
    
    df['CO2_filt'] = df['CO2_dry']
    df['CO2_filt'].loc[df['exhaust']] = np.nan
 
    
    plt.plot(df['CO2_dry'],'.',
             df['CO2_filt'],'x',
             df['CO']+350,'.',
             df['CO_filt']+350,'x',
             df['N2O']+75,'.'
             #df['CO2_dry_std']+395,'.',
             #df['CO2_dry_stdminute']+395,'.'
             )
    plt.show()
    
    return


main()