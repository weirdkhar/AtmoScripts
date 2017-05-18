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
import atmosplots as aplt
local_path_cn = 's:\\in2016_v03\\cpc\\'
local_path_uwy = 's:\\in2016_v03\\uwy\\'
local_path_pic = 'r:\\RV_Investigator\\GHGs\\Picarro\\'
local_path_aer = 'r:\\RV_Investigator\\GHGs\\Aerodyne\\'
local_path_ccn = 's:\\in2016_v03\\ccnc\\'

master_path = 'h:\\code\\AtmoScripts\\'




os.chdir(master_path)
if os.path.isfile('exhaust_filter_test_data.h5') & False:
    os.remove('exhaust_filter_test_data.h5')
#startdate = '2016-05-06 04:06:50'
startdate = '2016-05-16 00:00:00'
enddate = '2016-05-19 00:00:00'

#startdate = '2016-05-17 03:00:00'
#enddate = '2016-05-17 12:00:00'

'''
Determined:
    we want a filter that uses CO deviation, CN deviation and a BC threshold.
    
    Outlier detection should only be applied to individual datasets, not used 
    for determining exhaust.
    

'''
def main():
    df = load()
    #fwind = 150
    fwind = 60*10
    #==========================================================================
    # Create exhaust masks and plot after each step
    #==========================================================================
    
    #df1 = ghg.exhaust_flag_co(df.copy())
    df1 = exhaust_flag_rolling_std(df.copy(),
                                  column='CO',
                                  stat_window=10,
                                  stat_threshold=0.245,
                                  filt_around_window=fwind)
    
    #df2 = ghg.exhaust_flag_co2(df.copy())
    df2 = exhaust_flag_rolling_std(df.copy(),
                                  column='CO2_dry',
                                  stat_window=10,
                                  stat_threshold=0.035,
                                  filt_around_window=fwind)
#    figuring_out_std_threshold(df,column='BC')
#    df3 = exhaust_flag_rolling_std(df.copy(),
#                                  column='BC',
#                                  stat_window=10,
#                                  stat_threshold=2.36,
#                                  filt_around_window=fwind)
    
    df3 = exhaust_flag_bc(df.copy(),
                          bc_lim=0.07001,
                          filt_around_window=fwind)
    
    df4 = exhaust_flag_rolling_std(df.copy(),
                                   column='cn10',
                                   stat_window=10,
                                   #stat_threshold=16.2,
                                   stat_threshold=50,
                                   filt_around_window=fwind)
    
    '''
    df5 = rolling_outlier(df, df1, df2, df3, df4, 'cn10')
#    plot_wind_dir(df4)
    plot_com4_filters(df,df1,df2,df3,df4,df5,'cn10')
    df5 = rolling_outlier(df, df1, df2, df3, df4, 'CO')
    plot_com4_filters(df,df1,df2,df3,df4,df5,'CO')
    df5 = rolling_outlier(df, df1, df2, df3, df4, 'ccn')
    plot_com4_filters(df,df1,df2,df3,df4,df5,'ccn')
    '''
    plot_com3_filters(df,df1,df2,df3,df4,'cn10')
    plot_com3_filters(df,df1,df2,df3,df4,'CO')
    plot_com3_filters(df,df1,df2,df3,df4,'ccn')
    plot_raw(df)
    plot_single_filters(df, df1, df2, df3, df4)
    
    plot_combination_filters(df, df1, df2, df3, df4,'cn10')
    plot_combination_filters(df, df1, df2, df3, df4,'CO2')
    plot_combination_filters(df, df1, df2, df3, df4,'CO')
    
    return

def rolling_outlier(df, df1, df2, df3, df4, col='cn10'):
    df_co_cn_bc = df.copy()
    df_co_cn_bc[col].loc[
            np.array(df1['exhaust']) 
            | 
            np.array(df3['exhaust'])
            | 
            np.array(df4['exhaust'])
            ] = np.nan
    
    d = df_co_cn_bc[col]
    outliers = d.rolling(window=5,center=False).apply(outlier_detection)
    outliers.fillna(True,inplace=True)
    outliers = outliers.astype(bool)
    if 'exhaust' not in df:
        # Initialise
        df['exhaust'] = False
    
    df.loc[outliers,'exhaust'] = True
    
    return df

def outlier_detection(x):
    i = x[-1]
    ir = x[0:-1] #range before i
    x0_mean = ir.mean()
    x0_std = ir.std()
    if i > x0_mean + 3*x0_std:
        return True
    elif i < x0_mean - 3*x0_std:
        return True
    else:
        return False

def exhaust_flag_bc(df,bc_lim=0.05,filt_around_window=1):   
    if 'exhaust' not in df:
        # Initialise
        df['exhaust'] = False
    df['exhaust'].loc[df['BC'] > bc_lim] = True
    
    # Filter data around identified periods
    exhaust_rows = df['exhaust'].rolling(window=filt_around_window,
                                         center=True).apply(exhaust_window)
    exhaust_rows.fillna(True,inplace=True)
    exhaust_rows = exhaust_rows.astype(bool)
    df.loc[exhaust_rows,'exhaust'] = True
    return df

def figuring_out_std_threshold(df,column='cn10'):
    #startdate = '2016-05-06 04:06:50'
    threshold = [(df[column].rolling(window=x,center=True).std()).max() for x 
                 in np.arange(5,60*15,1)]
    plt.plot(np.arange(5,60*15,1),threshold,'.')
    plt.show()
    return

def exhaust_flag_rolling_std(df,
                             column='cn10', 
                             stat_window=10,
                             stat_threshold = 0.03,
                             filt_around_window=1):
    
    df[column+'_std'] = df[column].rolling(window=stat_window,center=True).std()
    #df['cn_stdminute'] = df[column].rolling(window=181,center=True).std()
    
    # Filter for min std over threshold
    exhaust_rows0 = df[column+'_std'] > stat_threshold
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

def plot_wind_dir(df):
    #df.loc[(df['WindDirRel_port'] > 0) & (df['WindDirRel_port'] < 360), 'exhaust'] = True
    #df['exhaust'].astype(bool)
    #df['cn10'].loc[(df['WindDirRel_port']>0) & (df['WindDirRel_port']<283)] = np.nan
    df['cn10'].loc[df['exhaust']] = np.nan

    plt.plot(df['WD_std'],df['cn10'],'.')
    fm = plt.get_current_fig_manager()
    fm.window.showMaximized()
    plt.show()
    return


    

def plot_raw(df):
    f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7,1,sharex=True,figsize=(10,20))
    aplt.plot_timeseries(axes_object = ax1,
                    x_data = df.index,
                    y_data = df[['cn10','ccn']],
                    legend=['cn10','ccn'],
                    ylim = [0,1000],
                    SaveOrShowPlot = 'wait'
                    )  
    aplt.plot_timeseries(axes_object = ax2,
                    x_data = df.index,
                    y_data = df['BC'],
                    title = 'BC',
                    ylim = [0,0.1],
                    SaveOrShowPlot = 'wait'
                    )  
    aplt.plot_timeseries(axes_object = ax3,
                    x_data = df.index,
                    y_data = df['CO'],
                    title = 'CO',
                    ylim = [45,65],
                    SaveOrShowPlot = 'wait'
                    )  
    aplt.plot_timeseries(axes_object = ax4,
                    x_data = df.index,
                    y_data = df['CO2_dry'],
                    title = 'CO2',
                    ylim = [395,400],
                    SaveOrShowPlot = 'wait'
                    )  
    aplt.plot_timeseries(axes_object = ax5,
                    x_data = df.index,
                    y_data = df['WindDirRel_port'],
                    title = 'Wind Direction',
                    ylim = [0,360],
                    SaveOrShowPlot = 'wait'
                    )
    aplt.plot_timeseries(axes_object = ax6,
                    x_data = df.index,
                    y_data = df['O3_2'],
                    title = 'Ozone',
                    ylim = [0,60],
                    SaveOrShowPlot = 'wait'
                    )
    aplt.plot_timeseries(axes_object = ax7,
                    x_data = df.index,
                    y_data = df['WindSpdRel_port'],
                    title = 'Wind Speed',
                    ylim = [0,60],
                    SaveOrShowPlot = 'wait'
                    )
    fm = plt.get_current_fig_manager()
    fm.window.showMaximized()
    plt.show()
    return

def plot_com4_filters(df,df1,df2,df3,df4,df5,plot_col='cn10'):
    df_co_cn_bc = df.copy()
    df_co_cn_bc[plot_col].loc[
            np.array(df1['exhaust']) 
            | 
            np.array(df3['exhaust'])
            | 
            np.array(df4['exhaust'])
            ] = np.nan
    
    df_c4 = df.copy()
    df_c4[plot_col].loc[np.array(df5['exhaust'])] = np.nan
    
    f, (ax1, ax2) = plt.subplots(2,1,sharex=True,figsize=(10,20))
    aplt.plot_timeseries(axes_object = ax1,
                    x_data = df_co_cn_bc.index,
                    y_data = df_co_cn_bc[plot_col],
                    title = plot_col + ' - co+cn+bc filter',
#                    ylim = [0,500],
                    SaveOrShowPlot = 'wait'
                    ) 
    aplt.plot_timeseries(axes_object = ax2,
                    x_data = df_c4.index,
                    y_data = df_c4[plot_col],
                    title = plot_col + ' - co+cn+bc+outlier filter',
#                    ylim = [0,500],
                    SaveOrShowPlot = 'wait'
                    )
    
    fm = plt.get_current_fig_manager()
    fm.window.showMaximized()
    plt.show()
    return

def plot_com3_filters(df,df1,df2,df3,df4,plot_col='cn10'):
    df_co_cn_bc = df.copy()
    df_co_cn_bc[plot_col].loc[
            np.array(df1['exhaust']) 
            | 
            np.array(df3['exhaust'])
            | 
            np.array(df4['exhaust'])
            ] = np.nan
    
    
    f, (ax1, ax2) = plt.subplots(2,1,sharex=True,figsize=(10,20))
    aplt.plot_timeseries(axes_object = ax1,
                    x_data = df.index,
                    y_data = df[plot_col],
                    title = 'Raw '+plot_col+' data',
#                    ylim = [0,500],
                    SaveOrShowPlot = 'wait'
                    )
    aplt.plot_timeseries(axes_object = ax2,
                    x_data = df_co_cn_bc.index,
                    y_data = df_co_cn_bc[plot_col],
                    title = 'co+cn+bc filter',
#                    ylim = [0,500],
                    SaveOrShowPlot = 'wait'
                    ) 
    fm = plt.get_current_fig_manager()
    fm.window.showMaximized()
    plt.show()
    return
    
def plot_combination_filters(df,df1,df2,df3,df4,plot_col='cn10'):
    df_c12 = df.copy()
    df_c13 = df.copy()
    df_c14 = df.copy()
    df_c23 = df.copy()
    df_c24 = df.copy()
    df_c34 = df.copy()
    df_c12[plot_col].loc[np.array(df1['exhaust']) | np.array(df2['exhaust'])] = np.nan
    df_c13[plot_col].loc[np.array(df1['exhaust']) | np.array(df3['exhaust'])] = np.nan
    df_c14[plot_col].loc[np.array(df1['exhaust']) | np.array(df4['exhaust'])] = np.nan
    df_c23[plot_col].loc[np.array(df2['exhaust']) | np.array(df3['exhaust'])] = np.nan
    df_c24[plot_col].loc[np.array(df2['exhaust']) | np.array(df4['exhaust'])] = np.nan
    df_c34[plot_col].loc[np.array(df3['exhaust']) | np.array(df4['exhaust'])] = np.nan
    df_c12 = df_c12.rename(columns = {plot_col:plot_col+'_c12'})
    df_c13 = df_c13.rename(columns = {plot_col:plot_col+'_c13'})
    df_c14 = df_c14.rename(columns = {plot_col:plot_col+'_c14'})
    df_c23 = df_c23.rename(columns = {plot_col:plot_col+'_c23'})
    df_c24 = df_c24.rename(columns = {plot_col:plot_col+'_c24'})
    df_c34 = df_c34.rename(columns = {plot_col:plot_col+'_c34'})
    df_c1213 = pd.concat([df_c12[plot_col+'_c12']+3,df_c13[plot_col+'_c13']],axis=1)
    df_c1423 = pd.concat([df_c14[plot_col+'_c14']+3,df_c23[plot_col+'_c23']],axis=1)
    df_c2434 = pd.concat([df_c24[plot_col+'_c24']+3,df_c34[plot_col+'_c34']],axis=1)
    
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,sharex=True,figsize=(10,20))
    aplt.plot_timeseries(axes_object = ax1,
                    x_data = df.index,
                    y_data = df[plot_col],
                    title = 'Raw '+plot_col+' data',
#                    ylim = [0,500],
                    SaveOrShowPlot = 'wait'
                    )  
    
    aplt.plot_timeseries(axes_object = ax2,
                    x_data = df_c1213.index,
                    y_data = df_c1213[[plot_col+'_c12',plot_col+'_c13']],
                    drawLegend = True,
                    legend=['CO+CO2','CO+BC'],
                    title = 'Filters',
#                    ylim = [0,500],
                    ylabel = 'Conc',
                    SaveOrShowPlot = 'wait'
                    )
    aplt.plot_timeseries(axes_object = ax3,
                    x_data = df_c1423.index,
                    y_data = df_c1423[[plot_col+'_c14',plot_col+'_c23']],
                    drawLegend = True,
                    legend=['CO+CN','CO2+BC'],
                    title = 'Filters',
#                    ylim = [0,500],
                    ylabel = 'Conc',
                    SaveOrShowPlot = 'wait'
                    )
    aplt.plot_timeseries(axes_object = ax4,
                    x_data = df_c2434.index,
                    y_data = df_c2434[[plot_col+'_c24',plot_col+'_c34']],
                    drawLegend = True,
                    legend=['CO2+CN','BC+CN'],
                    title = 'Filters',
#                    ylim = [0,500],
                    ylabel = 'Conc',
                    SaveOrShowPlot = 'wait'
                    )
    fm = plt.get_current_fig_manager()
    fm.window.showMaximized()
    plt.show()
    
    return

def plot_single_filters(df,df1,df2,df3,df4):
    
    for d in [df1, df2, df3, df4]:
        d['cn10'].loc[d['exhaust']] = np.nan
    
    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1,sharex=True,figsize=(10,20))
    aplt.plot_timeseries(axes_object = ax1,
                    x_data = df.index,
                    y_data = df['cn10'],
#                    y_data_R = df[y1_cols_right],
#                    logscale = True,
#                    logscale_R = y1_logscale_right,             
                    drawLegend = False,
#                    legend=y1_legend_labels,
#                    legend_R=y1_legend_labels_right,
                    title = 'Raw CN10 data',
                    ylim = [0,500],
#                    ylim_R = y1_lim_right,
#                    ylabel = y1_label,
#                    ylabel_R = y1_label_right,
                     
                    SaveOrShowPlot = 'wait'
                    )
    
    aplt.plot_timeseries(axes_object = ax2,
                    x_data = df1.index,
                    y_data = df1['cn10'],
                    title = 'co',
                    ylim = [0,500],
                    SaveOrShowPlot = 'wait'
                    )
    aplt.plot_timeseries(axes_object = ax3,
                    x_data = df2.index,
                    y_data = df2['cn10'],
                    title = 'co2',
                    ylim = [0,500],
                    SaveOrShowPlot = 'wait'
                    )
    
    aplt.plot_timeseries(axes_object = ax4,
                    x_data = df3.index,
                    y_data = df3['cn10'],
                    title = 'bc',
                    ylim = [0,500],
                    SaveOrShowPlot = 'wait'
                    )
    aplt.plot_timeseries(axes_object = ax5,
                    x_data = df4.index,
                    y_data = df4['cn10'],
                    title = 'cn',
                    ylim = [0,500],
                    SaveOrShowPlot = 'wait'
                    )
    fm = plt.get_current_fig_manager()
    fm.window.showMaximized()
    plt.show()
    return

def load():
    os.chdir(master_path)
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
        dfc = pd.read_hdf('CN3_raw_2016_wk20_flowCal_logFilt.h5',key='cn')
        dfc = dfc[startdate:enddate].copy()
        dfc.rename(columns={'Concentration':'cn10'},inplace=True)
        
        #======================================================================
        # Load ccn data
        #======================================================================
        os.chdir(local_path_ccn)
        dfcc = pd.read_hdf('CCN_raw_2016_wk20_QC_flowCal_ssCal_logFilt_ssSplit.h5',key='ccn')
        dfcc = dfcc[startdate:enddate].copy()
        dfcc.rename(columns={'ccn_0.5504':'ccn'},inplace=True)
        
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
        df = df.join(dfcc,how='outer')
        df = df.join(dfu,how='outer')
        
        # Interpolate missing uwy data caused by merging 1s and 5s data
        for col in dfu.columns:
            df[col] = df[col].interpolate(limit=4)
        #======================================================================
        # Save data for quick reuse
        #======================================================================
        os.chdir(master_path)
        df.to_hdf('exhaust_filter_test_data.h5',key='data')
    return df

main()