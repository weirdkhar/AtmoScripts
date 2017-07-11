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
import rvi_exhaust_filter as exh
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import atmosplots as aplt
import functools
from scipy import signal

local_path_cn = 's:\\in2016_v03\\cpc\\'
local_path_uwy = 's:\\in2016_v03\\uwy\\'
local_path_pic = 'r:\\RV_Investigator\\GHGs\\Picarro\\'
local_path_aer = 'r:\\RV_Investigator\\GHGs\\Aerodyne\\'
local_path_ccn = 's:\\in2016_v03\\ccnc\\'

master_path = 'h:\\code\\AtmoScripts\\'

exhaust_path = 'r:\\RV_Investigator\\Exhaust\\Data\\'


os.chdir(master_path)
if os.path.isfile('exhaust_filter_test_data.h5') & False:
    os.remove('exhaust_filter_test_data.h5')

startdate = '2016-04-25'
enddate = '2016-06-30'

#startdate = '2016-05-07'
#enddate = '2016-05-09'

'''
Determined:
    we want a filter that uses CO deviation, CN deviation and a BC threshold.
    
    Outlier detection should only be applied to individual datasets, not used 
    for determining exhaust.
    

'''
    

    
def main(force_reload_exhaust = False,
         force_recalculate_exhaust = True,
         abridged_exhaust = False,
         co_id = True,
         cn_id = True,
         bc_id = True,
         
         filter_window = 60*20,
         
         co_stat_window = 60*5,
         co_num_devs = 4,
        
         cn_stat_window = 60*5,
         cn_num_devs = 4,
         
         bc_lim = 0.07001):
    
#    deviation_dev()
    
    df = load()
    
    dfe, dfcn, dfco, dfbc = exh.create_exhaust_id(df,
                                exhaust_path = exhaust_path,
                                startdate = startdate,
                                enddate = enddate,
                                force_reload=force_reload_exhaust,
                                force_recalculate = force_recalculate_exhaust,
                                abridged = abridged_exhaust,
                                co_id = co_id,
                                cn_id = cn_id,
                                bc_id = bc_id,
                                filter_window = filter_window,
                                co_stat_window = co_stat_window,
                                co_num_devs = co_num_devs, 
                                cn_stat_window = cn_stat_window,
                                cn_num_devs = cn_num_devs,                        
                                bc_lim = bc_lim
                                )
    df.to_hdf('in2016_v03_df.h5',key='data')
    dfe.to_hdf('in2016_v03_dfe.h5',key='data')
    dfcn.to_hdf('in2016_v03_dfcn.h5',key='data')
    dfco.to_hdf('in2016_v03_dfco.h5',key='data')
    dfbc.to_hdf('in2016_v03_dfbc.h5',key='data')
    
    for col in dfe.columns:
        if col in df.columns:
            df = df.drop(col,axis=1)
    d = df.join(dfe,how='outer')
    
    cn = d['cn10']
    ex = d['exhaust']
    cn_filt = cn.loc[~ex]
    if 'cn_median' in d.columns:
        plt.plot(cn,'.',cn_filt,'xr',dfcn['cn_median'],'-k',dfcn['cn_var_u'],'--k',dfcn['cn_var_l'],'--k')
    else:
        plt.plot(cn,'.',cn_filt,'xr',dfcn['cn10_median'],'-k',dfcn['cn10_var_u'],'--k',dfcn['cn10_var_l'],'--k')
    
    plt.ylim([0,2000])
    plt.show()
    
    plt.plot(d['cn_std'],'.',d['cn'],'x')
    plt.show()
    
    
    ex = df['exhaust']    
    df_filt = df.loc[~ex]
    
    if 'cn10' in df.columns:
        plt.plot(df['cn10'],'.',df_filt['cn10'],'.r')
    else:
        plt.plot(df['cn'],'.',df_filt['cn'],'.r')
    plt.title('CN raw and filt')
    plt.ylim([0,2000])
    plt.show()
    
    plt.plot(df['ccn'],'.',df_filt['ccn'],'.r')
    plt.title('CCN raw and filt')
    plt.ylim([0,2000])
    plt.show()

    plt.plot(df_filt['WindDirRel_vmean'],df_filt['cn10'],'.')
    plt.title('wind dir vs cn10')
    plt.show()
    plt.plot(df_filt['WindDirRel_port'],df_filt['CO'],'.')
    plt.title('wind dir vs co')
    plt.show()
    plt.plot(df_filt['WindDirRel_port'],df_filt['ccn'],'.')
    plt.title('wind dir vs ccn')
    plt.show()
    return d, df, dfe


def n_plt_cn_std(d):
    plt.plot(d['cn_std'],'.',d['cn'],'x')
    plt.show()
    return

def n_plt_filt(df):
    df_filt = df.copy()
    df_filt.loc[np.array(df['exhaust'])] = np.nan
    plt.plot(df['cn10'],'.',df_filt['cn10'],'.')
    plt.ylim([0,4000])
    plt.show()


def n_plt_cn_co_bc_raw(d):
    plt.plot(d['bc'],'.',d['cn'],'.',d['CO'],'.')
    plt.show()

###############################################################################
def deviation_dev(df = None):
    os.chdir(exhaust_path)
    
    #dfs_flat = df['2016-05-28 18:40':'2016-05-28 19:00'].copy()
    #dfs_slope = df['2016-05-28 19:15':'2016-05-28 19:35'].copy()
    dfs_flat = pd.read_csv('window_flat.csv')
    dfs_slope = pd.read_csv('window_slope.csv')
    '''
    dfs_detrend = pd.Series(signal.detrend(dfs_slope['cn10']))
    
    print('Flat stats: mean, std, median, mad')
    print(dfs_flat['cn10'].mean())
    print(dfs_flat['cn10'].std())
    print(dfs_flat['cn10'].median())
    print(MAD(dfs_flat['cn10']))
    
    print('Slope stats: mean, std, median, mad')
    print(dfs_slope['cn10'].mean())
    print(dfs_slope['cn10'].std())
    print(dfs_slope['cn10'].median())
    print(MAD(dfs_slope['cn10']))
    
    print('Detrend stats: std, mad')
    print(dfs_detrend.std())
    print(MAD(dfs_detrend))

    '''
    ###################
    dfs_poll = pd.read_csv('window_poll.csv')
    dfs_poll = dfs_poll.drop('exhaust',axis=1)
    col = 'CO'
    cn = dfs_poll[col]
    
    #cn = cn[10480:16000]
    #cn = cn[5000:len(cn)]
    ex = exh.exhaust_flag_rolling_var(dfs_poll,
                                  col,
                                  stat_window=60*10,
                                  num_deviations = 4
                                  )
    ex = exh.filt_surrounding_window(
                                ex,
                                filt_around_window=60*10
                                )
    
    cn_filt = cn.loc[~ex['exhaust']]
    plt.plot(cn,'.',cn_filt,'xr',ex[col+'_median'],'-k',ex[col+'_var_u'],'--k',ex[col+'_var_l'],'--k')
    plt.ylim([0,1000])
    plt.show()
    plt.plot(cn_filt,'.')
    plt.show()
    

    
#    param = rayleigh.fit(samp.dropna()[samp<100]) #distribution fitting
#    pdf_fitted = rayleigh.pdf(bins,loc=param[0],scale=param[1])
#    
#    plt.hist(samp.dropna()[samp<100],500,normed=1,alpha=0.75)
#    plt.plot(bins,pdf_fitted,'.')
#    plt.show()


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
    


def rolling_pop_median(x, med_p,mad_p):
    '''
    Determine whether the median is being calculated on an exhaust period and 
    thus is actually exhaust. 
    If the SAMPLE median is outside 4*MAD of the POPULATION median, then use
    the population median instead.
    
    '''
    if any(np.isnan(x)):
        return med_p
    if type(x) == np.ndarray:
        med_s = np.median(x)
    else:
        med_s = x.median()

    
    if (med_s > med_p + 10*mad_p) or (med_s < med_p - 10*mad_p):
        return med_p
    else:
        return med_s
###############################################################################
def main_old():
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
    plt.plot(df4['cn_std'],'.',df4['cn10'],'x')
    plt.show()
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
        print('Data loaded from ' + 'exhaust_filter_test_data.h5')
    else:
        #======================================================================
        # Load GHG data
        #======================================================================
        print('Loading GHG data')
        os.chdir(local_path_pic)
        dfp = pd.read_hdf('concat_picarro.h5',key='ghg')
        dfp = dfp[startdate:enddate].copy()
        dfp = ghg.resample_interpolation(dfp)
        
        os.chdir(local_path_aer)
        dfa = pd.read_hdf('concat_aerodyne.h5',key='ghg')
#        dfa = exh.load_co(local_path_aer, startdate, enddate)
        dfa = dfa[startdate:enddate].copy()
        dfa = ghg.resample_interpolation(dfa)
        
        #======================================================================
        # Load cn data
        #======================================================================
        print('Loading CN data')
        os.chdir(local_path_cn)
        secfiles = glob.glob('*logFilt.h5')
        dfc = []
        for file in secfiles:
            cn_temp = pd.read_hdf(file, key='cn')
            dfc.append(cn_temp)
        # Concatenate each df in the dictionary
        dfc = pd.concat(dfc)
        dfc.sort_index()
        #dfc = pd.read_hdf('CN3_raw_2016_wk20_flowCal_logFilt.h5',key='cn')
        dfc = dfc[startdate:enddate].copy()
        dfc.rename(columns={'Concentration':'cn10'},inplace=True)
        
        #======================================================================
        # Load ccn data
        #======================================================================
        print('Loading CCN data')
        os.chdir(local_path_ccn)
        secfiles = glob.glob('*ssSplit.h5')
        dfcc = []
        for file in secfiles:
            ccn_temp = pd.read_hdf(file, key='ccn')
            dfcc.append(ccn_temp)
        # Concatenate each df in the dictionary
        dfcc = pd.concat(dfcc)
        dfcc.sort_index()
        #dfcc = pd.read_hdf('CCN_raw_2016_wk20_QC_flowCal_ssCal_logFilt_ssSplit.h5',key='ccn')
        dfcc = dfcc[startdate:enddate].copy()
        dfcc.rename(columns={'ccn_0.5504':'ccn'},inplace=True)
        
        #======================================================================
        # Load uwy data
        #======================================================================
        print('Loading UWY data')
        os.chdir(local_path_uwy)
        dfu = pd.read_hdf('uwy_filt.h5',key='uwy')
        dfu = dfu[startdate:enddate].copy()
        #======================================================================
        # Merge datasets
        #======================================================================
        print('Merging data')
        df = dfa#.join(dfp,how='outer')
        df = df.join(dfc,how='outer')
        df = df.join(dfcc,how='outer')
        df = df.join(dfu,how='outer')
        # Interpolate missing uwy data caused by merging 1s and 5s data
#        for col in dfu.columns:
#            df[col] = df[col].interpolate(limit=4)
#            print('Interpolated ' + col)
        
        #======================================================================
        # Save data for quick reuse
        #======================================================================
        os.chdir(master_path)
        print('Saving data to file for quick reuse')
        df.to_hdf('exhaust_filter_test_data.h5',key='data')
        
    return df

# if this script is run at the command line, run the main script   
if __name__ == '__main__': 
	main()
