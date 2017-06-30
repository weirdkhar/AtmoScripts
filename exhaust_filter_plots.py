# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:09:13 2017

@author: hum094
"""

import sys
sys.path.append('h:\\code\\atmoscripts\\')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import atmosplots as aplt
from matplotlib.backends.backend_pdf import PdfPages
from atmosplots import saveorshowplot as handle_plt

master_path = 'h:\\code\\AtmoScripts\\'
exhaust_path = 'r:\\RV_Investigator\\'
plt_path = 'r:\\RV_Investigator\\Exhaust\\'

saveorshowplot = 'save'
#dfe,dfcn,dfco,dfbc,dfe_f,dfcn_f,dfco_f,dfbc_f
def main():
    plt_ts_cn10('show')
    boxplot_madarrays(saveorshowplot)
    plt_compare_wdws(saveorshowplot)
    subplt_wd('cn10',saveorshowplot)
    subplt_wd('CO',saveorshowplot)
    
    plt_ts_cn10(saveorshowplot)
    plt_ts_cn10_f(saveorshowplot)
    
    plt_ts_co(saveorshowplot)
    plt_ts_co_f(saveorshowplot)
    
    plt_ts_ccn(saveorshowplot)
    plt_ts_ccn_f(saveorshowplot)
    
    plt_ts_co2(saveorshowplot)
    plt_ts_co2_f(saveorshowplot)
    print('Done!')
    return

def boxplot_madarrays(saveorshowplot='save'):
    # https://matplotlib.org/examples/pylab_examples/broken_axis.html
    os.chdir(master_path)
    mcn = pd.read_hdf('mad_array_cn10.h5',key='mad')
    mco = pd.read_hdf('mad_array_co.h5',key='mad')
    

    f, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharex='col')
    
    f.subplots_adjust(hspace=0.1)
    
    plt.suptitle('Tukey plot \n Rolling Median Absolute Deviation')
    f.text(0.04,0.5,'Number Concentration ($cm^{-3}$)',va='center',rotation='vertical')
    f.text(0.96,0.5,'Mixing ratio ($ppb$)',va='center',rotation='vertical')
    
    ax1.boxplot(mcn.dropna(),0,'xb')
    ax2.boxplot(mcn.dropna(),0,'xb')
    
    ax1.set_yscale('log')
    ax1.set_ylim([75,10**6])
    ax2.set_ylim([0,75])
    
    plt.setp(ax2,xticklabels=['$CN_{10}$'])
    
    
    # hide the spines between ax and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop='off')  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    
    
    ax3.boxplot(mco.dropna(),0,'xb')
    ax4.boxplot(mco.dropna(),0,'xb')
    
    ax3.set_yscale('log')
    ax3.yaxis.tick_right()
    ax4.yaxis.tick_right()
    ax3.set_ylim([0.35,10**3])
    ax4.set_ylim([0,0.35])
    
    plt.setp(ax4,xticklabels=['$CO$'])
    
    # hide the spines between ax and ax2
    ax3.spines['bottom'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax3.xaxis.tick_top()
    ax3.tick_params(labeltop='off')  # don't put tick labels at the top
    ax4.xaxis.tick_bottom()
    
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax3.transAxes, color='k', clip_on=False)
    ax3.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax3.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    
    kwargs.update(transform=ax4.transAxes)  # switch to the bottom axes
    ax4.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax4.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    
    handle_plt(plt,saveorshowplot,plt_path,outputfilename='boxplot_madarray')
    
    return

def subplt_wd(column='cn10',saveorshowplot='save'):
    plt.figure(figsize=(8,11))
    
    ax = plt.subplot(5,1,1)
    plt.plot(dfe['WindDirRel_vmean'],dfe[column],'.')
    plt.title('Wind direction vs ' + column + ' - raw data')
    plt.ylabel('Number Conc ($cm^{-3}$)')
    plt.xlim([0,360])
    ax.xaxis.set_visible(False)
    
    ax = plt.subplot(5,1,2)
    plt.plot(dfbc_f['WindDirRel_vmean'],dfbc_f[column],'.')
    plt.title('Wind direction vs ' + column + ' - BC filter')
    plt.ylabel('Number Conc ($cm^{-3}$)')
    plt.xlim([0,360])
    ax.xaxis.set_visible(False)
    
    ax = plt.subplot(5,1,3)
    plt.plot(dfco_f['WindDirRel_vmean'],dfco_f[column],'.')
    plt.title('Wind direction vs ' + column + ' - CO filter')
    plt.ylabel('Number Conc ($cm^{-3}$)')
    plt.xlim([0,360])
    ax.xaxis.set_visible(False)
    
    ax = plt.subplot(5,1,4)
    plt.plot(dfcn_f['WindDirRel_vmean'],dfcn_f[column],'.')
    plt.title('Wind direction vs ' + column + ' - CN filter')
    plt.ylabel('Number Conc ($cm^{-3}$)')
    plt.xlabel('Relative Wind Direction (degrees)')
    plt.xlim([0,360])
    ax.xaxis.set_visible(False)
    
    ax = plt.subplot(5,1,5)
    plt.plot(dfe_f['WindDirRel_vmean'],dfe_f[column],'.')
    plt.title('Wind direction vs ' + column + ' - full filter')
    plt.ylabel('Number Conc ($cm^{-3}$)')
    plt.xlim([0,360])
    ax.xaxis.set_visible(False)
    
    handle_plt(plt,saveorshowplot,plt_path,outputfilename='subplt_wd_'+column)
    return

def plt_ts_window(saveorshowplot='save'):
    import rvi_exhaust_filter as exh
    dfe_w15 = exh.filt_surrounding_window(dfcomb.copy(),60*15)
    dfe_w15_f = dfe_w15.loc[~dfe_w15['exhaust']]

    dfe_w20 = exh.filt_surrounding_window(dfcomb.copy(),60*20)
    dfe_w20_f = dfe_w20.loc[~dfe_w20['exhaust']]
    
    dfe_w25 = exh.filt_surrounding_window(dfcomb.copy(),60*25)
    dfe_w25_f = dfe_w25.loc[~dfe_w25['exhaust']]

    dfe_w30 = exh.filt_surrounding_window(dfcomb.copy(),60*30)
    dfe_w30_f = dfe_w30.loc[~dfe_w30['exhaust']]
    
    print("% loss from combination only") 
    print((1 - len(dfcomb_f['exhaust'])/len(dfe['exhaust']))*100)
    print("% loss from combination + 10 min window") 
    print((1 - len(dfe_f['exhaust'])/len(dfe['exhaust']))*100)
    print("% loss from combination + 15 min window")
    print((1 - len(dfe_w15_f['exhaust'])/len(dfe['exhaust']))*100)
    print("% loss from combination + 20 min window") 
    print((1 - len(dfe_w20_f['exhaust'])/len(dfe['exhaust']))*100)
    print("% loss from combination + 25 min window") 
    print((1 - len(dfe_w25_f['exhaust'])/len(dfe['exhaust']))*100)
    print("% loss from combination + 30 min window") 
    print((1 - len(dfe_w30_f['exhaust'])/len(dfe['exhaust']))*100)
    
    plt.plot(dfcomb_f['cn10'],'.',dfe_f['cn10'],'.g', dfe_w30_f['cn10'],'.r')
    
    plt.ylim([0,5000])
    plt.legend('No window','10min window', '20min window')
    handle_plt(plt,saveorshowplot,plt_path,outputfilename='ts_cn10_filter_window')
    return


def plt_ts_cn10(saveorshowplot='save'):
    plt.figure(figsize=(11,8))
    
    plt.plot(dfe['cn10'],'.b',dfe_f['cn10'],'.r',dfcn['cn10_median'],'-k',dfcn['cn10_var_u'],'--k',dfcn['cn10_var_l'],'--k')
    plt.ylabel('Number Conc ($cm^{-3}$)')
    plt.ylim([0,2000])
    plt.title('cn10 timeseries - full filt with cn10 filter parameters')
    
    handle_plt(plt,saveorshowplot,plt_path,outputfilename='ts_cn10_raw')
    return

def plt_ts_cn10_f(pdf=None,saveorshowplot='save'):
    plt.figure(figsize=(11,8))
    
    plt.plot(dfe_f['cn10'],'.')
    plt.ylabel('Number Conc ($cm^{-3}$)')
    plt.ylim([0,2000])
    plt.title('cn10 timeseries - cn10 full filt only')
    
    handle_plt(plt,saveorshowplot,plt_path,outputfilename='ts_cn10_full_filter')
    return
    
    
def plt_ts_co(saveorshowplot='save'):
    plt.figure(figsize=(11,8))
    
    plt.plot(dfe['CO'],'.b',dfe_f['CO'],'.r',dfco['CO_median'],'-k',dfco['CO_var_u'],'--k',dfco['CO_var_l'],'--k')
    #plt.ylim([0,2000])
    plt.ylabel('Mixing ratio (ppb)')
    
    plt.title('CO timeseries - full filt with CO filter parameters')
    
    handle_plt(plt,saveorshowplot,plt_path,outputfilename='ts_co_raw')
    return

def plt_ts_co_f(saveorshowplot='save'):
    plt.figure(figsize=(11,8))
    
    plt.plot(dfe_f['CO'],'.')
    plt.ylabel('Mixing ratio (ppb)')
    #plt.ylim([0,2000])
    plt.title('CO timeseries - CO full filt only')
    
    handle_plt(plt,saveorshowplot,plt_path,outputfilename='ts_co_full_filter')
    return

def plt_ts_ccn(saveorshowplot='save'):
    plt.figure(figsize=(11,8))
    
    plt.plot(dfe['ccn'],'.b',dfe_f['ccn'],'.r')
    plt.ylabel('Number Conc ($cm^{-3}$)')
    plt.ylim([0,1000])
    plt.title('ccn timeseries')
    
    handle_plt(plt,saveorshowplot,plt_path,outputfilename='ts_ccn_raw')
    return
    
def plt_ts_ccn_f(saveorshowplot='save'):
    plt.figure(figsize=(11,8))
    
    plt.plot(dfe_f['ccn'],'.')
    plt.ylabel('Number Conc ($cm^{-3}$)')
    #plt.ylim([0,2000])
    plt.title('ccn timeseries - ccn full filt only')
    
    handle_plt(plt,saveorshowplot,plt_path,outputfilename='ts_ccn_full_filter')
    return

def plt_ts_co2(saveorshowplot='save'):
    plt.figure(figsize=(11,8))
    
    plt.plot(dfe['co2'],'.b',dfe_f['co2'],'.r')
    plt.ylabel('Mixing ratio (ppm)')
    plt.ylim([0,1000])
    plt.title('co2 timeseries')    
    
    handle_plt(plt,saveorshowplot,plt_path,outputfilename='ts_co2_raw')
    return

def plt_ts_co2_f(saveorshowplot='save'):
    plt.figure(figsize=(11,8))
    
    plt.plot(dfe_f['co2'],'.')
    plt.ylabel('Mixing ratio (ppm)')
    plt.title
    #plt.ylim([0,2000])
    plt.title('co2 timeseries - co2 full filt only')
    
    handle_plt(plt,saveorshowplot,plt_path,outputfilename='ts_co2_full_filter')
    return
    
def plt_wdfilt(saveorshowplot='save'):
    dfwd = dfe[[False if ((wd>90) and (wd<270)) else True for wd in dfe['WindDirRel_vmean']]]
    plt.plot(dfwd['WindDirRel_vmean'],dfwd['cn10'],'.')
    plt.xlim([0,360])
    
    handle_plt(plt,saveorshowplot,plt_path,outputfilename='wd_cn10_wd_filter')
    return

def plt_ts_cn10_wdfilt(saveorshowplot='save'):
    dfwd = dfe[[False if ((wd>90) and (wd<270)) else True for wd in dfe['WindDirRel_vmean']]]
    plt.plot(dfwd['cn10'],'.')
    
    
    handle_plt(plt,saveorshowplot,plt_path,outputfilename='ts_cn10_wd_filter')
    return

def plt_wd_hist(saveorshowplot='save'):
    dfe['WindDirRel_vmean'][dfe['WindDirRel_vmean']>0].hist(bins=180)
    plt.xlim([0,360])
    handle_plt(plt,saveorshowplot,plt_path,outputfilename='hist_wd')
    return

def plt_wdwsfilt(saveorshowplot='save'):
    dfwdws = dfe[[False if (((wd>90) and (wd<270)) or ws<10) else True for wd,ws in zip(dfe['WindDirRel_vmean'],dfe['WindSpdRel_port'])]]
    plt.plot(dfwdws['WindDirRel_vmean'],dfwdws['cn10'],'.')
    plt.xlim([0,360])
    
    handle_plt(plt,saveorshowplot,plt_path,outputfilename='wd_cn10_wdws_filter')
    return

def plt_compare_wdws(saveorshowplot='save'):
    dfwdws = dfe[[False if (((wd>90) and (wd<270)) or ws<10) else True for wd,ws in zip(dfe['WindDirRel_vmean'],dfe['WindSpdRel_port'])]]
    
    plt.plot(dfe['WindDirRel_vmean'],dfe['cn10'],'.',             dfwdws['WindDirRel_vmean'],dfwdws['cn10'],'.r')

    plt.xlabel('Relative Wind Direction ($^o$)')
    plt.ylabel('Num Conc ($cm^{-3}$)')
    
    plt.legend(['raw','wind filter'])
    
    plt.xlim([0,360])
    
    handle_plt(plt,saveorshowplot,plt_path,outputfilename='wd_cn10_wdws_filter_compare')
    return

def load():
    os.chdir(exhaust_path)
    startdate = '2016-04-25 00:00:00'
    enddate = '2016-06-09 00:00:00'
    
    print('Loading data from file - this may take a few seconds. Please wait...')
    
    dfe = pd.read_hdf('in2016_v03_dfe.h5',key='data')[startdate:enddate]
    dfcn = pd.read_hdf('in2016_v03_dfcn.h5',key='data')[startdate:enddate]
    dfco = pd.read_hdf('in2016_v03_dfco.h5',key='data')[startdate:enddate]
    dfbc = pd.read_hdf('in2016_v03_dfbc.h5',key='data')[startdate:enddate]
    
    
    # Create a combined filter without the filtering around the window
    
    # Merge into one
    ecn = dfcn['exhaust']
    ebc = dfbc['exhaust']
    eco = dfco['exhaust']
    ex = pd.Series([True if any([cn,bc,co]) else False for cn,bc,co in zip(ecn,ebc,eco)],index=eco.index)
    dfcomb = dfe.copy()
    dfcomb['exhaust'] = ex
    
    print('Data loaded successfully from file')
    
  
    print('Creating filtered datasets')
    
    dfcomb_f = dfcomb.loc[~ex]
    
    ex = dfe['exhaust']    
    dfe_f = dfe.loc[~ex]
    
    ex = dfcn['exhaust']    
    dfcn_f = dfcn.loc[~ex]
    
    ex = dfco['exhaust']    
    dfco_f = dfco.loc[~ex]
    
    ex = dfbc['exhaust']    
    dfbc_f = dfbc.loc[~ex]
    
    print('Done!')
    
    return dfe,dfcn,dfco,dfbc,dfcomb,dfe_f,dfcn_f,dfco_f,dfbc_f,dfcomb_f

dfe,dfcn,dfco,dfbc,dfcomb,dfe_f,dfcn_f,dfco_f,dfbc_f,dfcomb_f = load()

# if this script is run at the command line, run the main script   
if __name__ == '__main__': 
	main()
