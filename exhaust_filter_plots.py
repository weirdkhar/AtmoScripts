# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:09:13 2017

@author: hum094
"""
saveorshowplot = 'show'
load_all_data = True
load_partial = False


import sys
sys.path.append('h:\\code\\atmoscripts\\')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import atmosplots as aplt
from matplotlib.backends.backend_pdf import PdfPages
from atmosplots import saveorshowplot as handle_plt

master_path_svr = 'h:\\code\\AtmoScripts\\'
master_path_lcl = 'c:\\OneDrive\\RuhiFiles\\Research\\ProgramFiles\\git\\AtmoScripts\\'
exhaust_path_svr = 'r:\\RV_Investigator\\Exhaust\\Data\\'
exhaust_path_lcl = 'c:\\Temp\\ExhaustData\\'
plt_path_svr = 'r:\\RV_Investigator\\Exhaust\\'
plt_path_lcl = 'c:\\OneDrive\\RuhiFiles\\Research\\Writing\\Writing_RVI\\AMT_RVI_ExhaustFilter\\Plots\\'



#dfe,dfcn,dfco,dfbc,dfe_f,dfcn_f,dfco_f,dfbc_f
def main():
    plt_ts_cn10_zoom(saveorshowplot)
    subplt_wd('cn10',False,saveorshowplot)
    subplt_wd('CO',False,saveorshowplot)
    subplt_wd('ccn_0.5504',False,saveorshowplot)
    
    boxplot_hist_madarrays('cn10',saveorshowplot)
    boxplot_hist_madarrays('co',saveorshowplot)
    plt_ts_all6subplots(saveorshowplot)
    ts_all3subplots(False,saveorshowplot)
    ts_all3subplots(True,saveorshowplot)
    ts_all3subplots_zoomed(False,saveorshowplot)
    ts_all3subplots_zoomed(True,saveorshowplot)
#    plt_ts_cn10('show')
#    boxplot_madarrays(saveorshowplot)
    plt_compare_wdws(False,saveorshowplot)
    plt_compare_wdws(True,saveorshowplot)
    
    
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

def plt_ts_all6subplots(saveorshowplot):
    # Create subset of data
    startdate = '2016-05-28 00:00:00'
    enddate = '2016-06-02 00:00:01'
    dfe1 = dfe[startdate:enddate]
    
    
    f, ((ax1, ax1z), (ax2, ax2z), (ax3,ax3z)) = plt.subplots(3, 2, sharex='col',figsize=(9.5,7))
    
    ### FULL TIME SERIES
    
    #plot cn10
    ax1.plot(dfe['cn10'],'.',markersize=0.5)
    ax1.set_ylim([0,2000])
    ax1.set_ylabel('$CN_{10}$ \n Num. Conc. $(cm^{-3})$')
    
    # plot CO
    ax2.plot(dfe['CO'],'.',markersize=0.5)
    ax2.set_ylabel('$CO$ \n Mixing Ratio $(ppb)$')
    ax2.set_ylim([47.5,62.5])
    
    # plot BC
    ax3.plot(dfe['BC'],'.',markersize=1)
    ax3.set_ylim([0,0.5])
    ax3.set_ylabel('$BC$ \n Conc. $(ng.m^{-3})$')
    
    # Format the x axis
    days = mdates.DayLocator()
    weeks = mdates.WeekdayLocator(mdates.SUNDAY)
    
    ax3.xaxis.set_minor_locator(days)
    ax3.xaxis.set_major_locator(weeks)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    
    
    ### SUBSET (zoomed)
    
    #plot cn10
    ax1z.plot(dfe1['cn10'],'.',markersize=2)
    ax1z.set_ylim([0,2000])
    
    # plot CO
    ax2z.plot(dfe1['CO'],'.',markersize=2)
    ax2z.set_ylim([52,63])
    ax2z.yaxis.set_major_locator(mticker.MultipleLocator(1))
    
    # plot BC
    ax3z.plot(dfe1['BC'],'.',markersize=2)
    ax3z.set_ylim([0,0.5])
    
    # Format the x axis
    hours = mdates.HourLocator([6,12,18])
    days = mdates.DayLocator()
    
    ax3z.xaxis.set_minor_locator(hours)
    ax3z.xaxis.set_major_locator(days)
    ax3z.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    
    f.autofmt_xdate()
    
    # Show or save
    handle_plt(plt,saveorshowplot,plt_path,
               outputfilename='ts_all6subplots')
    return

def ts_all3subplots(logscale=True,saveorshowplot='save'):
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col',figsize=(9.5,7))

    ### FULL TIME SERIES

    #plot cn10
    ax1.plot(dfe['cn10'],'.',markersize=0.5)
    ax1.set_ylabel('$CN_{10}$ \n Num. Conc. ($cm^{-3}$)')

    # plot CO
    ax2.plot(dfe['CO'],'.',markersize=0.5)
    ax2.set_ylabel('$CO$ \n Mixing Ratio ($ppb$)')
    

    # plot BC
    ax3.plot(dfe['BC'],'.',markersize=1)
    ax3.set_ylabel('$BC$ \n Conc. ($ng.m^{-3}$)')

    # Format the x axis
    days = mdates.DayLocator()
    weeks = mdates.WeekdayLocator(mdates.SUNDAY)

    ax3.xaxis.set_minor_locator(days)
    ax3.xaxis.set_major_locator(weeks)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    
    f.autofmt_xdate()

    if logscale:
        ax1.set_yscale('log')
        ax1.set_ylim([10,10**6])
        ax2.set_yscale('log')
        ax2.set_ylim([40,200])
        ax3.set_yscale('log')
        ax3.set_ylim([0.01,30])
        label = 'logscale'
    else:
        ax1.set_ylim([0,2000])
        ax2.set_ylim([47.5,62.5])
        ax3.set_ylim([0,0.5])
        label = 'linscale'
        
    
    # Show or save
    handle_plt(plt,saveorshowplot,plt_path,
               outputfilename='ts_all3subplots_'+label)
    return

def ts_all3subplots_zoomed(logscale=False,saveorshowplot='save'):
    # Create subset of data
    startdate = '2016-05-28 00:00:00'
    enddate = '2016-06-02 00:00:01'
    dfe1 = dfe[startdate:enddate]
    
    
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col',figsize=(9.5,7))
    
    ### SUBSET (zoomed)
    
    #plot cn10
    ax1.plot(dfe1['cn10'],'.',markersize=2)
    ax1.set_ylabel('$CN_{10}$ \n Num. Conc. $(cm^{-3})$')
    
    # plot CO
    ax2.plot(dfe1['CO'],'.',markersize=2)
    ax2.set_ylabel('$CO$ \n Mixing Ratio $(ppb)$')
    
    # plot BC
    ax3.plot(dfe1['BC'],'.',markersize=2)
    ax3.set_ylabel('$BC$ \n Conc. $(ng.m^{-3})$')
    
    # Format the x axis
    hours = mdates.HourLocator([6,12,18])
    days = mdates.DayLocator()
    
    ax3.xaxis.set_minor_locator(hours)
    ax3.xaxis.set_major_locator(days)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    
    f.autofmt_xdate()
    
    if logscale:
        ax1.set_yscale('log')
        ax1.set_ylim([10,10**6])
        ax2.set_yscale('log')
        ax2.set_ylim([45,300])
        ax2.yaxis.set_major_locator(mticker.FixedLocator([50,70,100,200]))
        ax2.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
        ax3.set_yscale('log')
        ax3.set_ylim([0.01,30])
        ax3.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
        label = 'logscale'
    else:
        ax1.set_ylim([0,2000])
        ax2.set_ylim([52,63])
        ax2.yaxis.set_major_locator(mticker.MultipleLocator(2))
        ax3.set_ylim([0,0.31])
        ax3.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        label = 'linscale'
    
    # Show or save
    handle_plt(plt,saveorshowplot,plt_path,
               outputfilename='ts_all3subplots_zoomed_'+label)
    return


def boxplot_hist_madarrays(data = 'cn10', saveorshowplot='save'):
    # https://matplotlib.org/examples/pylab_examples/broken_axis.html
    os.chdir(exhaust_path)
    if data.lower() in ['cn','cn10']:
        df = pd.read_hdf('mad_array_cn10.h5',key='mad')
        df = df.dropna()
        ylabel = 'Number Concentration ($cm^{-3}$)'
        ylim_l = [0,25]
        ylim_u = [25, 3*10**5]
        xtick_label = '$CN_{10}$'
        log_bins = np.logspace(0, 6, 1000, endpoint=True)
    elif data.lower() == 'co':
        df = pd.read_hdf('mad_array_co.h5',key='mad')
        df = df.dropna()
        ylabel = 'Mixing ratio ($ppb$)'
        ylim_l = [0,0.35]
        ylim_u = [0.35, 30]
        xtick_label = '$CO$'
        log_bins = np.logspace(-2, 2, 1000, endpoint=True)
        
    

    f, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharex='col',figsize=(6,7))
    
    # Fix figure so that the axis changes from linear to log scale
    # Hide part of the y axes between the two plots
    d = 0.005 #diagonal size
    f.subplots_adjust(hspace=0.0)
    L1 = lines.Line2D([0.125, 0.125], [0.48+2*d, 0.52-2*d], lw = 10,
                     transform=f.transFigure, figure=f, color='white')
    L2 = lines.Line2D([0.477, 0.477], [0.48+2*d, 0.52-2*d], lw = 10,
                     transform=f.transFigure, figure=f, color='white')
    L3 = lines.Line2D([0.548, 0.548], [0.48+2*d, 0.52-2*d], #lw = 10,
                     transform=f.transFigure, figure=f, color='white')
    L4 = lines.Line2D([0.900, 0.900], [0.48+2*d, 0.52-2*d], lw = 10,
                     transform=f.transFigure, figure=f, color='white')
    
    # Add diagonal lines showing the break
    
    col = 'k'
    D1L = lines.Line2D([0.125-d, 0.125+d], [0.52-d, 0.52+d], 
                     transform=f.transFigure, figure=f, color=col)
    D1R = lines.Line2D([0.477-d, 0.477+d], [0.52-d, 0.52+d], 
                     transform=f.transFigure, figure=f, color=col)
    D2L = lines.Line2D([0.125-d, 0.125+d], [0.48-d, 0.48+d], 
                     transform=f.transFigure, figure=f, color=col)
    D2R = lines.Line2D([0.477-d, 0.477+d], [0.48-d, 0.48+d], 
                     transform=f.transFigure, figure=f, color=col)
    D3L = lines.Line2D([0.548-d, 0.548+d], [0.52-d, 0.52+d], 
                     transform=f.transFigure, figure=f, color=col)
    D3R = lines.Line2D([0.900-d, 0.900+d], [0.52-d, 0.52+d], 
                     transform=f.transFigure, figure=f, color=col)
    D4L = lines.Line2D([0.548-d, 0.548+d], [0.48-d, 0.48+d], 
                     transform=f.transFigure, figure=f, color=col)
    D4R = lines.Line2D([0.900-d, 0.900+d], [0.48-d, 0.48+d], 
                     transform=f.transFigure, figure=f, color=col) 
    f.lines.extend([L1,L2,L4,
                    D1L,D1R,D2L,D2R,
                    D3L,D3R,D4L,D4R])
    
    # hide the x axis spines between ax1 and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop='off')  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    
    # hide the x axis spines between ax3 and ax4
    ax3.spines['bottom'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax3.xaxis.tick_top()
    ax3.tick_params(labeltop='off')  # don't put tick labels at the top
    ax4.xaxis.tick_bottom()
    
    
    
    ###  Box and whisker plot
    
    plt.suptitle('Rolling Median Absolute Deviation \n' + xtick_label)
    f.text(0.02,0.5,ylabel,va='center',rotation='vertical')
    f.text(0.93,0.5,ylabel,va='center',rotation='vertical')
    
    ax1.boxplot(df,0,'xb', widths=(0.5))
    ax2.boxplot(df,0,'xb', widths=(0.5))
    
    ax1.set_yscale('log')
    ax1.set_ylim(ylim_u)
    ax2.set_ylim(ylim_l)
    
    plt.setp(ax2,xticklabels=[xtick_label])
    
    
    
    ### Histogram
    
    ax3.hist(df, bins=log_bins, orientation=u'horizontal',color='black')
    ax4.hist(df, bins=log_bins, orientation=u'horizontal',color='black')
    
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_ylim(ylim_u)
    
    ax4.set_xscale('log')
    ax4.set_ylim(ylim_l)
    
    ax3.tick_params(labelleft='off')
    ax4.tick_params(labelleft='off')
    
    ax4.set_xlabel('Frequency')
    
    
    # Show or save
    handle_plt(plt,saveorshowplot,plt_path,
               outputfilename='boxplot_madarray_' + data.lower())
    
    return

def boxplot_madarrays(saveorshowplot='save'):
    # https://matplotlib.org/examples/pylab_examples/broken_axis.html
    os.chdir(master_path)
    mcn = pd.read_hdf('mad_array_cn10.h5',key='mad')
    mco = pd.read_hdf('mad_array_co.h5',key='mad')
    

    f, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharex='col')
    
    f.subplots_adjust(hspace=0.)
    
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


def subplt_wd(column='cn10',logscale = False, saveorshowplot='save'):
    
    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6,1,figsize=(8,11))
        
    ax1.plot(dfe['WindDirRel_vmean'],dfe[column],'.k', markersize=2)
    ax1.set_title('Raw ' + column + ' data')
    ax1.set_xlim([0,360])
    ax1.xaxis.set_visible(False)
    
    ax2.plot(dfbc_f['WindDirRel_vmean'],dfbc_f[column],'.k', markersize=2)
    ax2.set_title('BC only filter')
    ax2.set_xlim([0,360])
    ax2.xaxis.set_visible(False)
    
    ax3.plot(dfco_f['WindDirRel_vmean'],dfco_f[column],'.k', markersize=2)
    ax3.set_title('CO only filter')
    ax3.set_xlim([0,360])
    ax3.xaxis.set_visible(False)
    
    ax4.plot(dfcn_f['WindDirRel_vmean'],dfcn_f[column],'.k', markersize=2)
    ax4.set_title('CN only filter')
    ax4.set_xlim([0,360])
    ax4.xaxis.set_visible(False)
    
    ax5.plot(dfcomb_f['WindDirRel_vmean'],dfcomb_f[column],'.k', markersize=2)
    ax5.set_title('CO + BC + CN (no window) filter')
    ax5.set_xlim([0,360])
    ax5.xaxis.set_visible(False)
    
    ax6.plot(dfe_f['WindDirRel_vmean'],dfe_f[column],'.k', markersize=2)
    ax6.set_title('Full filter')
    ax6.set_xlim([0,360])
    ax6.xaxis.set_visible(True)
    ax6.set_xlabel('Relative Wind Direction ($^o$)')
    
    if column.lower() == 'co':
        units = 'Mixing Ratio (ppb)'
    else:
        units = 'Number Concentration ($cm^{-3}$)'
    
    f.text(0.02, 0.5,units,va='center', rotation='vertical')
    
    if 'ccn' in column:
        for ax in [ax1,ax2,ax3,ax4,ax5]:
            ax.set_ylim([0,6000])
            ax.yaxis.set_major_locator(mticker.FixedLocator([0, 2000, 4000, 6000]))
    
    if logscale:
        for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
            ax.set_yscale('log')
    
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

def plt_ts_cn10_zoom(saveorshowplot='save'):
    startdate = '2016-05-29 00:00:00'
    enddate = '2016-06-02 00:00:01'
    dfe1 = dfe[startdate:enddate]
    dfe_f1 = dfe_f[startdate:enddate]
    
    plt.figure(figsize=(11,8))
    
    plt.plot(dfe1['cn10'],'.b',alpha=0.2)
    plt.plot(dfe_f1['cn10'],'.r')
    
    plt.ylabel('Number Conc ($cm^{-3}$)')
    plt.ylim([0,2000])
    
    #plt.title('cn10 timeseries - full filt with cn10 filter parameters')
    
    handle_plt(plt,saveorshowplot,plt_path,outputfilename='ts_cn10_full_filt_zoom')
    return

def plt_ts_ccn_zoom(saveorshowplot='save'):
    startdate = '2016-05-28 00:00:00'
    enddate = '2016-06-02 00:00:01'
    dfe1 = dfe[startdate:enddate]
    dfe_f1 = dfe_f[startdate:enddate]
    
    plt.figure(figsize=(11,8))
    
    plt.plot(dfe1['ccn_0.5504'],'.b',alpha=0.2)
    plt.plot(dfe_f1['ccn_0.5504'],'.r')
    
    plt.ylabel('Number Conc ($cm^{-3}$)')
    plt.ylim([0,1000])
    
    #plt.title('cn10 timeseries - full filt with cn10 filter parameters')
    
    handle_plt(plt,saveorshowplot,plt_path,outputfilename='ts_cn10_full_filt_zoom')
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
    
    handle_plt(plt,saveorshowplot,plt_path,
               outputfilename='wd_cn10_wdws_filter')
    return

def plt_compare_wdws(logscale = False, saveorshowplot='save'):
    dfwdws = dfe[[False if (((wd>90) and (wd<270)) or ws<10) else
                  True for wd,ws in 
                  zip(dfe['WindDirRel_vmean'],dfe['WindSpdRel_port'])]]
    
    
    f, ax = plt.subplots(1, 1)
    
    ax.plot(dfe['WindDirRel_vmean'],dfe['cn10'],'.',             
             dfwdws['WindDirRel_vmean'],dfwdws['cn10'],'.r')
    
    ax.set_xlabel('Relative Wind Direction ($^o$)')
    ax.set_ylabel('Num Conc ($cm^{-3}$)')
    
    ax.legend(['raw','wind filter'])
    
    ax.set_xlim([0,360])
    
    if logscale:
        ax.set_yscale('log')
        ax.set_ylim([5,10**7])
        label = 'log'
    else:
        label = 'lin'
        
        
    handle_plt(plt,saveorshowplot,plt_path,
               outputfilename='wd_cn10_wdws_filter_compare_'+label)
    return

def load():
    
    startdate = '2016-04-25 00:00:00'
    enddate = '2016-06-09 00:00:00'
    
    os.chdir(exhaust_path)
    

            
    print('Loading data from file - this may take a few seconds. Please wait...')
    
    if os.path.isfile('in2016_v03_dfe_publication.h5'):
        ext = '_publication'
    else:
        ext = ''
    
    dfe = pd.read_hdf('in2016_v03_dfe'+ext+'.h5',key='data')[startdate:enddate]
    dfcn = pd.read_hdf('in2016_v03_dfcn'+ext+'.h5',key='data')[startdate:enddate]
    dfco = pd.read_hdf('in2016_v03_dfco'+ext+'.h5',key='data')[startdate:enddate]
    dfbc = pd.read_hdf('in2016_v03_dfbc'+ext+'.h5',key='data')[startdate:enddate]
    
    if 'ccn_0.5504' not in dfe.columns:
        ccn = pd.read_hdf('ccn_1s_in2016_v03.h5',key='ccn')
        ccn = ccn.shift(10,freq='H') # Correct a time offset
        
        
        dfe = merge_new_ccn(dfe,ccn)
        dfcn = merge_new_ccn(dfcn,ccn)
        dfco = merge_new_ccn(dfco,ccn)
        dfbc = merge_new_ccn(dfbc,ccn)
        
        dfe.to_hdf('in2016_v03_dfe_publication.h5',key='data')
        dfcn.to_hdf('in2016_v03_dfcn_publication.h5',key='data')
        dfco.to_hdf('in2016_v03_dfco_publication.h5',key='data')
        dfbc.to_hdf('in2016_v03_dfbc_publication.h5',key='data')
    
    
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


def merge_new_ccn(d,ccn):
    s = d.index[0]
    e = d.index[-1]
    
    for col in ccn.columns:
        if col in d.columns:
            d = d.drop(col,axis=1)
    if 'ccn' in d.columns:
        d = d.drop('ccn',axis=1)
    
    d = pd.concat([d,ccn],axis=1,join_axes=[d.index])
    d = d[s:e]
    return d

if os.path.isdir(master_path_lcl):
    master_path = master_path_lcl
    exhaust_path = exhaust_path_lcl
    plt_path = plt_path_lcl
elif os.path.isdir(master_path_svr):
    master_path = master_path_svr 
    exhaust_path = exhaust_path_svr
    plt_path = plt_path_svr 

else:
    assert False, "Can't find data! Please check path"

if load_all_data:
    dfe,dfcn,dfco,dfbc,dfcomb,dfe_f,dfcn_f,dfco_f,dfbc_f,dfcomb_f = load()
elif load_partial:
    startdate = '2016-04-25 00:00:00'
    enddate = '2016-06-09 00:00:00'
    
    os.chdir(exhaust_path)
    
    print('Loading data from file - this may take a few seconds. Please wait...')
    
    dfe = pd.read_hdf('in2016_v03_dfe.h5',key='data')[startdate:enddate]
    ex = dfe['exhaust']    
    dfe_f = dfe.loc[~ex]

# if this script is run at the command line, run the main script   
if __name__ == '__main__': 
	main()
