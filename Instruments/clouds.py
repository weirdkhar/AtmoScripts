# -*- coding: utf-8 -*-
"""
A bunch of functions related to processing and plotting cloud data
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.dates as mdates
import pandas as pd
import re
import os
import atmosplots

from matplotlib import mlab

from mpl_toolkits.axes_grid1 import make_axes_locatable

#def main():
#    
#    import pandas as pd
#    smpsgrimm_path = 'c:/Dropbox/RuhiFiles/Research/Projects/2016_V02_CAPRICORN/Data_Raw/SMPS_GRIMM'
#    os.chdir(smpsgrimm_path)
#    s_raw = pd.read_hdf('SMPS_Grimm_raw.h5',key='smps')
#    
#    s_raw_subset = s_raw['2016-03-23 04:00:00' : '2016-03-29 12:00:00']
#    plot(smps = s_raw_subset.iloc[:,4:136], aps = s_raw_subset.iloc[:,4:136], nano = s_raw_subset.iloc[:,4:136])
#
#    #-- Generate some data
#    
#    #plot(aps, smps, nano, time)
#    plt.show()




def plot_basta(
        x_axis = None,
        y_axis = None,
        z_matrix = None,
        figsize = (12,4),
        common_xlim = None,
        zmin1 = -70,
        zmax1 = 20,
        fig = None,
        ax = None,
        #logscale_z1 = True,
        yticklocation1 = None,
        title1 = 'BASTA RV Investigator',
        MajorTitle = '',
        x_label = '',
        y_label = 'Height (km)',
        z_label = 'Z (dBZ)',
        SaveOrShowPlot = 'show',
        outputfilename = 'plot.pdf',
        outputpath = None
        ):
    # Ignore warning in code z[z < zmin] = zmin
    import warnings
    warnings.simplefilter(action = "ignore", category = RuntimeWarning)
    
    
#    # Determine what needs to be plotted:    
#    datasets_available = [aps is not None, smps is not None, nano is not None]
#    num_plots = sum(datasets_available)
#    which_datasets = [i for i, x in enumerate(datasets_available) if x] #outputs the index of the true values in the list - http://stackoverflow.com/questions/21448225/getting-indices-of-true-values-in-a-boolean-list
#    if num_plots == 1:
#        plt_ax2 = False
#        plt_ax3 = False
#        data1, sizeBins1, ylabel1, title1, = determine_data(which_datasets,aps,smps,nano,0)
#    elif num_plots  == 2:
#        plt_ax2 = True
#        plt_ax3 = False
#        data1, sizeBins1, ylabel1, title1 = determine_data(which_datasets,aps,smps,nano,0)
#        data2, sizeBins2, ylabel2, title2 = determine_data(which_datasets,aps,smps,nano,1)
#    elif num_plots  == 3:
#        plt_ax2 = True
#        plt_ax3 = True
#        data1, sizeBins1, ylabel1, title1 = determine_data(which_datasets,aps,smps,nano,0)
#        data2, sizeBins2, ylabel2, title2 = determine_data(which_datasets,aps,smps,nano,1)
#        data3, sizeBins3, ylabel3, title3 = determine_data(which_datasets,aps,smps,nano,2)
    
    #Determine common_xlim if not given
    if common_xlim is None:
        common_xlim = [x_axis[0], x_axis[-1]]
        
    # Plot
    if fig is None:
        fig = plt.figure(figsize=(figsize[0],figsize[1]))
         
    #-- Panel 1
    ax = plot_axes(x_axis, y_axis, z_matrix, y_label,title1, common_xlim, fig, ax, zmin1, zmax1, yticklocation1,z_label)
    ax.set_title(MajorTitle)    


    plt.setp(ax.get_xticklabels(), rotation=-20, horizontalalignment='left')

    # If less than 1 day of data is being plotted, put a day label on the plot so that xlabels can be times.
    if (common_xlim[1]-common_xlim[0]).days < 1:   
        daylabel = (common_xlim[0]).strftime("%b %d")
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8)
        ax.annotate(daylabel, xy=(0.9, 0.9), xycoords='axes fraction', bbox=bbox_props)
        
    # Set the space between subplots
    plt.subplots_adjust(hspace=0.1)
        
    atmosplots.saveorshowplot(plt,SaveOrShowPlot, outputpath, outputfilename)
    
    return
    
    
    
def plot_axes(x_axis, y_axis, z_matrix, ylabel, title, common_xlim, fig, ax, zmin, zmax, yticklocation, z_label):
    
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8)  
    
#    z_matrix = project_to_continuous_timestep(z_matrix) #Ensure data is projected onto a continuous timestep so that the image doesn't fill in data
    if ax is None:    
        ax = fig.add_subplot(1,1,1)
    im, cbar = specgram_smps(x_axis, y_axis, z_matrix, ax, fig, common_xlim, zmin = zmin, zmax = zmax, yticklocation = yticklocation, z_label = z_label)
    ax.set_ylabel(ylabel)
    ax.annotate(title, xy=(0.025, 0.9), xycoords='axes fraction', bbox=bbox_props)

    return ax
    
    
def specgram_smps(x_axis, y_axis, z_matrix, ax, fig, common_xlim, zmin, zmax, yticklocation, z_label):
    ''' Make and plot a log-scaled spectrogram suitable for SMPS matrix data '''
    
    
    
    #Setup colouring
    #mymap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['blue','red'])
    mymap = plt.cm.jet
    mymap.set_bad(color='w')
    
    
    # Format data appropriately
    x_time = mdates.datestr2num(x_axis.strftime('%Y-%m-%d %H:%M:%S')) # Convert from datetime64 to string to datenum
    y_lims = y_axis
    X, Y = np.meshgrid(x_time,y_lims,copy=False,indexing='xy')
    
    
    Z = z_matrix.transpose()
    Z[Z == -999] = np.nan
    #z[z < zmin] = zmin #Replace values below a threshold so that log scale works
    Z = np.ma.array(Z, mask=np.isnan(Z)) # Mask nan's   
    
    
    #Plot
    norm_scale = mpl.colors.Normalize(vmin=zmin, vmax=zmax)
    
    
    im = ax.pcolormesh(X,Y,Z, cmap = mymap, norm=norm_scale)

    # Format y-axis
    #ax.set_yscale('log')
    ax.set_ylim([np.floor(y_axis.min()), np.ceil(y_axis.max())])
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter()) 


    #Format x-axis as a date
    plotrange = (common_xlim[1]-common_xlim[0]).days
    if plotrange > 5:
        xlabel_format = '%b %d'
    elif plotrange > 1:
        xlabel_format = '%b %d %H:%M'
    else:
        xlabel_format = '%H:%M'
        
        
    ax.set_xlim([x_time.min(), x_time.max()])
    ax.xaxis_date()
    date_format = mdates.DateFormatter(xlabel_format)
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate() # This simply sets the x-axis data to diagonal so it fits better.
    

    # Add the colorbar in a seperate axis
    cax = make_legend_axes(ax)
    cbar = fig.colorbar(im, cax=cax, format=r'%d')
    cbar.set_label(z_label, rotation=90)

    return im, cbar

#def extract_sizeBins(df):
#    
#    #Extract size bin information from dataframe
#    sizeBins = df.columns.tolist()
#    sizeBins = np.asarray([float(re.findall('\d+\.\d+',i)[0]) for i in sizeBins]) # extract float from string
#    
#    return sizeBins
#
#def determine_data(which_datasets,aps,smps,nano, num):
#    if which_datasets[num] == 0:
#        data = aps
#        ylabel = 'Diameter ($\mu$m)'
#        title = 'APS'
#        
#    elif which_datasets[num] == 1:
#        data = smps
#        ylabel = 'Diameter (nm)'
#        title = 'SMPS'
#        
#    elif which_datasets[num] == 2:
#        data = nano
#        ylabel = 'Diameter (nm)'
#        title = 'Nano SMPS'
#
#    sizeBins = extract_sizeBins(data)
#    
#    # Check what form the data is in, raw, with all the parameter, or the cut down version
#    if 'Liq' in data: # Raw form
#        d_plt = data.iloc[:,4:136]
#    else:
#        # another form, need to figure out
#        d_plt = data
#
#    return d_plt, sizeBins, ylabel, title

def project_to_continuous_timestep(data):
    # Calculate the timestep, taking into account that the gap in data could be at the very beginning
    time_diff0 = (data.index[1]-data.index[0]).total_seconds() * 10**6 #Determine the time difference in microseconds
    time_diff1 = (data.index[2]-data.index[1]).total_seconds() * 10**6 #Determine the time difference in microseconds
    if time_diff0 != time_diff1:
        time_diff = min([time_diff0,time_diff1])
    else:
        time_diff = time_diff0
        
    # Create new time index    
    index = pd.date_range(start = data.index[0], end = data.index[-1], freq="%sS" % int(time_diff / 10**6)) # Create Datetimeindex with the given interval
    
    df = pd.DataFrame(index = index, columns = ['removeMe'])

    df = df.join(data,how='outer')

    del df['removeMe']
    
    return df

def make_legend_axes(ax, leftorright = 'right'):
    divider = make_axes_locatable(ax)
    legend_ax = divider.append_axes(leftorright, 0.4, pad=0.2)
    return legend_ax



   
