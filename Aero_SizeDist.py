import datetime
import SMPS_Grimm
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




def plot(
        aps=None, 
        smps=None,
        nano=None,
        figsize = (12,4),
        common_xlim = None,
        zmin1 = 1,
        zmax1 = 1000,
        logscale_z1 = True,
        yticklocation1 = None,
        zmin2 = 1,
        zmax2 = 1000,
        logscale_z2 = True,
        yticklocation2 = None,
        zmin3 = 1,
        zmax3 = 1000,
        yticklocation3 = None,        
        logscale_z3 = True,
        SaveOrShowPlot = 'show',
        outputfilename = 'plot.pdf',
        outputpath = None
        ):
    # Ignore warning in code z[z < zmin] = zmin
    import warnings
    warnings.simplefilter(action = "ignore", category = RuntimeWarning)
    
    
    # Determine what needs to be plotted:    
    datasets_available = [aps is not None, smps is not None, nano is not None]
    num_plots = sum(datasets_available)
    which_datasets = [i for i, x in enumerate(datasets_available) if x] #outputs the index of the true values in the list - http://stackoverflow.com/questions/21448225/getting-indices-of-true-values-in-a-boolean-list
    if num_plots == 1:
        plt_ax2 = False
        plt_ax3 = False
        data1, sizeBins1, ylabel1, title1, = determine_data(which_datasets,aps,smps,nano,0)
    elif num_plots  == 2:
        plt_ax2 = True
        plt_ax3 = False
        data1, sizeBins1, ylabel1, title1 = determine_data(which_datasets,aps,smps,nano,0)
        data2, sizeBins2, ylabel2, title2 = determine_data(which_datasets,aps,smps,nano,1)
    elif num_plots  == 3:
        plt_ax2 = True
        plt_ax3 = True
        data1, sizeBins1, ylabel1, title1 = determine_data(which_datasets,aps,smps,nano,0)
        data2, sizeBins2, ylabel2, title2 = determine_data(which_datasets,aps,smps,nano,1)
        data3, sizeBins3, ylabel3, title3 = determine_data(which_datasets,aps,smps,nano,2)
    
    #Determine common_xlim if not given
    if common_xlim is None:
        common_xlim = [data1.index[0], data1.index[-1]]
        
    # Plot
    fig = plt.figure(figsize=(figsize[0],figsize[1]*num_plots))
          
    #-- Panel 1
    ax1 = plot_axes(1, data1,sizeBins1,ylabel1,title1, common_xlim, fig,num_plots, zmin1, zmax1, yticklocation1, logscale_z1)
    ax1.set_title('Aerosol Size Distribution')    
    #-- Panel 2
    if plt_ax2:
        ax2 = plot_axes(2, data2,sizeBins2,ylabel2,title2, common_xlim, fig,num_plots, zmin2, zmax2, yticklocation2, logscale_z1)
    #-- Panel 3
    if plt_ax3:
        ax3 = plot_axes(3, data3,sizeBins3,ylabel3,title3, common_xlim, fig,num_plots, zmin3, zmax3, yticklocation3, logscale_z1)
    

    # Set the labels on the bottom axis to be rotated at 20 deg and aligned left to use less space
    if plt_ax3:    
        plt.setp(ax3.get_xticklabels(), rotation=-20, horizontalalignment='left')
        ax2.get_xaxis().set_visible(False)
        ax1.get_xaxis().set_visible(False)
        
    elif plt_ax2:
        plt.setp(ax2.get_xticklabels(), rotation=-20, horizontalalignment='left')
        ax1.get_xaxis().set_visible(False)
    else:
        plt.setp(ax1.get_xticklabels(), rotation=-20, horizontalalignment='left')

    # If less than 1 day of data is being plotted, put a day label on the plot so that xlabels can be times.
    if (common_xlim[1]-common_xlim[0]).days < 1:   
        daylabel = (common_xlim[0]).strftime("%b %d")
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8)
        ax1.annotate(daylabel, xy=(0.9, 0.9), xycoords='axes fraction', bbox=bbox_props)
        
    # Set the space between subplots
    plt.subplots_adjust(hspace=0.1)
        
    atmosplots.saveorshowplot(plt,SaveOrShowPlot, outputpath, outputfilename)
    
    return
    
    
    
def plot_axes(axesnum, data, sizeBins, ylabel, title, common_xlim, fig, num_plots, zmin, zmax, yticklocation, logscale_z):
    
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8)  
    
    data = project_to_continuous_timestep(data) #Ensure data is projected onto a continuous timestep so that the image doesn't fill in data
    ax = fig.add_subplot(num_plots,1,axesnum)
    im, cbar = specgram_smps(data.index, sizeBins, data, ax, fig, common_xlim, zmin = zmin, zmax = zmax, yticklocation = yticklocation,logscale_z = logscale_z)
    ax.set_ylabel(ylabel)
    ax.annotate(title, xy=(0.025, 0.9), xycoords='axes fraction', bbox=bbox_props)

    return ax
    
    
def specgram_smps(time, sizeBins, data, ax, fig, common_xlim, zmin, zmax, yticklocation, logscale_z):
    ''' Make and plot a log-scaled spectrogram suitable for SMPS matrix data '''
    
    
    
    #Setup colouring
    #mymap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['blue','red'])
    mymap = plt.cm.jet
    mymap.set_bad(color='w')
    
    
    # Format data appropriately
    x_time = mdates.datestr2num(time.strftime('%Y-%m-%d %H:%M:%S')) # Convert from datetime64 to string to datenum
    y_lims = sizeBins   
    X, Y = np.meshgrid(x_time,y_lims,copy=False,indexing='xy')
    
    
    z = data.values.transpose()
    z[z < zmin] = zmin #Replace values below a threshold so that log scale works
    Z = np.ma.array(z, mask=np.isnan(z)) # Mask nan's   
    
    
    #Plot
    if logscale_z:
        norm_scale = mpl.colors.LogNorm(vmin=zmin, vmax=zmax)
    else:
        norm_scale = mpl.colors.Normalize(vmin=zmin, vmax=zmax)
    
    
    im = ax.pcolormesh(X,Y,Z, cmap = mymap, norm=norm_scale)

    # Format y-axis
    ax.set_yscale('log')
    ax.set_ylim([sizeBins.min(), sizeBins.max()])
    if yticklocation is None:    
        if sizeBins.min() < 10:
            if sizeBins.max() > 500:
                yticklocation = [sizeBins.min(),10,30,100,300, sizeBins.max()]
            else:
                yticklocation = [sizeBins.min(),10,30,100,sizeBins.max()]
        else:
            if sizeBins.max() > 500:
                yticklocation = [sizeBins.min(),100,300, sizeBins.max()]
            else:
                yticklocation = [sizeBins.min(),100,sizeBins.max()]
    ax.set_yticks(yticklocation)
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
    cbar.set_label('dN/d(log$_{10}$d$_0$) ($cm^{-3}$)', rotation=90)

    return im, cbar

def extract_sizeBins(df):
    
    #Extract size bin information from dataframe
    sizeBins = df.columns.tolist()
    sizeBins = np.asarray([float(re.findall('\d+\.\d+',i)[0]) for i in sizeBins]) # extract float from string
    
    return sizeBins

def determine_data(which_datasets,aps,smps,nano, num):
    if which_datasets[num] == 0:
        data = aps
        ylabel = 'Diameter ($\mu$m)'
        title = 'APS'
        
    elif which_datasets[num] == 1:
        data = smps
        ylabel = 'Diameter (nm)'
        title = 'SMPS'
        
    elif which_datasets[num] == 2:
        data = nano
        ylabel = 'Diameter (nm)'
        title = 'Nano SMPS'

    sizeBins = extract_sizeBins(data)
    
    # Check what form the data is in, raw, with all the parameter, or the cut down version
    if 'Liq' in data: # Raw form
        d_plt = data.iloc[:,4:136]
    else:
        # another form, need to figure out
        d_plt = data

    return d_plt, sizeBins, ylabel, title

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

def make_legend_axes(ax):
    divider = make_axes_locatable(ax)
    legend_ax = divider.append_axes('right', 0.4, pad=0.2)
    return legend_ax



    
#def specgram(x, time, ax, fig):
#    """Make and plot a log-scaled spectrogram"""
#    dt = np.diff(time)[0] # In days...
#    fs = dt * (3600 * 24) # Samples per second
#
#    spec_img, freq, _ = mlab.specgram(x, Fs=fs, noverlap=200)
#    t = np.linspace(time.min(), time.max(), spec_img.shape[1])
#
#    # Log scaling for amplitude values
#    spec_img = np.log10(spec_img)
#
#    # Log scaling for frequency values (y-axis)
#    ax.set_yscale('log')
#
#    # Plot amplitudes
#    im = ax.pcolormesh(t, freq, spec_img)
#
#    # Add the colorbar in a seperate axis
#    cax = make_legend_axes(ax)
#    cbar = fig.colorbar(im, cax=cax, format=r'$10^{%0.1f}$')
#    cbar.set_label('Amplitude', rotation=-90)
#
#    ax.set_ylim([freq[1], freq.max()])
#
#    # Hide x-axis tick labels
#    plt.setp(ax.get_xticklabels(), visible=False)
#
#    return im, cbar

