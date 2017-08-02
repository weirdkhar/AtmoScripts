import sys
sys.path.append('c:\\OneDrive\\RuhiFiles\\Research\\ProgramFiles\\git\\')
sys.path.append('h:\\code\\atmoscripts\\')

import datetime
from Atmoscripts.Instruments import SMPS_Grimm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.dates as mdates
import pandas as pd
import re
import os
import Atmoscripts.atmosplots as atmosplots


from scipy.optimize import curve_fit
from matplotlib import dates
import matplotlib.ticker as tck

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
    
    
    
def plot_axes(axesnum, data, sizeBins, ylabel, title, common_xlim, fig, num_plots, zmin, zmax, yticklocation, logscale_z, ax = None):
    
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8)  
    
    data = project_to_continuous_timestep(data) #Ensure data is projected onto a continuous timestep so that the image doesn't fill in data
    # try to add a subplot if creating a new figure, if not, then use the fig variable which is already a subplot axes
    if ax is None:
        ax = fig.add_subplot(num_plots,1,axesnum)
    #except AttributeError:
    #    ax = fig

    im, cbar = specgram_smps(data.index, np.asarray(sizeBins), data, ax, fig, common_xlim, zmin = zmin, zmax = zmax, yticklocation = yticklocation,logscale_z = logscale_z)
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



###############################################################################
###############################################################################
###############################################################################
### 2017 update of SMPS plotting code
###############################################################################
###############################################################################
###############################################################################

def gaussian(x, a, b, c):
    val = a * np.exp(-(x - b)**2 / c**2)
    return val

def _nan_curve_fit():
    a_opt = np.empty([1,3])[0]
    a_opt[:] = np.NaN
    a_cov = np.empty([3,3])
    a_cov[:] = np.NaN
    return a_opt, a_cov

def mode_max_from_dist_fit(d_mtx, 
                           HR_fit = True,
                           print_fit_params = False, 
                           plot_fit = False):
    '''
    Works through the data matrix and fits lognormal distributions to the 3 
    modes of the size distribution. Returns the maximum of each fitted mode as
    a time series
    '''
    d_mtx = d_mtx.dropna(axis=1, how='any') #Drop any columns where a nan exists so that curve_fit doesn't crash
    
    time = np.array([dates.date2num(d) for d in d_mtx.index]) # Time axis
    size_arr = np.array([float(s) for s in d_mtx.columns.values]) # size axis
    z = d_mtx.as_matrix()
    
    
    mode_max = pd.DataFrame(np.nan, index=time, columns=['m1max','m2max',
                                                         'm3max','m4max'
#                                                         ,'mnmax'
                                                         ])
    for i in np.arange(0,len(time)):
        mode_max.iloc[i] = _mode_max_from_dist_fit_worker(
                                        dist_arr = z[i,:],
                                        size_arr = size_arr,
                                        print_fit_params = print_fit_params,
                                        plot=plot_fit,
                                        HR_fit = HR_fit) 
    
    return mode_max

def _mode_max_from_dist_fit_worker(dist_arr, size_arr, 
                  print_fit_params = False,plot=True, HR_fit = True):
    
    # Define data
    y = dist_arr
    x_data = size_arr
    y0 = dist_arr[0:40]
    y1 = dist_arr[50:85]
    y2 = dist_arr[85::]
    y3 = dist_arr[20:70]
#    yn = dist_arr[0:10] # Nucleation mode
    
    xf = np.arange(-100,len(y),1)
    x0 = np.arange(0,len(y0),1)
    x1 = np.arange(0,len(y1),1)
    x2 = np.arange(0,len(y2),1)
    x3 = np.arange(0,len(y3),1)
#    xn = np.arange(0,len(yn),1)

    

    
    
    #Fit and check for a valid distribution by comparing the fitted mean to input x range
    try:
        popt0, pcov0 = curve_fit(gaussian, x0, y0)
        if (popt0[1] < x0.min()) or (popt0[1] > x0.max()): # Quality check
            popt0, pcov0 = _nan_curve_fit()
    except RuntimeError:
        popt0, pcov0 = _nan_curve_fit()
    
    try:
        popt1, pcov1 = curve_fit(gaussian, x1, y1)
        if (popt1[1] < x1.min()) or (popt1[1] > x1.max()):
            popt1, pcov1 = _nan_curve_fit()
    except RuntimeError:
        popt1, pcov1 = _nan_curve_fit()
    
    try:
        popt2, pcov2 = curve_fit(gaussian, x2, y2)
        if (popt2[1] < x2.min()) or (popt2[1] > x2.max()):
            popt2, pcov2 = _nan_curve_fit()
    except RuntimeError:
        popt2, pcov2 = _nan_curve_fit()
        
    try:
        popt3, pcov3 = curve_fit(gaussian, x3, y3)
        yf3 = gaussian(x3, popt3[0], popt3[1], popt3[2])
        fit3_err = np.sqrt(np.square(yf3/y3.max()-y3/y3.max()).sum())
        if (popt3[1] < x3.min()) or (popt3[1] > x3.max()) or (fit3_err > 0.6) \
            or (not np.isnan(popt0[0])): # Only use if the first mode fit doesn't work
            popt3, pcov3 = _nan_curve_fit()            
    except RuntimeError:
        popt3, pcov3 = _nan_curve_fit()
        
#    try:
#        poptn, pcovn = curve_fit(gaussian, xn, yn)
#        yfn = gaussian(xn, poptn[0], poptn[1], poptn[2])
#        fitn_err = np.sqrt(np.square(yfn/yn.max()-yn/yn.max()).sum())
#        if (poptn[1] < xn.min()) or (poptn[1] > xn.max()):
#            poptn, pcovn = _nan_curve_fit()            
#    except RuntimeError:
#        poptn, pcovn = _nan_curve_fit()

    if print_fit_params:
        # Print results
        print("First mode")
        print("Scale =  %.3f +/- %.3f" % (popt0[0], np.sqrt(pcov0[0, 0])))
        print("Offset = %.3f +/- %.3f" % (popt0[1], np.sqrt(pcov0[1, 1])))
        print("Sigma =  %.3f +/- %.3f" % (popt0[2], np.sqrt(pcov0[2, 2])))

        print("Second mode")
        print("Scale =  %.3f +/- %.3f" % (popt1[0], np.sqrt(pcov1[0, 0])))
        print("Offset = %.3f +/- %.3f" % (popt1[1], np.sqrt(pcov1[1, 1])))
        print("Sigma =  %.3f +/- %.3f" % (popt1[2], np.sqrt(pcov1[2, 2])))

        print("Third mode")
        print("Scale =  %.3f +/- %.3f" % (popt2[0], np.sqrt(pcov2[0, 0])))
        print("Offset = %.3f +/- %.3f" % (popt2[1], np.sqrt(pcov2[1, 1])))
        print("Sigma =  %.3f +/- %.3f" % (popt2[2], np.sqrt(pcov2[2, 2])))

    
    
    
    if HR_fit:
        xf0_HR = np.arange(0,len(y)-0.9,0.1)
        xf1_HR = np.arange(-50,len(y)-50.9,0.1)
        xf2_HR = np.arange(-85,len(y)-85.9,0.1)
        xf3_HR = np.arange(-20,len(y)-20.9,0.1)
#        xfn_HR = np.arange(0,len(y)-0.9,0.1)
        xf_HR_size = np.logspace(np.log10(x_data[0]),np.log10(x_data[-1]),len(xf1_HR))
        
        yf0_HR = gaussian(xf0_HR, popt0[0], popt0[1], popt0[2])
        yf1_HR = gaussian(xf1_HR, popt1[0], popt1[1], popt1[2])
        yf2_HR = gaussian(xf2_HR, popt2[0], popt2[1], popt2[2])
        yf3_HR = gaussian(xf3_HR, popt3[0], popt3[1], popt3[2])
#        yfn_HR = gaussian(xfn_HR, poptn[0], poptn[1], poptn[2])
        
    
        
        # Align distribution
        ds_HR = {
            'yf0_HR':pd.Series(yf0_HR,index=xf_HR_size),
            'yf1_HR':pd.Series(yf1_HR,index=xf_HR_size),
            'yf2_HR':pd.Series(yf2_HR,index=xf_HR_size),
            'yf3_HR':pd.Series(yf3_HR,index=xf_HR_size)
#            'yfn_HR':pd.Series(yfn_HR,index=xf_HR_size)
             }
        df_HR = pd.DataFrame(ds_HR)
        # Calculate sum of all distributions
        df_HR['fit_HR'] = np.nansum(df_HR,axis=1)
        
    
        # Extract the mode size from each mode
        yf0_HR_max = round(df_HR['yf0_HR'].idxmax(skipna=True),1)
        yf1_HR_max = round(df_HR['yf1_HR'].idxmax(skipna=True),1)
        yf2_HR_max = round(df_HR['yf2_HR'].idxmax(skipna=True),1)
        yf3_HR_max = round(df_HR['yf3_HR'].idxmax(skipna=True),1)
#        yfn_HR_max = round(df_HR['yfn_HR'].idxmax(skipna=True),1)
        
        
        if plot:            
            # Plot against diameter
            plt.semilogx(x_data,y,'.b')
            plt.semilogx(xf_HR_size,yf0_HR,'-r')
            plt.semilogx(xf_HR_size,yf1_HR,'-g')
            plt.semilogx(xf_HR_size,yf2_HR,'-m')
            plt.semilogx(xf_HR_size,yf3_HR,'-y')
            plt.semilogx(df_HR['fit_HR'],'-k',lw=2)
            plt.legend(['data','y0','y1','y2','y3','sum'])
            plt.title('HR fit against aerosol diameter')
            plt.show()

            
        return yf0_HR_max, yf1_HR_max, yf2_HR_max, yf3_HR_max#, yfn_HR_max
    else:
        # Calculate model data over full x range
        yf0 = gaussian(xf, popt0[0], popt0[1], popt0[2])
        yf1 = gaussian(xf, popt1[0], popt1[1], popt1[2])
        yf2 = gaussian(xf, popt2[0], popt2[1], popt2[2])
        
    
        # Align distribution
        ds = {
            'yf0':pd.Series(yf0,index=xf),
            'yf1':pd.Series(yf1,index=xf+50),
            'yf2':pd.Series(yf2,index=xf+85)
             }
        dff = pd.DataFrame(ds)
        # Calculate sum of all distributions
        dff['fit'] = np.nansum(dff,axis=1)
        df = dff.loc[0:106].set_index(x_data)
    
        # Extract the mode size from each mode
        yf0_max = round(df['yf0'].idxmax(skipna=True),1)
        yf1_max = round(df['yf1'].idxmax(skipna=True),1)
        yf2_max = round(df['yf2'].idxmax(skipna=True),1)
        
        if plot:
            # Plot data
            plt.semilogx(x_data,y,'.b')
    
            # Plot model
            plt.semilogx(
                    df['yf0'], '--r',
                    df['yf1'],'--g',
                    df['yf2'],'--m')
            plt.semilogx(df['fit'],'-k',lw=2)
    
            plt.xlim([14,700])
    
    
            plt.show()
    
            input("Press Enter to continue...")
        
        return yf0_max, yf1_max, yf2_max
    '''
    # Plot both HR and normal fit together
    plt.semilogx(x_data,y,'.b')
    plt.semilogx(df['fit'],'-k')
    plt.semilogx(df_HR['fit_HR'],'-g')
    plt.legend(['Input data','fit','HR fit'])
    plt.show()
    '''

def calc_time_tick_gap(times,ticks = 5):
    x = (times[-1]-times[0])/ticks
    if x < 1/24*1:
        return 1/24*1
    elif x < 1/24*4.5:
        return 1/24*3
    elif x < 1/24*7.5:
        return 1/24*6
    elif x < 1/24*13.5:
        return 1/24*12
    else:
        return 1
    
def get_time_ticks(times,num_ticks=5):
    return np.arange(np.round(times[0]),times[-1],calc_time_tick_gap(times,num_ticks))

def round_to_1(x):
    return round(x, -int(np.floor(np.log10(abs(x)))))

def plot_smps(d_mtx, 
              fit_lognormal_modes = True,
              fig=None, ax=None, 
              zlim = [10**1, 10**6],
              logscale_z = True,
              title='Size distribution',
              xlabel = 'Timestamp',
              ylabel = 'Mobility diameter (nm)',
              zlabel = 'dN/d(log$_{10}$d$_0$) ($cm^{-3}$)',
              saveorshowplot = 'show',
              output_path = None,
              output_filename = 'SMPS.png'):
    #https://matplotlib.org/examples/images_contours_and_fields/pcolormesh_levels.html
    # Setup data input
    x0 = np.array([dates.date2num(d) for d in d_mtx.index]) # Time axis
    y0 = np.array([float(s) for s in d_mtx.columns.values]) # size axis
    x, y = np.meshgrid(x0,y0)
    z = d_mtx.as_matrix().transpose()

    z[z<zlim[0]] = zlim[0] # mask bad values so that there are no holes in the data

    

    
    # Plot contour
    cmap = plt.get_cmap('jet')
    if logscale_z:
        tick_loc = tck.LogLocator()
        cont_levels = np.logspace(np.log10(zlim[0]),np.log10(zlim[1]),100)
        z_loc_arr = np.logspace(np.log10(zlim[0]),np.log10(zlim[1]),6)
    else:
        tick_loc = tck.LinearLocator()
        cont_levels = np.linspace(zlim[0],zlim[1],100)
        z_loc_arr = np.array([round_to_1(x) for x in np.linspace(zlim[0],zlim[1],6)])
    
    
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(nrows=1, figsize=(15,5))
    elif (fig is None) or (ax is None):
        assert True, \
            'You must pass both a figure and axis object in to the \
            size distribution plotting routine'

    cf = ax.contourf(x,
                      np.log(y), 
                      z,
                      levels = cont_levels,
                      locator=tick_loc,
                      cmap=cmap,
                      interpolation=None)

    ax.set_title(title)

    # Format y axis labels
    y_loc_arr = np.logspace(np.log10(y0[0]),np.log10(y0[-1]),7)
    rounding = [round_to_1(y) for y in y_loc_arr[1:-1]]
    y_loc_arr = np.append(np.insert(rounding,0,np.ceil(y_loc_arr[0])),np.floor(y_loc_arr[-1]))
    
    y_loc = tck.FixedLocator(np.log(y_loc_arr))
    y_loc_min = tck.FixedLocator(np.log(np.concatenate((np.arange(10,100,10), np.arange(100,700,100)))))

    ax.yaxis.set_major_locator(y_loc)
    ax.set_yticklabels(y_loc_arr)
    ax.yaxis.set_minor_locator(y_loc_min)
    ax.set_ylabel(ylabel)

    # Format x axis labels
    x_loc_arr = get_time_ticks(x0,6)
    x_loc = tck.FixedLocator(x_loc_arr)
    x_labels = np.array([dates.num2date(dt).strftime('%d-%b\n%H:%M') for dt in x_loc_arr])
    ax.xaxis.set_major_locator(x_loc)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(xlabel)

    # Format colorbar 
    cbar = fig.colorbar(cf, ax=ax, ticks=z_loc_arr, pad=0.01)

    cbar.ax.set_xticklabels(z_loc_arr)
    cbar.set_label(zlabel)


    if fit_lognormal_modes:
        # Calculate the maximum in each mode using a lognormal fitting procedure
        mode_max = mode_max_from_dist_fit(d_mtx) 
        # Overlay the mode sizes
        ax.plot(np.log(mode_max),'-k')
    
    
    atmosplots.saveorshowplot(plt,saveorshowplot,output_path,output_filename)
    
    return
