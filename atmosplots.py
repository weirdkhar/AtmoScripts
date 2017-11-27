'''
A collection of plots utilised for atmospheric data
A complement to atmoscripts

Written by Ruhi Humphries, 2016

STILL DO TO
- Map plot of GPS location with optional color on track that changes with a variable such as time or concentration. This can be log or linear scale.
- SMPS plots
- Altitude plots (i.e. for sonde data)
'''

'''
GREAT BASEMAP tutorial - https://basemaptutorial.readthedocs.io/en/latest/plotting_data.html#plot



To install basemap:
in anaconda prompt type:
conda install -c conda-forge basemap-data-hires=1.0.8.dev0

From <https://stackoverflow.com/questions/34979424/importing-mpl-toolkits-basemap-on-windows> 
'''
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as d
import datetime as dt
import matplotlib
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfFileMerger, PdfFileReader
#from mpl_toolkits.basemap import Basemap as basemap #Only works on python2.7
import windrose as wr # https://pypi.python.org/pypi/windrose/
#import cartopy as crt #doesn't work
from AtmoScripts import Aero_SizeDist as aero_sizedist




def diurnal(data_timeseries,
            StatToPlot = 'median',
            errorBars = None,
            NumDeviationsForErrorBars = 2,
            ylabel = '',
            xlabel = 'Time of Day',
            color = 'g',
            ylim = None, 
            ylim_bottom = None,
            ylim_top = None,
            SaveOrShowPlot = 'Show',
            output_path = None,
            outputfilename = 'diurnal_cycle',
            title = None
            ):
    ''' Based on https://www.davidhagan.me/articles?id=7
    
    errorBars has the options of:
        None - no error bars are plotted
        deviation - the choice is determined automatically based on which stat is plotted. If mean is chosen, standard deviation is plotted. If median is chosen, median absolute deviation is plotted.
        quartiles - plots the 75th and 25th
        quartiles&range - plots the 75th and 25th quartiles, and an additional line for min and max
        
    
    '''
    
    
    
    ''' Set up your daily statistics '''          
    # Convert a series object to a dataframe
    data = pd.DataFrame(data_timeseries)
    
    # Convert index from string to datetime
    data.index = pd.to_datetime(data.index.astype(str))
    
    # determine time resolution of input dataset, and then group based on that
    if (data.index[1]-data.index[0]).components.seconds > 0:
        timeFormat = "%H:%M:%S"
    elif (data.index[1]-data.index[0]).components.minutes > 0:
        timeFormat = "%H:%M"
    elif (data.index[1]-data.index[0]).components.hours > 0:
        timeFormat = "%H"
    else:
        print("Time resolution too course to do diurnal plot! Ensure the resolution is hourly or higher.")
        return
            
    # create a new column that contains timestamps in the form Hour:Min, make this the index, then delete the column
    data['Time'] = data.index.map(lambda x: x.strftime(timeFormat))
    data.index = data['Time']
    data.drop('Time',axis=1,inplace=True)
    
    # Group the data by its new Time column, and perform some statistics
    #data_day = data.groupby(level=0).describe().unstack()
    #data_day.columns=data_day.columns.get_level_values(1) # Flatten multiindex
    #data_day.rename(columns={'50%' : 'median'},inplace=True)
    
    data_day = stats_from_groupby(data)
    
    
    
    # Set our new index to be of the datetime format for easy plotting
    data_day.index = pd.to_datetime(data_day.index.astype(str))
    
    annotationText = StatToPlot #Initialise
        
        
        
    ''' Plot '''
    # Setup your axis and labels
    fig, ax = plt.subplots(1, figsize=(12,6))
    ax.set_title(data.columns[0]+' Diurnal Profile', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    
    
    # Plot
    ax.plot(data_day.index, data_day[StatToPlot], color, linewidth=2.0)
    
    # Add error bars - either std, MAD, or quartiles
    if errorBars is not None:
        if "deviation" in errorBars.lower():
            if StatToPlot == "median": # Plot MAD with median
                errorToPlot = 'mad'
            elif StatToPlot == "mean":
                errorToPlot = 'std'
            else:
                print("Don't know what to plot for error bars!")
                return
            # Add shading
            ax.fill_between(data_day.index, data_day[StatToPlot], data_day[StatToPlot] - NumDeviationsForErrorBars * data_day[errorToPlot], alpha=.5, facecolor=color, edgecolor='none')
            ax.fill_between(data_day.index, data_day[StatToPlot], data_day[StatToPlot] + NumDeviationsForErrorBars * data_day[errorToPlot], alpha=.5, facecolor=color, edgecolor='none')
            annotationText = StatToPlot + ' $\pm$ ' + str(NumDeviationsForErrorBars) + errorToPlot
        
        # Add quartiles to the plot
        if "quartiles" in errorBars.lower():
            # Add shading
            ax.fill_between(data_day.index, data_day[StatToPlot], data_day['75%'], alpha=.5, facecolor=color, edgecolor='none')
            ax.fill_between(data_day.index, data_day[StatToPlot], data_day['25%'], alpha=.5, facecolor=color, edgecolor='none')
            annotationText = StatToPlot + ' $\pm$ quartiles'
            
            # Add min and max to the plot        
            if "range" in errorBars.lower():
                # Add min max lines
                ax.plot(data_day.index, data_day['max'], alpha=.5, color=color)
                ax.plot(data_day.index, data_day['min'], alpha=.5, color=color)
                annotationText = StatToPlot + ' $\pm$ quartiles, with range'
    
    # Make the axes pretty and show
    ticks = ax.get_xticks()
    ax.set_xticks(np.linspace(ticks[0], d.date2num(d.num2date(ticks[-1]) + dt.timedelta(hours=3)), 5))
    ax.set_xticks(np.linspace(ticks[0], d.date2num(d.num2date(ticks[-1]) + dt.timedelta(hours=3)), 25), minor=True)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%I:%M %p'))
    ax.text(pd.to_datetime('00:30'), (ax.get_ylim()[1]-ax.get_ylim()[0])*0.975 + ax.get_ylim()[0], 
            annotationText, horizontalalignment='left', verticalalignment='top')
    if ylim is not None:
        ax.set_ylim(ylim)
    if ylim_bottom is not None:
        ax.set_ylim(bottom = ylim_bottom)
    if ylim_top is not None:
        ax.set_ylim(top = ylim_top)
    
    plt.tight_layout()
    
    
    saveorshowplot(plt,SaveOrShowPlot,output_path,outputfilename+'_'+data.columns[0])
    
    return
      
def stats_from_groupby(data):
    # group by the index 
    #data = pd.to_numeric(data.columns[0], errors='coerce')
    data = data.apply(lambda x: pd.to_numeric(x, errors='coerce')) # ensure all data is numeric
#    for i in range(0, len(data['CN'])):
#   #     data['CN'][i] = float(data['CN'][i])
#        if type(data['CN'][i]) != float:
#            print(i)
    data_day = pd.DataFrame(data.groupby(level=0).apply(lambda x: x.mean()))
    data_day.rename(columns={data_day.columns[0] : 'mean'},inplace=True)
    data_day['median'] = data.groupby(level=0).apply(lambda x: x.median())
    data_day['min'] = data.groupby(level=0).apply(lambda x: x.min())
    data_day['max'] = data.groupby(level=0).apply(lambda x: x.max())
    data_day['75%'] = data.groupby(level=0).apply(lambda x: np.percentile(x,75))
    data_day['25%'] = data.groupby(level=0).apply(lambda x: np.percentile(x,25))
    data_day['std'] = data.groupby(level=0).apply(lambda x: x.std())    
    data_day['mad'] = data.groupby(level=0).apply(lambda x: np.abs(x - x.median()).median())
    
    
    return data_day

def weekly_cycle(data_timeseries,
            StatToPlot = 'median',
            time_resolution = 'daily',
            errorBars = None,
            NumDeviationsForErrorBars = 2,
            ylabel = '',
            xlabel = 'Day of Week',
            color = 'g',
            ylim_bottom = None,
            ylim_top = None,
            SaveOrShowPlot = 'Show',
            output_path = None,
            outputfilename = 'weekly_cycle',
            title = None,
            SundayStart = True
            ):
    ''' Based on https://www.davidhagan.me/articles?id=7
    
    errorBars has the options of:
        None - no error bars are plotted
        deviation - the choice is determined automatically based on which stat is plotted. If mean is chosen, standard deviation is plotted. If median is chosen, median absolute deviation is plotted.
        quartiles - plots the 75th and 25th
        quartiles&range - plots the 75th and 25th quartiles, and an additional line for min and max
        
    
    '''
    
    
    
    ''' Set up your statistics '''          
    # Convert a series object to a dataframe
    data = pd.DataFrame(data_timeseries)
    
    # Convert index from string to datetime
    data.index = pd.to_datetime(data.index.astype(str))
              
    # create a new column that contains timestamps in the form Hour:Min, make this the index, then delete the column
    
    data['dayofweek'] = data.index.dayofweek    
    if time_resolution == 'hourly':
        data['hour'] = data.index.hour
        data['dayhour'] = [data['dayofweek'][i] + '_' + data['hour'][i] for i in data]
        data.drop('hour',axis=1,inplace=True)
    data.index = data['dayofweek']
    data.drop('dayofweek',axis=1,inplace=True)
    
    
    # Group the data by its new Time column, and perform some statistics
#    data_day = data.groupby(level=0).describe().unstack()
#    data_day.columns=data_day.columns.get_level_values(1) # Flatten multiindex
#    data_day['mad'] = data.groupby(level=0).apply(lambda x: np.abs(x - x.median()).median())
#    data_day.rename(columns={'50%' : 'median'},inplace=True)
    
    data_day = stats_from_groupby(data)
    
    annotationText = StatToPlot #Initialise
        
    if SundayStart: #Create a new dataframe, inserting the last row first
        data_day = pd.DataFrame(data_day.ix[6]).transpose().append(data_day.ix[0:5])
        # Reset index
        data_day.index = np.arange(7)
        xlabels = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']
    else:
        xlabels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        
    ''' Plot '''
    # Setup your axis and labels
    fig, ax = plt.subplots(1, figsize=(12,6))
    ax.set_title(data.columns[0]+' Weekly Profile', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    
    
    # Plot
    ax.plot(data_day.index, data_day[StatToPlot], color, linewidth=2.0)
    
    # Add error bars - either std, MAD, or quartiles
    if errorBars is not None:
        if "deviation" in errorBars.lower():
            if StatToPlot == "median": # Plot MAD with median
                errorToPlot = 'mad'
            elif StatToPlot == "mean":
                errorToPlot = 'std'
            else:
                print("Don't know what to plot for error bars!")
                return
            # Add lines
#            ax.plot(data_day.index, data_day[StatToPlot] - NumDeviationsForErrorBars * data_day[errorToPlot], color=color)
#            ax.plot(data_day.index, data_day[StatToPlot] + NumDeviationsForErrorBars * data_day[errorToPlot], color=color)
            # Add shading
            ax.fill_between(data_day.index, data_day[StatToPlot], data_day[StatToPlot] - NumDeviationsForErrorBars * data_day[errorToPlot], alpha=.5, facecolor=color)
            ax.fill_between(data_day.index, data_day[StatToPlot], data_day[StatToPlot] + NumDeviationsForErrorBars * data_day[errorToPlot], alpha=.5, facecolor=color)
            annotationText = StatToPlot + ' $\pm$ ' + str(NumDeviationsForErrorBars) + errorToPlot
        
        # Add quartiles to the plot
        if "quartiles" in errorBars.lower():
            # Add lines        
            ax.plot(data_day.index, data_day['75%'], alpha=.0, color=color)
            ax.plot(data_day.index, data_day['25%'], alpha=.0, color=color)
            # Add shading
            ax.fill_between(data_day.index, data_day[StatToPlot], data_day['75%'], alpha=.5, facecolor=color)
            ax.fill_between(data_day.index, data_day[StatToPlot], data_day['25%'], alpha=.5, facecolor=color)
            annotationText = StatToPlot + ' $\pm$ quartiles'
            
            # Add min and max to the plot        
            if "range" in errorBars.lower():
                # Add min max lines
                ax.plot(data_day.index, data_day['max'], alpha=.5, color=color)
                ax.plot(data_day.index, data_day['min'], alpha=.5, color=color)
                annotationText = StatToPlot + ' $\pm$ quartiles, with range'
    
    # Make the axes pretty and show
#    ticks = ax.get_xticks()
#    ax.set_xticks(np.linspace(ticks[0], d.date2num(d.num2date(ticks[-1]) + dt.timedelta(hours=3)), 5))
#    ax.set_xticks(np.linspace(ticks[0], d.date2num(d.num2date(ticks[-1]) + dt.timedelta(hours=3)), 25), minor=True)
#    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%I:%M %p'))
    ax.set_xticklabels(xlabels)
    ax.text(0.25, (ax.get_ylim()[1]-ax.get_ylim()[0])*0.975 + ax.get_ylim()[0], 
            annotationText, horizontalalignment='left', verticalalignment='top')
    if ylim_bottom is not None:
        ax.set_ylim(bottom = ylim_bottom)
    if ylim_top is not None:
        ax.set_ylim(top = ylim_top)
    
    plt.tight_layout()

    saveorshowplot(plt,SaveOrShowPlot,output_path,outputfilename+'_'+data.columns[0])

def windrose(wd, ws, bins = np.arange(0,35,5),
             colormap = plt.cm.hot,
             title = 'Wind Rose',
             legendposition = 'lower left',
             SaveOrShowPlot='show',
             output_path = None,
             outputfilename = 'windrose.pdf',
             outputpath = None
             ):
                 
    ax = wr.WindroseAxes.from_ax()
    bins = ax.bar(wd, ws, normed=True, bins = bins, cmap=colormap)
    ax.legend(loc = legendposition)
    ax.set_title(title)
    
    saveorshowplot(plt,SaveOrShowPlot,outputpath,outputfilename)
    
    return  

def plot_timeseries(x_data = None,
             y_data = None,
             y_data_R = None,
             z_data=None,
             colorbar_label = '',
             axes_object = None,
             logscale = False,
             logscale_R = False,             
             drawLegend = False,
             legend='',
             legend_R='',
             title = '',
             xlim = None,
             ylim = None,
             ylim_R = None,
             ylabel = '',
             ylabel_R = '',
             xlabel = '',
             SaveOrShowPlot = 'save',
             outputfilename = 'timeseries.pdf',
             outputpath = None
             ):
    # Function to deal with legacy code that still calls plot_timeseries, rather than just plot  
    plot(x_data = x_data,
         y_data = y_data,
         y_data_R = y_data_R,
         z_data=z_data,
         colors = None,
         colorbar_label = colorbar_label,
         axes_object = axes_object,
         logscale = logscale,
         logscale_R = logscale_R,             
         drawLegend = drawLegend,
         legend=legend,
         legend_R=legend_R,
         title = title,
         xlim = xlim,
         ylim = ylim,
         ylim_R = ylim_R,
         ylabel = ylabel,
         ylabel_R = ylabel_R,
         xlabel = xlabel,
         SaveOrShowPlot = SaveOrShowPlot,
         outputfilename = outputfilename,
         outputpath = outputpath
         )
    return

def plot(x_data = None,
         y_data = None,
         markerstyle = '.',
         y_data_R = None,
         z_data=None,
         colormap = None,
         colors = None,
         figsize = (12,6),
         colorbar_label = '',
         colorbar_orientation = 'vertical',
         axes_object = None,
         logscale = False,
         logscale_R = False,             
         drawLegend = False,
         legendLocation = 'lower right',
         legend='',
         legend_R='',
         bbox_to_anchor = None,
         leg_ncols = 1,
         title = '',
         xlim = None,
         ylim = None,
         ylim_R = None,
         ylabel = '',
         ylabel_R = '',
         xlabel = '',
         SaveOrShowPlot = 'show',
         outputfilename = 'plot.pdf',
         outputpath = None
         ):
    ''' Function that plots each subplot '''
    y_data = pd.DataFrame(y_data)

    
    if legend !=  '':
        drawLegend = True
    if (y_data_R is not None):    #If there is a second axis to plot, use a legend
        drawLegend = True
        
    if (legend == '') and drawLegend:
        legend = y_data.columns.values.tolist()
    if (legend_R == '') and drawLegend:
        legend_R = ylabel_R
    
    
    if z_data is not None:
        figsize0 = (figsize[0]*1.05,figsize[1])
        if colormap is None:
            colormap = 'jet'
        else:
            colormap = colormap
    else:
        figsize0 = figsize
    
    if axes_object is None:
        fig, axes_object = plt.subplots(1,figsize=figsize0)  
        fig.add_subplot()
    
    if colors is None:
        colors = ['b','g','r','c','m','y','k']
    
#    for i in range(len(y_data.columns)):
#        y_data_i = y_data.iloc[:,i]
#        if logscale:
#            a1 = axes_object.semilogy(x_data,y_data_i,'.',color = colors[i], label=legend)
#        else:
#            #a1 = axes_object.plot(x_data,y_data_i,'.',label=legend)
#            a1 = axes_object.scatter(x_data,y_data_i,c=z_data,edgecolors='none',facecolor = colors[i],label=legend)
    a1s = []
    for col, lgd in zip(y_data.columns, legend):
        if logscale:
             a1 = axes_object.semilogy(x_data,y_data[col],markerstyle, label=lgd,markeredgecolor='none')
        else:
    #        if z_data is None: 
            if y_data.shape[1]>1:
                a1 = axes_object.plot(x_data,y_data[col],markerstyle,label=lgd,markeredgecolor='none')
    #        else: #only plotting 1 dataset, but with z data optional. Need to use this to add a colorbar
            #a1 = axes_object.scatter(x_data, y_data, c=z_data, cmap = colormap, marker = markerstyle, edgecolors='none', label=legend)
            else:
                a1 = axes_object.scatter(x_data, y_data[col], c=z_data, marker = markerstyle, edgecolors='none', label=lgd)
    #         a1 = axes_object.plot(x_data, y_data, c=z_data, color = colormap, marker = markerstyle, markeredgecolor='none', label=legend)
        a1s = a1s + a1  
         
    axes_object.set_ylabel(ylabel, fontsize=14)
    axes_object.set_xlabel(xlabel, fontsize=14)
    
    # Add a colorbar such that the original frame doesn't change size to fit it (i.e. plot colorbar on separate axis)
    if z_data is not None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(plt.gca())
        if colorbar_orientation == 'horizontal':          
            cax = divider.append_axes("bottom","3%", pad="12%")
        else:
            cax = divider.append_axes("right","3%", pad="1.76%")
    #        box = axes_object.get_position()     
    #        axColor = plt.axes([box.x0 + box.width * 1.05, box.y0, 0.03, box.height])
            #plt.colorbar(a1, cax = cax ,label=colorbar_label)
        # Getting colorbar to work with plt.plot, rather than plt.scatter - see http://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots
    #    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
        # fake up the array of the scalar mappable. Urgh...
    #    sm._A = []
    
        plt.colorbar(a1, cax = cax, label=colorbar_label, orientation = colorbar_orientation)

    
    
    if title != "":
        axes_object.set_title(title)
    else:
        axes_object.set_title('time series', fontsize=14)
    
    axes_object.set_ylabel(ylabel, fontsize=14)
    axes_object.set_xlabel(xlabel, fontsize=14)

    
    
    
    
    if y_data_R is not None:
        ax = axes_object.twinx()
        # Change colour cycle for second axes
#        from cycler import cycler
#        ax.set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']) +
#                   cycler('lw', [1, 2, 3, 4]))
#        for advance in range(y_data.shape[1]):
#            next(ax._get_lines.prop_cycler)
        colors = ['c', 'm', 'y', 'k']
        
        # Change any series data to dataframe so that iteration will work 
        y_data_R = pd.DataFrame(y_data_R)
        a2s = []
        for i in range(len(y_data_R.columns)):
            y_data_i = y_data_R.iloc[:,i]
            if logscale:
                a2 = ax.semilogy(x_data,y_data_i,'.',label=legend_R, color=colors[i])
            else:
                if y_data.shape[1]>1:
                    a2 = ax.plot(x_data,y_data_i,'.',label=legend_R,markeredgecolor='none', color=colors[i])        
                else:
                    a2 = ax.scatter(x_data,y_data_i,c=z_data,edgecolors='none',label=legend_R, color=colors[i])   
            a2s = a2s+a2
        ax.set_ylabel(ylabel_R)
        
        if ylim_R is not None:
            ax.set_ylim(ylim_R)
        
        if xlim is None:
            xlim = [x_data.min(),x_data.max()]
        ax.set_xlim(xlim)
        
        if drawLegend:
            # Create combined legend
            labels1 = [a.get_label() for a in a1s]
            labels2 = [a.get_label() for a in a2s]
            

            
            labs = labels1+labels2
            axes = a1s+a2s
            #labs = [l.get_label() for l in a1s]
            ax.legend(axes, labs, loc=legendLocation, 
                      bbox_to_anchor = bbox_to_anchor, ncol= leg_ncols, 
                      numpoints = 1)
    else:
        if drawLegend:
            axes_object.legend(legend, loc=legendLocation, 
                               bbox_to_anchor = bbox_to_anchor, ncol= leg_ncols, 
                               numpoints = 1)
            
    
    
    # Make the axes pretty and show
    if ylim is not None:
        axes_object.set_ylim(ylim)
        
    if xlim is None:
        xlim = [x_data.min(),x_data.max()]
    axes_object.set_xlim(xlim)
    
    
    
    ticks = axes_object.get_xticks()
    
    # if we're plotting time on the x axis, Set the x-tick format to change depending on the span being plotted
    if type(x_data[0]) is pd.Timestamp:
        if xlim is not None:
            # convert string date to datetime
            if type(xlim[0]) is str:
                try:
                    dateformat= '%Y-%m-%d'
                    d.datetime.datetime.strptime(xlim[1],dateformat)
                except TypeError:                     
                    try:
                        dateformat= '%Y-%b-%d'
                        d.datetime.datetime.strptime(xlim[1],dateformat)
                    except TypeError:
                        print('check your xlim date format - it should be either 2016-03-15 or 2016-Mar-15')
    
                span = (
                        (d.datetime.datetime.strptime(xlim[1],dateformat)-d.datetime.datetime.strptime(xlim[0],dateformat)).days + 
                        (d.datetime.datetime.strptime(xlim[1],dateformat)-d.datetime.datetime.strptime(xlim[0],dateformat)).seconds/3600/24
                        )
            else:
                span = (xlim[1]-xlim[0]).days + (xlim[1]-xlim[0]).seconds/3600/24
        else:
            span = (x_data.max()-x_data.min()).days + (x_data.max()-x_data.min()).seconds/3600/24
        
        xax = axes_object.get_xaxis()
        if span <= 1:
            if span < 1/4: # less than 6 hours, do 30 min steps
                xax.set_major_locator(d.MinuteLocator(interval=30))
            elif span <= 1/3: # less than 8 hours, do 1 hr steps
                xax.set_major_locator(d.HourLocator(interval=1))                
            elif span <= 1/2: # less than 12 hours, use 2 hr steps
                xax.set_major_locator(d.HourLocator(interval=2))
            else: # use 3 hour steps
                xax.set_major_locator(d.HourLocator(interval=3))
            axes_object.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        elif span <= 4:            
            xax.set_major_locator(d.DayLocator())
            xax.set_major_formatter(d.DateFormatter('%b-%d'))

            xax.set_minor_locator(d.HourLocator(byhour=range(0,24,3)))
            xax.set_minor_formatter(d.DateFormatter('%H:%M'))
            xax.set_tick_params(which='major', pad=15)
        else:                
            axes_object.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%b-%d'))
            axes_object.set_xticks(np.linspace(ticks[0], d.date2num(d.num2date(ticks[-1])), 5))
            axes_object.set_xticks(np.linspace(ticks[0], d.date2num(d.num2date(ticks[-1])), 25), minor=True)

    plt.tight_layout()    
    
    
    
    
    saveorshowplot(plt,SaveOrShowPlot,outputpath,outputfilename)
    
    return

def plot_daily(x_data = None,
             y_data = None,
             y_data_R = None,
             axes_object = None,
             logscale = False,
             logscale_R = False,             
             drawLegend = False,
             legend='',
             legend_R='',
             title = '',
             xlim = None,
             ylim = None,
             ylim_R = None,
             ylabel = '',
             ylabel_R = '',
             xlabel = '',
             SaveOrShowPlot = 'save',
             outputfilename = 'Daily.pdf',
             outputpath = None
             ): 
    numDays = (x_data.max() - x_data.min()).days
    for i in range(0,numDays):
        if i == 0:
            plotday = x_data.min().strftime('%Y-%b-%d')
            plotday_altFormat = x_data.min().strftime('%Y-%m-%d')
        else:
            plotday = d.datetime.datetime.fromordinal((xlim_min.toordinal()+1)).strftime('%Y-%b-%d')
            plotday_altFormat = d.datetime.datetime.fromordinal((xlim_min.toordinal()+1)).strftime('%Y-%m-%d')
        xlim_min = d.datetime.datetime.strptime(plotday,'%Y-%b-%d')
        xlim_max = d.datetime.datetime.fromordinal((xlim_min.toordinal()+1))
        
        filename = outputfilename.split(sep=".")[0] + ' ' + plotday_altFormat +'.pdf'
        
        if outputpath is not None:            
            try:
                directory 
            except:
                directory = outputpath
                if not os.path.exists(directory):
                    if str.split(outputpath,'\\')[0] == 'c:': # if the user specified a folder name, rather than a full path
                        directory = outputpath
                    else:
                        directory = os.getcwd() + '\\' + outputpath
                    if not os.path.exists(directory):
                        os.makedirs(directory)
        else:
            directory = None
                    
        plot_timeseries(x_data = x_data,
                     y_data = y_data,
                     y_data_R = y_data_R,
                     axes_object = axes_object,
                     logscale = logscale,
                     logscale_R = logscale_R,             
                     drawLegend = drawLegend,
                     legend=legend,
                     legend_R=legend_R,
                     title = title,
                     xlim = [xlim_min,xlim_max],
                     ylim = ylim,
                     ylim_R = ylim_R,
                     ylabel = ylabel,
                     ylabel_R = ylabel_R,
                     xlabel = plotday,
                     SaveOrShowPlot = SaveOrShowPlot,
                     outputfilename = filename,
                     outputpath = directory
                     )
    return
  
def plot_interval(x_data = None,
             y_data = None,
             interval_hours = 24,
             y_data_R = None,
             axes_object = None,
             logscale = False,
             logscale_R = False,             
             drawLegend = False,
             legend='',
             legend_R='',
             title = '',
             xlim = None,
             ylim = None,
             ylim_R = None,
             ylabel = '',
             ylabel_R = '',
             xlabel = '',
             SaveOrShowPlot = 'save',
             outputfilename = 'Daily.pdf',
             outputpath = None
             ): 

    # Calculate number of periods to plot
    timerange = (x_data.max() - x_data.min())
    range_hrs = timerange.days * 24 + timerange.seconds/3600
    periods = math.ceil(range_hrs/interval_hours)

    for i in range(0,periods):
        # Take the modulus - if the time interval doesn't fit into 24 hours, 
        # begin time intervals at the beginning of the period, and work 
        # continuously through.
        # if it does fit into 24 hours, start each day at midnight
        if 24 % interval_hours == 0 :
            if i == 0: #initialise
                period_min = x_data[0] - d.datetime.timedelta(hours=(x_data[0].hour % interval_hours)) #start the graph earlier than the start point, so that the interval makes sense within the day (e.g. 3 hour plots with data starting at 7pm would range between 6-9pm)
                period_max = period_min + d.datetime.timedelta(hours=interval_hours)
            else:
                period_min = period_max
                period_max = period_min + d.datetime.timedelta(hours=interval_hours)                
        else:
            if i == 0: #initialise
                period_min = x_data[0].replace(minute=0,second=0)
                period_max = period_min + d.datetime.timedelta(hours=interval_hours)
            else:
                period_min = period_max
                period_max = period_min + d.datetime.timedelta(hours=interval_hours)
        
        datestr = period_min.date().strftime('%Y-%m-%d')
        t0str = period_min.time().strftime('%H%M')
        t1str = period_max.time().strftime('%H%M')
        time_label = datestr + ' ' + t0str + '-' + t1str
                
        filename = outputfilename.split(sep=".")[0] + ' ' + time_label +'.pdf'
        
        if outputpath is not None:
            try:
                directory 
            except:
                directory = outputpath
                if not os.path.exists(directory):
                    if str.split(outputpath,'\\')[0] == 'c:': # if the user specified a folder name, rather than a full path
                        directory = outputpath
                    else:
                        directory = os.getcwd() + '\\' + outputpath
                    if not os.path.exists(directory):
                        os.makedirs(directory)
            
        plot_timeseries(x_data = x_data,
                     y_data = y_data,
                     y_data_R = y_data_R,
                     axes_object = axes_object,
                     logscale = logscale,
                     logscale_R = logscale_R,             
                     drawLegend = drawLegend,
                     legend=legend,
                     legend_R=legend_R,
                     title = title,
                     xlim = [period_min,period_max],
                     ylim = ylim,
                     ylim_R = ylim_R,
                     ylabel = ylabel,
                     ylabel_R = ylabel_R,
                     xlabel = datestr,
                     SaveOrShowPlot = SaveOrShowPlot,
                     outputfilename = filename,
                     outputpath = directory
                     )
    return

def subplots(INDEXED_DATA, 
             XLIMS = None, 
             YLIMS = None, 
             TITLES = None, 
             XLABELS = None, 
             YLABELS = None,
             LOGSCALES = None,
             LEGENDS = None,
             LEGENDLOCATIONS = None,
             BBOX_TO_ANCHOR = None,
			 
             sharex = True,
             figsize = (9,9),
             
             SaveOrShowPlot='show',
             outputpath = None,
             outputfilename = 'subplots.pdf'
             
             ):
    # For variables that aren't specified, populate so that the zip function will work
    if XLIMS is None:
        XLIMS = [None] * len(INDEXED_DATA)
    if YLIMS is None:
        YLIMS = [None] * len(INDEXED_DATA)
    if TITLES is None:
        TITLES = [''] * len(INDEXED_DATA)
    if XLABELS is None:
        XLABELS = [''] * len(INDEXED_DATA)
    if YLABELS is None:
        YLABELS = [''] * len(INDEXED_DATA)
    if LOGSCALES is None:
        LOGSCALES = [False] * len(INDEXED_DATA)
    if LEGENDS is None:
        LEGENDS = [None] * len(INDEXED_DATA)
    if LEGENDLOCATIONS is None:
        LEGENDLOCATIONS = [1] * len(INDEXED_DATA)
    if BBOX_TO_ANCHOR is None:
        BBOX_TO_ANCHOR = [None] * len(INDEXED_DATA)
	
    
    f, axarr = plt.subplots(len(INDEXED_DATA),1, sharex=sharex, figsize = figsize)
    for j, (dat, logscale, xlim, ylim, title, xlabel, ylabel, logscale, legend, legendLoc, bbox_anchor) in enumerate(zip(INDEXED_DATA, LOGSCALES, XLIMS, YLIMS, TITLES, XLABELS, YLABELS, LOGSCALES, LEGENDS, LEGENDLOCATIONS, BBOX_TO_ANCHOR)):
        ax = axarr[j]
        if logscale:
            ax.semilogy(dat,'.')
        else:
            ax.plot(dat,'.')
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)       
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        f.autofmt_xdate() # Automatically rotate the date axis labels
        if legend is not None:
            ax.legend(legend, bbox_to_anchor=bbox_anchor, loc=legendLoc, numpoints = 1)
        
    
    saveorshowplot(plt,SaveOrShowPlot,outputpath,outputfilename)
    
    return


def subplots_singlecolumn_timeseries(df,
             NumOfSubplots = 2,
             common_xaxis = True,
             x_col_common = None,
             
             x1_col = None,
             y1_cols = None,
             y1_legend_labels = None,             
             y1_cols_right = None,
             y1_legend_labels_right = None,
             y1_lim = None,
             y1_lim_right = None,
             y1_title = '',
             y1_label = '',
             y1_label_right = '',
             y1_logscale = False,
             y1_logscale_right = False,
             
             x2_col = None,
             y2_cols = None,
             y2_legend_labels = None,             
             y2_cols_right = None,
             y2_legend_labels_right = None,
             y2_lim = None,
             y2_lim_right = None,
             y2_title = '',
             y2_label = '',
             y2_label_right = '',
             y2_logscale = False,
             y2_logscale_right = False,
             
             x3_col = None,
             y3_cols = None,
             y3_legend_labels = None,             
             y3_cols_right = None,
             y3_legend_labels_right = None,
             y3_lim = None,
             y3_lim_right = None,
             y3_title = '',
             y3_label = '',
             y3_label_right = '',
             y3_logscale = False,
             y3_logscale_right = False,
             
             x4_col = None,
             y4_cols = None,
             y4_legend_labels = None,             
             y4_cols_right = None,
             y4_legend_labels_right = None,
             y4_lim = None,
             y4_lim_right = None,
             y4_title = '',
             y4_label = '',
             y4_label_right = '',
             y4_logscale = False,
             y4_logscale_right = False,
             
             SaveOrShowPlot='show',
             outputpath = None,
             outputfilename = 'timeseries_subplots.pdf'
             ):
    '''
    This function creates a plot with up to 4 subplots, each of which can have dual y-axes
    '''
    #Determine how many axes are required based on the input data
    if y1_cols is None:
        print('Please specify the data for the first plot')
        return
    else:
        if y2_cols is None:
            print('Please specify the data for the second plot')
            return
        else:
            if y3_cols is not None:
                NumOfSubplots = 3
                if y4_cols is not None:
                    NumOfSubplots = 4
    
    # Create axes
    if NumOfSubplots == 2:
        f, (ax1, ax2) = plt.subplots(2, 1, sharex = common_xaxis, figsize=(12,6))
    elif NumOfSubplots == 3:
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = common_xaxis, figsize=(12,6))        
    elif NumOfSubplots == 4:
        f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex = common_xaxis, figsize=(12,6))
    else:
        print("Number of subplots out of range of this function! Please specify between 2 and 4")
        return
    
    if common_xaxis:
        if x_col_common is not None:
            x_col = x_col_common
        else:
            print("Please specify what the common x axis is")
            return
    else:
        if x1_col is not None:        
            x_col = x1_col
        else:
            print("You are not using a common x-axis. Please specify each x data")
            return
    if x_col.lower() == "index":
        xdata = df.index
    else:
        xdata = df[x_col]
    
    if y1_title == '':
            y1_title = y1_cols
    if y2_title == '':
            y2_title = y2_cols
    if y3_title == '':
            y3_title = y3_cols     
    if y4_title == '':
            y4_title = y4_cols

    plot_timeseries(axes_object = ax1,
                    x_data = xdata,
                    y_data = df[y1_cols],
                    y_data_R = df[y1_cols_right],
                    logscale = y1_logscale,
                    logscale_R = y1_logscale_right,             
#                    drawLegend = False,
                    legend=y1_legend_labels,
                    legend_R=y1_legend_labels_right,
                    title = y1_title,
                    ylim = y1_lim,
                    ylim_R = y1_lim_right,
                    ylabel = y1_label,
                    ylabel_R = y1_label_right,
                     
                    SaveOrShowPlot = 'wait'
                    )
    plot_timeseries(axes_object = ax2,
                    x_data = xdata,
                    y_data = df[y2_cols],
                    logscale = y2_logscale,
                    logscale_R = y2_logscale_right,             
#                    drawLegend = False,
                    legend=y2_legend_labels,
                    legend_R=y2_legend_labels_right,
                    title = y2_title,
                    ylim = y2_lim,
                    ylim_R = y2_lim_right,
                    ylabel = y2_label,
                    ylabel_R = y2_label_right,
                     
                    SaveOrShowPlot = 'wait'
                    )
    
    if NumOfSubplots >= 3:
        plot_timeseries(axes_object = ax3,
                    x_data = xdata,
                    y_data = df[y3_cols],
                    logscale = y3_logscale,
                    logscale_R = y3_logscale_right,             
#                    drawLegend = False,
                    legend=y3_legend_labels,
                    legend_R=y3_legend_labels_right,
                    title = y3_title,
                    ylim = y3_lim,
                    ylim_R = y3_lim_right,
                    ylabel = y3_label,
                    ylabel_R = y3_label_right,
                     
                    SaveOrShowPlot = 'wait'
                    )
    
    if NumOfSubplots >= 4:
        plot_timeseries(axes_object = ax4,
                    x_data = xdata,
                    y_data = df[y4_cols],
                    logscale = y4_logscale,
                    logscale_R = y4_logscale_right,             
#                    drawLegend = False,
                    legend=y4_legend_labels,
                    legend_R=y4_legend_labels_right,
                    title = y4_title,
                    ylim = y4_lim,
                    ylim_R = y4_lim_right,
                    ylabel = y4_label,
                    ylabel_R = y4_label_right,
                     
                    SaveOrShowPlot = 'wait'
                    )
            
            
    
    saveorshowplot(plt,SaveOrShowPlot,outputpath,outputfilename)
        
    return


def plot_specgram(
            sizedist_matrix,
            sizedist_xaxis = None,
            sizedist_yaxis = None,
            as_subplot = False,
            fig = None,
            ax = None,
            figsize = (12,4),
            zmin = 1,
            zmax = 1000,
            yticklocation = None,        
            logscale_z = True,
            ylabel = 'Mobility diameter (nm)',
            xlabel = '',
            title = '',
            x_axis_visible = True,
            SaveOrShowPlot = 'show',
            outputfilename = 'plot.pdf',
            outputpath = None
		):
	''' Input data can be:
		- a dataframe with x and y values as the index and column names respectively, any definted x or y axis is ignored
		- a numpy matrix, where the sizedist_xaxis and sizedist_yaxis must be defined
		
		as_subplot - default False. 
                  Set to true if you want this figure as a subplot of a larger figure. 
                  If set to true, you must input a fig object
		
		
	'''	
	
	# Format the input data
	if type(sizedist_matrix) == pd.core.frame.DataFrame: # dataframe input
		xaxis = sizedist_matrix.index
		yaxis = [float(i) for i in sizedist_matrix.columns]	
	elif type(sizedist_matrix) == np.ndarray:
		if (sizedist_xaxis or sizedist_yaxis) == None:
			print('Input data is a numpy array, please define both x and y axis')
		xaxis = sizedist_xaxis
		yaxis = sizedist_yaxis
	xlim = (xaxis.min(), xaxis.max())
	
     # Make sure data gaps are displayed
	dfidx = pd.DataFrame(index = pd.date_range(start = xaxis[0], end = xaxis[-1], freq = '5Min'))
	sizedist_matrix = pd.concat([sizedist_matrix, dfidx],axis=1)
	
	# Plot
	if as_subplot:
         if (fig is None) or (ax is None):
             print("ERROR - to plot as part of a subplot, you must define fig and ax")
         ax = aero_sizedist.plot_axes(1, sizedist_matrix,yaxis,ylabel,title,xlim, fig,1, zmin, zmax, yticklocation, logscale_z, ax = ax)
	
	else:
         fig = plt.figure(figsize=(figsize))
         ax = aero_sizedist.plot_axes(1, sizedist_matrix,yaxis,ylabel,title,xlim, fig,1, zmin, zmax, yticklocation, logscale_z)
	
	
     
	if x_axis_visible:
		plt.setp(ax.get_xticklabels(), rotation=-20, horizontalalignment='left')
		# If less than 1 day of data is being plotted, put a day label on the plot so that xlabels can be times.
		if (xlim[1]-xlim[0]).days < 1:   
			daylabel = (xlim[0]).strftime("%b %d")
			bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8)
			ax.annotate(daylabel, xy=(0.9, 0.9), xycoords='axes fraction', bbox=bbox_props)
	
	saveorshowplot(plt,SaveOrShowPlot, outputpath, outputfilename)
	return
#%% Map Plots
### Plotting maps
# import cartopy
# import basemap
# import folium

# Projection - winkel Tripel

def map_satellite_overlay():
    #http://scitools.org.uk/cartopy/docs/latest/matplotlib/advanced_plotting.html
    return



 
#%%
def save2pdf(fig_object = None, output_pdf_filename = 'plots.pdf', output_path = None):
    # Save plots to pdf file http://matplotlib.org/faq/howto_faq.html
    
    
    if fig_object is None:
        print("Error! Please pass a figure object")
        return

    if output_path is not None:
        os.chdir(output_path)
    
    # If file exists, append, otherwise create a new file
    if os.path.isfile(output_pdf_filename):
        pdf = PdfPages('temporary_fig.pdf') # Open the file
        append2pdf = True
    else:
        pdf = PdfPages(output_pdf_filename) # Open the file
        append2pdf = False
    
    pdf.savefig(fig_object)
    pdf.close() # Close the file        
            
    if append2pdf:
        appendfig2pdf(output_pdf_filename)
        
    return

def appendfig2pdf(original_filename):
    
    merger = PdfFileMerger()    
    merger.append(PdfFileReader(original_filename,"rb"))
    merger.append(PdfFileReader('temporary_fig.pdf',"rb"))
    merger.write('merged.pdf')
    
    os.remove('temporary_fig.pdf')
    os.remove(original_filename)
    os.rename('merged.pdf', original_filename)    
    
    return
 


#%%
def map_shiptrack(lat,
                  lon,
                  z_data = None,
                  trace_colour = 'r',
                  maplim_lat = None,
                  maplim_lon = None,
                  llcrnrlat = None,
                  llcrnrlon = None,
                  urcrnrlat = None,
                  urcrnrlon = None,
                  projection = 'ortho',
                  map_origin = [0,0],
                  bluemarble = False,
                  NoFillColour = False,
                  OceanColour = 'b',
                  ContinentColour = 'coral',
                  LakeColour = None,
                  SaveOrShowPlot='show',
                  outputpath = None,
                  outputfilename = 'map_shiptrack'):
    
    '''
    Projection options:
    http://matplotlib.org/basemap/users/mapsetup.html
    http://matplotlib.org/basemap/api/basemap_api.html
    
    ''' 
#    # Setup iteration if more than one dataset is being plotted  
#    i = 0
#    while i in range(0,len(lat)):
#        if type(lat) is not list: # When only one dataset is input
#            i = len(lat)
#            lat0 = lat.values
#            lon0 = lon.values
#            trace_colour0 = trace_colour
#        else: #When multiple datasets are input
#            lat0 = lat[i].values
#            lon0 = lon[i].values
#            if trace_colour is not list:
#                trace_colour = ['r','b','g','c','m','y','b','w']
#            trace_colour0 = trace_colour[i]
#                
#        for j in range(0,len(lon0)):
#            if lon0[j] < 0:
#                lon0[j] = lon0[j] + 360
#                
#        # Format data  before plotting
#        # Remove nans
#        lats = lat0[~np.isnan(lon0) | ~np.isnan(lat0)]
#        lons = lon0[~np.isnan(lon0) | ~np.isnan(lat0)]
#        
#        if z_data is not None:
#            z_data = z_data.values
#            z_data = z_data(~np.isnan(lons) | ~np.isnan(lats))
#        
#      
#
#        map = basemap(lat_0=map_origin[0],lon_0=map_origin[1],
#                      projection = projection,
#                      llcrnrlat = llcrnrlat,
#                      llcrnrlon = llcrnrlon,
#                      urcrnrlat = urcrnrlat,
#                      urcrnrlon = urcrnrlon,
#                      resolution = 'l')
#        
#        x, y = map(lons,lats)
#        
#        if LakeColour is None:
#            LakeColour = OceanColour
#        if bluemarble:    
#            map.bluemarble()
#        elif NoFillColour:
#            map.drawmapboundary()
#            map.fillcontinents()
#            map.drawcoastlines()
#        else:
#            map.drawmapboundary(fill_color=OceanColour)
#            map.fillcontinents(color=ContinentColour,lake_color=LakeColour)
#            map.drawcoastlines()
#        
#        map.drawmeridians([0,30,60,90,120,150,180,210,240,270,300,330], labels=[0,0,1,1])
#        map.drawparallels([-60,-45,-30,-15,0,15,30,45,60], labels=[1,1,0,0])
#        
#        # Set the function to interpret lat lons as what they are.
#        map.latlon = True
#        
#        if z_data is None:
#            map.scatter(x,y,color=trace_colour0)
#            #map.plot(x,y,color=trace_colour0)
#        else:
#            map.scatter(x,y,c=z_data)
#            #map.plot(x,y,c=z_data)
#            map.colorbar()
#        
#        i = i+1
#        
        
    saveorshowplot(plt,SaveOrShowPlot,outputpath,outputfilename)
    
    return
    

def plot_smps_aps_nano(
        aps=None, 
        smps=None,
        nano=None,
        figsize = (12,4),
        common_xlim = None,
        zmin1 = 1,
        zmax1 = 1000,
        yticklocation1 = None,
        zmin2 = 1,
        zmax2 = 1000,
        yticklocation2 = None,
        zmin3 = 1,
        zmax3 = 1000,
        yticklocation3 = None,
        SaveOrShowPlot = 'show',
        outputfilename = 'plot.pdf',
        outputpath = None):
    
    aero_sizedist.plot(
        aps=aps, 
        smps=smps,
        nano=nano,
        figsize = figsize,
        common_xlim = common_xlim,
        zmin1 = zmin1,
        zmax1 = zmax1,
        yticklocation1 = yticklocation1,
        zmin2 = zmin2,
        zmax2 = zmax2,
        yticklocation2 = yticklocation2,
        zmin3 = zmin3,
        zmax3 = zmax3,
        yticklocation3 = yticklocation3,
        SaveOrShowPlot = SaveOrShowPlot,
        outputfilename = outputfilename,
        outputpath = outputpath
        )
    return


def saveorshowplot(plt,SaveOrShowPlot, outputpath, outputfilename,dpi=None):
    outputfilename = friendly_filename(outputfilename)
    
    if SaveOrShowPlot.lower() == 'show':
        plt.show()
    elif SaveOrShowPlot.lower() == 'save':
        # Save plots to pdf file http://matplotlib.org/faq/howto_faq.html
        if outputpath is None:
            outputpath = os.path.join(os.path.expanduser('~'), 'Desktop\\Plots')
            if not os.path.exists(outputpath):
                os.makedirs(outputpath)
            print('Output path was not specified. Plots have been saved to "Desktop\Plots".')
        os.chdir(outputpath)
        plt.savefig(outputfilename+'.png',dpi=dpi)
        #save2pdf(fig,outputfilename+'.pdf',outputpath)
        plt.close()
    elif SaveOrShowPlot.lower() == 'wait':
        # do nothing for now, because we're just part of a subplot
        return
    else:
        #Assume you're asking for a different file format that can be handled by pyploy
        fname = outputfilename.split('.')[0]
        if outputpath is None:
            plt.savefig(fname+'.'+SaveOrShowPlot)
        else:
            plt.savefig(outputpath+fname+'.'+SaveOrShowPlot)
        plt.close()
            
    return
    
def friendly_filename(fname, deletechars = '\/:*?"<>|'):
    for ch in deletechars:
        fname= fname.replace(ch,'-')
    #fname = fname.split('.')[0]
    
    x = fname.split('.')
    if len(x) > 1:
        fname = fname.replace('.','p')
    return fname
'''
import sys
sys.path.append('c:\\Dropbox\\RuhiFiles\\Research\\ProgramFiles\\pythonfiles\\')

import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
import pandas as pd
import RVI_Underway
import CPC_TSI
import CCNC
import CWT
import Filter_Timeseries as fTS

import windrose as wr # https://pypi.python.org/pypi/windrose/


# CAPRICORN MASTER PROCESSING DOCUMENT
MASTER_path = 'c:/Dropbox/RuhiFiles/Research/Projects/2015_CWT/Data/Processing/'

os.chdir(MASTER_path)
cwt = pd.read_hdf('cwt_aerosol_1min.h5',key='aerosol')
manual_exhaust = CWT.exhaust_filt()

RVI_Underway.create_exhaust_mask(cwt, 
                                 mask_level_num = 1, 
                                 Filter4WindDir = True, 
                                 Filter4BC = True,
                                 Filter4O3 = True,
                                 Filter4CNstd = False,
                                 WD_exhaust_upper_limit = 219, WD_exhaust_lower_limit = 142,
                                 BC_lim = 0.05
                                 ,manual_exhaust_mask = manual_exhaust
                                 )
cwtL1 = cwt.copy()
cwtL1.loc[cwt['exhaust_mask_L1'].isnull()] = np.nan

#diurnal(cwtL1,'cn10', SaveOrShowPlot='save', StatToPlot = 'median', errorBars = 'quartiles', ylim_bottom = 0,title='CN10 Diurnal Cycle, CWT', ylabel = 'Number Conc. ($cm^{-3}$)')
#weekly_cycle(cwtL1,'cn10', SaveOrShowPlot='save', StatToPlot = 'median', errorBars = 'quartiles', ylim_bottom = 0,title='CN10 Diurnal Cycle, CWT', ylabel = 'Number Conc. ($cm^{-3}$)')

#weekly_cycle(cwtL1,'atmPress', SaveOrShowPlot='save', StatToPlot = 'median', errorBars = 'quartiles', ylim_bottom = 0,title='atmPress Diurnal Cycle, CWT', ylabel = 'Number Conc. ($cm^{-3}$)')

#plot_timeseries(cwtL1.index,cwtL1['cn10'],ylabel='cn10', logscale=True,
#                y_data_R = cwtL1['cn10']*2, ylabel_R='lat',
#                SaveOrShowPlot='show',drawLegend=True)


map_shiptrack([cwtL1['lat'],cwt['lat']],[cwtL1['lon'],cwt['lon']+20],map_origin=[-45,140],NoFillColour=True)
#subplots_singlecolumn_timeseries(cwtL1,
#                                 x_col_common = 'index',
#                                 y1_cols = 'lat',
#                                 y1_cols_right = 'lon',
#                                 y2_cols = 'cn10',
#                                 y3_cols = 'lon',
#                                 SaveOrShowPlot='save')
#df,
#             NumOfSubplots = 2,
#             common_xaxis = True,
#             x_col_common = None,
#             
#             x1_col = None,
#             y1_cols_left = None,
#             y1_legend_labels_left = None,             
#             y1_cols_right = None,
#             y1_legend_labels_right = None,
#             y1_lim_left = None,
#             y1_lim_right = None,
#             y1_title = '',
#             y1_label_left = '',
#             y1_label_right = '',
#             y1_logscale_left = False,
#             y1_logscale_right = False,
#
#
#cwtL1.index,cwtL1['cn10'],ylabel='cn10', logscale=True,
#                y_data_R = cwtL1['lat'], ylabel_R='lat',
#                SaveOrShowPlot='show',drawLegend=True)

#data = pd.DataFrame(data)
#data.index = pd.to_datetime(data.index.astype(str))
#data['Time'] = data.index.map(lambda x: x.strftime("%H:%M"))
#data_day = data.groupby('Time').describe().unstack()
'''