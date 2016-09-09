'''
This module contains functions related to the filtering of timeseries
'''

def linearFit_filt(df,
                   dfColumn,
                   data_time_resolution = 1, 
                   data_time_resolution_units = 'sec',
                   subset_span_period = 10,
                   subset_span_period_units = 'sec'):
    ''' Input data formats explained:
        df:                         the dataframe containing the data you want to filter
        dfColumn:                   the column within the dataframe you want to filter
        data_time_resolution_val:   the value of the time resolution (e.g. "5" if resolution is 5 seconds)
        data_time_resolution_units: the unit of the time resolution. Options of sec, min, hr, day
        subset_span_period:         the period over which the linear fit is calculated. MUST be in the same units as time resolution and a multiple of val
    
    
    Function removes outliers based on values being outside the residual of a linear regression to the 
    data from the previous subset_span_period (forward in time)
    '''
                   

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Define MAD calculation
    mad = lambda x: np.median(np.abs(x - np.median(x)))
    
        
    # Extract x and y vectors to fit
    y = df[dfColumn].values.copy()
    x = np.arange(len(y))
    
    
    # Calculate the span
    # Convert everything to seconds except month and year
    lst_units = [data_time_resolution_units,subset_span_period_units]
    lst_vals = [data_time_resolution,subset_span_period]
    for i in range(2):
        if lst_units[i] == 'min':
            lst_vals[i] = lst_vals[i]*60
        if lst_units[i] == 'hr':
            lst_vals[i] = lst_vals[i]*60*60
        if lst_units[i] == 'day':
            lst_vals[i] = lst_vals[i]*60*60*24
    # Calculate the number of data points used in the calculation, i.e. span
    if (np.mod(lst_vals[1], lst_vals[0]) == 0) & (lst_vals[1] > lst_vals[0]):
        span = lst_vals[1]/lst_vals[0]
    else:
        return print("span period must be a multiple of the time resolution")
        

    # Weighting - more towards first half of fitting set, none on final point in question
    w1 = np.ones(span/2)
    w2 = np.ones(span/2-1)/2
    w3 = np.zeros(1)
    
    
    for i in range(len(y)):
    #for i in range(0,6920):
    #i = 5900 # 5928
        weights = np.concatenate((w1,w2,w3),axis=0)
        
        ysub = y[i-int(span):i].copy()
        xsub = x[i-int(span):i].copy()
        
        if np.isfinite(y[i-1]):    
            if np.isfinite(ysub).sum() > 10: # only continue if there is sufficient data!
                # Remove nan's since poly fitting doesn't work on them!
                idx = np.isfinite(ysub)
    
                ysub = ysub[idx]
                xsub = xsub[idx]
                weights = weights[idx]
                
                # Calculate polynomial
                z = np.polyfit(xsub,ysub,1,w=weights)
                f = np.poly1d(z)
                
                #Calculate new x's and y's
                xfit = xsub
                yfit = f(xfit)
                
                #print(i)
                
                # Calculate and get stats on residual 
                res = ysub - yfit
                med_val = np.median(res[0:-1])
                mad_val = mad(res[0:-1])
                
                # Set values outside the range to nan
                if ~(med_val - 4*mad_val < res[-1] < med_val + 4*mad_val):
                    y[i-1] = np.nan
            
            elif ('z' in locals()) & (np.isfinite(y[i-int(1.5*span):i-int(0.5*span)]).sum() > 20): 
            # if there is a previous fit, use it when there is some data recently, but not enough for a good fit
                #Calculate new x's and y's using the previous z and f
                xfit = xsub
                yfit = f(xfit)
                
                #print(i)
                
    #            plt.plot(xsub,ysub,'o',xfit,yfit)
    #            plt.xlim([xsub[0],xsub[-1]])
    #            plt.show()
    #            plt.close()            
                
                idx = np.isfinite(ysub)
                
                res = ysub[idx] - yfit[idx]
                #med_val = np.median(res[0:-1])
                #mad_val = mad(res[0:-1])
                
                if ~(med_val - 4*mad_val < res[-1] < med_val + 4*mad_val):
                    y[i-1] = np.nan
                    
                
    # Append data to original data frame
    df[dfColumn+'_filt'] = y
    return df
    
    
    
    

#def linearFitFWDBWD_filt(df,
#                   dfColumn,
#                   data_time_resolution = 1, 
#                   data_time_resolution_units = 'sec',
#                   subset_span_period = 10,
#                   subset_span_period_units = 'sec'):
#    ''' Input data formats explained:
#        df:                         the dataframe containing the data you want to filter
#        dfColumn:                   the column within the dataframe you want to filter
#        data_time_resolution_val:   the value of the time resolution (e.g. "5" if resolution is 5 seconds)
#        data_time_resolution_units: the unit of the time resolution. Options of sec, min, hr, day
#        subset_span_period:         the period over which the linear fit is calculated. MUST be in the same units as time resolution and a multiple of val
#    '''
#                   
#
#    import numpy as np
#    import matplotlib.pyplot as plt
#    import pandas as pd
#    
#    # Define MAD calculation
#    mad = lambda x: np.median(np.abs(x - np.median(x)))
#    
#        
#    # Extract x and y vectors to fit
#    y = df[dfColumn].values.copy()
#    x = np.arange(len(y))
#    
#    mask = np.ones(len(y))
#    mask_BWD = np.ones(len(y))
#    mask_both = np.ones(len(y))
#    
#    # Calculate the span
#    # Convert everything to seconds except month and year
#    lst_units = [data_time_resolution_units,subset_span_period_units]
#    lst_vals = [data_time_resolution,subset_span_period]
#    for i in range(2):
#        if lst_units[i] == 'min':
#            lst_vals[i] = lst_vals[i]*60
#        if lst_units[i] == 'hr':
#            lst_vals[i] = lst_vals[i]*60*60
#        if lst_units[i] == 'day':
#            lst_vals[i] = lst_vals[i]*60*60*24
#    # Calculate the number of data points used in the calculation, i.e. span
#    if (np.mod(lst_vals[1], lst_vals[0]) == 0) & (lst_vals[1] > lst_vals[0]):
#        span = lst_vals[1]/lst_vals[0]
#    else:
#        return print("span period must be a multiple of the time resolution")
#        
#
#    # Weighting - more towards first half of fitting set, none on final point in question
#    w1 = np.ones(span/2)
#    w2 = np.ones(span/2-1)/2
#    w3 = np.zeros(1)
#    
#    for j in range(2):
#        
#        #Repeate twice, once going forward in time, once going backwards
#        if j>0:
#            # Reverse dataset on the second iteration
#            y = y[::-1] 
#        
#        for i in range(len(y)):
#        #for i in range(0,6920):
#        #i = 5900 # 5928
#            weights = np.concatenate((w1,w2,w3),axis=0)
#            
#            ysub = y[i-int(span):i].copy()
#            xsub = x[i-int(span):i].copy()
#            
#            if np.isfinite(y[i-1]): # only continue when the point in question is finite  
#                
#                if np.isfinite(ysub).sum() > 10: # only continue if there is sufficient data!
#                    
#                    # Remove nan's since poly fitting doesn't work on them!
#                    idx = np.isfinite(ysub)
#        
#                    ysub = ysub[idx]
#                    xsub = xsub[idx]
#                    weights = weights[idx]
#                    
#                    # Calculate polynomial
#                    z = np.polyfit(xsub,ysub,1,w=weights)
#                    f = np.poly1d(z)
#                    
#                    #Calculate new x's and y's
#                    xfit = xsub
#                    yfit = f(xfit)
#                    
#                    #print(i)
#                    
#                    # Calculate and get stats on residual 
#                    res = ysub - yfit
#                    med_val = np.median(res[0:-1])
#                    mad_val = mad(res[0:-1])
#                    
#                    # Set values outside the range to nan
#                    if ~(med_val - 4*mad_val < res[-1] < med_val + 4*mad_val):
#                        if j>0:
#                            mask_BWD[i-1] = np.nan
#                        else:
#                            mask[i-1] = np.nan
#                
#                elif ('z' in locals()) & (np.isfinite(y[i-int(1.5*span):i-int(0.5*span)]).sum() > 20): 
#                # if there is a previous fit, use it when there is some data recently, but not enough for a good fit
#                    #Calculate new x's and y's using the previous z and f
#                    xfit = xsub
#                    yfit = f(xfit)
#                    
#                    idx = np.isfinite(ysub)
#                    
#                    res = ysub[idx] - yfit[idx]
#
#                    
#                    if ~(med_val - 4*mad_val < res[-1] < med_val + 4*mad_val):
#                        if j>0:
#                            mask_BWD[i-1] = np.nan
#                        else:
#                            mask[i-1] = np.nan
#        del weights, ysub, xsub, idx, z, f, xfit, yfit, res, med_val, mad_val
#        if j>0:
#            # Reverse dataset back to original and mask_BWD
#            y = y[::-1] 
#            mask_BWD = mask_BWD[::-1]
#    
#    mask_both[np.isnan(mask) & np.isnan(mask_BWD)] = np.nan
#    
#    # Remove values where both directions yielded the values as an outlier
#    y[np.isnan(mask) & np.isnan(mask_BWD)] = np.nan
#           
#    # Append data to original data frame
#    df[dfColumn+'_filt'] = y
#    return df
    

#def MAD_filt(df,
#                   dfColumn,
#                   data_time_resolution = 5, 
#                   data_time_resolution_units = 'sec',
#                   subset_span_period = 10,
#                   subset_span_period_units = 'sec'):
#    ''' Input data formats explained:
#        df:                         the dataframe containing the data you want to filter
#        dfColumn:                   the column within the dataframe you want to filter
#        data_time_resolution_val:   the value of the time resolution (e.g. "5" if resolution is 5 seconds)
#        data_time_resolution_units: the unit of the time resolution. Options of sec, min, hr, day
#        subset_span_period:         the period over which the linear fit is calculated. MUST be in the same units as time resolution and a multiple of val
#       
#       Filters data based on the Median Absolute Deviation of the data 
#       CURRENTLY DOESN'T WORK - MAY NOTE ACTUALLY WORK
#
#    '''
#                   
#
#    import numpy as np
#    import matplotlib.pyplot as plt
#    import pandas as pd
#    
#    # Define MAD calculation
#    mad = lambda x: np.median(np.abs(x - np.median(x)))
#    
#        
#    # Extract x and y vectors to fit
#    y = df[dfColumn].values.copy()
#    mad_series = np.ones(len(y))
#    mad_series[:] = np.nan
#    
#    # Calculate the span
#    # Convert everything to seconds except month and year
#    lst_units = [data_time_resolution_units,subset_span_period_units]
#    lst_vals = [data_time_resolution,subset_span_period]
#    for i in range(2):
#        if lst_units[i] == 'min':
#            lst_vals[i] = lst_vals[i]*60
#        if lst_units[i] == 'hr':
#            lst_vals[i] = lst_vals[i]*60*60
#        if lst_units[i] == 'day':
#            lst_vals[i] = lst_vals[i]*60*60*24
#    # Calculate the number of data points used in the calculation, i.e. span
#    if (np.mod(lst_vals[1], lst_vals[0]) == 0) & (lst_vals[1] > lst_vals[0]):
#        span = lst_vals[1]/lst_vals[0]
#    else:
#        return print("span period must be a multiple of the time resolution")
#        
#        
#
#    for i in range(int(span/2),len(y)-int(span/2)):
#
#        ysub = y[i-int(span)/2:i+int(span)/2].copy()
#        
#        if np.isfinite(ysub).sum() > 3: # only continue if there is sufficient data!
#                                        
#            # Remove nan's since poly fitting doesn't work on them!
#            idx = np.isfinite(ysub)
#
#            ysub = ysub[idx]
#            
#            mad_series[i-int(span)/2:i+int(span)/2] = mad(ysub)
#
#    # Append data to original data frame
#    df['Running_MAD'] = mad_series
#    df = linearFitFWD_filt(df,'Running_MAD',data_time_resolution = 5, subset_span_period = 30, subset_span_period_units = 'min')
#    
#        
#    df[dfColumn+'_MADfilt'] = df[dfColumn].loc[np.isfinite(df['Running_MAD_filt'])]
#    #df[dfColumn+'_MADfilt'] = mad_series
#    return df



    
#import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
#import os
#import Filter_Timeseries as fTS
#CCN_path = 'c:/Dropbox/RuhiFiles/Research/Projects/2016_V02_CAPRICORN/Data_Raw/CCNC/'
#os.chdir(CCN_path)
#del CCN_path
#d = pd.read_hdf('CCN_UWY_5s_avg_filt.h5',key='CCN')
##d1 = d.iloc[0:10000,:]
##d2 = linearFitFWD_filt(d1,'CCN Number Conc',data_time_resolution = 5,subset_span_period = 5, subset_span_period_units = 'min')
#
#d1 = d.loc['2016-03-13 22:00:00':'2016-03-16 01:00:00']
#d2 = d1.copy()
#fTS.linearFit_filt(d,'CCN Number Conc',data_time_resolution = 5,subset_span_period = 5, subset_span_period_units = 'min')
#d3 = d2.copy()
##fTS.MAD_filt(d3,'CCN Number Conc_filt',data_time_resolution = 5,subset_span_period = 5, subset_span_period_units = 'min')
#
##d1['CCN Number Conc'].plot()
##d2['CCN Number Conc_filt'].plot()
#
#d3['CCN Number Conc_filt'].plot()
#d3['CCN Number Conc_filt_MADfilt'].plot()
#d3['Running_MAD_filt'].plot()
#d3['Running_MAD'].plot()
#plt.plot()