"""
Functions related to the loading and processing of CCNC data from DMT

version: 0.1
date: 2016-08-31
"""

def Load_to_HDF(DataPath, output_h5_filename = 'CCNC', resample_timebase = None):
    #To reload data from the h5 file
    import pandas as pd
    import os
    import glob
    import pickle
    os.chdir(DataPath)
    
    # Specify the column names once.
    colnames = [
                'Time', 'Current SS', 'Temps Stabilized', 'Delta T', 'T1 Set', 
                'T1 Read', 'T2 Set', 'T2 Read', 'T3 Set', 'T3 Read', 'Nafion Set', 
                'T Nafion', 'Inlet Set', 'T Inlet', 'OPC Set', 'T OPC', 'T Sample', 
                'Sample Flow', 'Sheath Flow', 'Sample Pressure', 'Laser Current', 
                'overflow', 'Baseline Mon', '1st Stage Mon', 'Bin #', 'Bin 1', 
                'Bin 2', 'Bin 3', 'Bin 4 ', 'Bin 5', 'Bin 6', 'Bin 7', 'Bin 8', 
                'Bin 9', 'Bin 10', 'Bin 11', 'Bin 12', 'Bin 13', 'Bin 14', 'Bin 15',
                'Bin 16', 'Bin 17', 'Bin 18', 'Bin 19', 'Bin 20', 'CCN Number Conc', 
                'Valve Set', 'Alarm Code', 'Alarm Sum'
                ]
    
    if os.path.isfile(output_h5_filename +'.h5'): #If previous file exists, append, if not start new
    
        data = pd.read_hdf(output_h5_filename +'.h5',key='CCN')
        
        with open('last_file_loaded.csv', 'rb') as f:
            last_file_loaded = pickle.load(f)
        #last_file_loaded = 'CCN 100 data 160326090000.csv' ####################
        filelist = glob.glob('*.csv')
        filelist.sort()
        
        filelist_index_start = filelist.index(last_file_loaded)
        filelist_index_final = len(filelist)-1
        
        filelist = filelist[filelist_index_start:filelist_index_final]
        
        del filelist_index_start, filelist_index_final, last_file_loaded
        
            
        for i in range(0,len(filelist)):
                # Load csv data        
                data_temp = pd.read_csv(filelist[i], 
                            names = colnames, 
                            skiprows = range(0,6), 
                            engine='python',
                            skipinitialspace = True, 
                            usecols=range(49)
                            )  
                # Read date from csv file 
                date_temp = pd.read_csv(filelist[i], names = ['label', 'date'], skiprows = range(2,len(data_temp)+6))['date'][1]
                
                # Use the read date and replace the time column with timestamps.                   
                for j in range(0, len(data_temp)):
                        time_temp = str(data_temp['Time'][j])
                        data_timestamp = pd.to_datetime(date_temp + ' ' + time_temp)
                        data_temp = data_temp.replace(to_replace=data_temp['Time'][j], value=data_timestamp)
                
                data_temp = data_temp.set_index('Time')        
                #Append new csv file data to existing dataframe
                data = data.append(data_temp)#, ignore_index=True) 
                
    
    else:
        filelist = glob.glob('*.csv')
        filelist.sort()
        #Initialise
        data = pd.read_csv(filelist[0], 
                            names = colnames, 
                            skiprows = range(0,6), 
                            engine='python',
                            skipinitialspace = True, 
                            usecols=range(49)
                            )
        # Read date from csv file 
        date_temp = pd.to_datetime(pd.read_csv(filelist[0], names = ['label', 'date'], skiprows = range(2,len(data)+6))['date'][1], format = "%m/%d/%y")
        
        # Use the read date and replace the time column with timestamps.
        for j in range(0,len(data)):
            data_timestamp = pd.to_datetime(date_temp + ' ' + data['Time'][j])
            data = data.replace(to_replace=data['Time'][j], value=data_timestamp)
        
        
        
        for i in range(1,len(filelist)):
                # Load csv data        
                data_temp = pd.read_csv(filelist[i], 
                            names = colnames, 
                            skiprows = range(0,6), 
                            engine='python',
                            skipinitialspace = True, 
                            usecols=range(49)
                            )  
                # Read date from csv file 
                date_temp = pd.read_csv(filelist[i], names = ['label', 'date'], skiprows = range(2,len(data_temp)+6))['date'][1]
                # Use the read date and replace the time column with timestamps.        
                        
                for j in range(0, len(data_temp)):
                        time_temp = str(data_temp['Time'][j])
                        data_timestamp = pd.to_datetime(date_temp + ' ' + time_temp)
                        data_temp = data_temp.replace(to_replace=data_temp['Time'][j], value=data_timestamp)
        
                #Append new csv file data to existing dataframe
                data = data.append(data_temp)#, ignore_index=True) 
                
                # Save new data to file
                fname_current = 'CCNC_noIndex_temp_'+str(i)+'of'+str(len(filelist))+'.h5'
                data.to_hdf(fname_current, key='CCN')
                # Remove the temporary file    
                if os.path.isfile('CCNC_noIndex_temp_'+str(i-1)+'of'+str(len(filelist))+'.h5'):
                    os.remove('CCNC_noIndex_temp_'+str(i-1)+'of'+str(len(filelist))+'.h5')
            
    del  colnames, data_temp, i, j, data_timestamp
    
        
    # Change the index to the timestamp
    data = data.set_index('Time')
    
    #Remove duplicates
    data = data.drop_duplicates()
    
    # Sort data by ascending time
    data = data.sort_index()
    
    
    data.to_hdf(output_h5_filename +'.h5', key='CCN')
    os.remove(fname_current)
    
    #Save the last filenme loaded to file for next update
    last_file_loaded = filelist[len(filelist)-1]
    with open('last_file_loaded.csv', 'wb') as f:
        pickle.dump(last_file_loaded, f)
    
    if resample_timebase is not None:
        resample_timebase(data)    
    
    return None

def resample_timebase(data=0,RawDataPath='',input_h5_filename='',variable='CCN',time_int='default'):
    ### Time resampling

    import pandas as pd
    import numpy as np
    import os
    
    if not isinstance(data, pd.DataFrame): #if no data provided, try to load from file
        if (not RawDataPath == '') & (not input_h5_filename == ''):
            os.chdir(RawDataPath)
            if os.path.isfile(input_h5_filename+'.h5'): 
                data = pd.read_hdf(input_h5_filename+'.h5', key=variable)
        else:
            print("Please input either a dataframe or a datapath and filename where data can be found")
            return
    
    
    # define time resampling intervals unless specified in function input
    if time_int == 'default':
        time_int = ['5S','1Min', '5Min', '10Min', '30Min', '1H', '3H', '6H', '12H', '1D']    
    
    
    # define MAD calculation
    mad = lambda x: np.fabs(x - x.median()).median() 
    
    # define time resampling intervals
    i_lim = len(time_int)
    sub = data.iloc[:,24:44].copy()
    sub_ccn = data['CCN Number Conc'].copy()
    for i in range(0, i_lim):
        t_int = time_int[i]    
        # Initialise    
        data_resamp = sub.resample(t_int,fill_method=None).median()
        data_resamp['ccn_count'] = sub_ccn.resample(t_int,fill_method=None).count()
        data_resamp['ccn_med'] = sub_ccn.resample(t_int,fill_method=None).median()
        data_resamp['ccn_mad'] = sub_ccn.resample(t_int,fill_method=None).apply(mad)
        data_resamp['ccn_avg'] = sub_ccn.resample(t_int,fill_method=None).mean()
        data_resamp['ccn_std'] = sub_ccn.resample(t_int,fill_method=None).std()
        
        # Rename the cloud droplet bins so they make sense when the full data is merged
        data_resamp.rename(columns={'Bin 1': 'CDN Bin 1',
                                    'Bin 2': 'CDN Bin 2',
                                    'Bin 3': 'CDN Bin 3',
                                    'Bin 4': 'CDN Bin 4',
                                    'Bin 5': 'CDN Bin 5',
                                    'Bin 6': 'CDN Bin 6',
                                    'Bin 7': 'CDN Bin 7',
                                    'Bin 8': 'CDN Bin 8',
                                    'Bin 9': 'CDN Bin 9',
                                    'Bin 10': 'CDN Bin 10',
                                    'Bin 11': 'CDN Bin 11',
                                    'Bin 12': 'CDN Bin 12',
                                    'Bin 13': 'CDN Bin 13',
                                    'Bin 14': 'CDN Bin 14',
                                    'Bin 15': 'CDN Bin 15',
                                    'Bin 16': 'CDN Bin 16',
                                    'Bin 17': 'CDN Bin 17',
                                    'Bin 18': 'CDN Bin 18',
                                    'Bin 19': 'CDN Bin 19',
                                    'Bin 20': 'CDN Bin 20'},
                           inplace=True)
        
        # Save to file
        if isinstance(data,pd.DataFrame):        
            outputfilename = variable+'_'+time_int[i]+'.h5'
        else:
            outputfilename = input_h5_filename+'_'+ time_int[i] +'.h5'
        data_resamp.to_hdf(outputfilename, key=variable)
    
    return data_resamp
    
    
    
    
def DataQC(CCN_data, 
           FlowRatio=10.0,
           T1diffLim=0.25,
           T2diffLim=0.25,
           T3diffLim=0.15,
           NafionTdiffLim=0.3,
           OPCT1diffLim=1
           ):
    """
    Filter data that is out of spec. 
    """    
    import numpy as np
    
    
    ### Flag data to have a closer look at.

    CCNC_data = CCN_data.copy()
    
    # Initialise
    
#    ReviewData = pd.DataFrame(np.nan,index=CCNC_data.index,columns=['review'])
#    ReviewData['CCN Number Conc'] = CCNC_data['CCN Number Conc'].copy()
#    # Concentration lower than 10 /cm3 (as per factory setting)
#    #Data4CloserLook['check'].loc[CCNC_data['CCN Number Conc'] >= 10] = np.nan    
#    ReviewData.loc[CCNC_data['CCN Number Conc'] < 10] = -999

    
    ### Filter primary dataset
    with np.errstate(invalid = 'ignore'): # Ignore error warnings caused by arithmetic on nans
    
        # Alarms detected by software - note many of these only activate after a certain time period
        CCNC_data.loc[CCNC_data['Alarm Code'] > 0] = np.nan
    
        # Concentration lower than 10 /cm3 (as per factory setting)
        CCNC_data.loc[CCNC_data['CCN Number Conc'] < 10] = np.nan
        
        # Flow ratio outside 10 +/- 0.4
        CCNC_data['Flow Ratio'] = CCNC_data['Sheath Flow'] / CCNC_data['Sample Flow']
        CCNC_data.loc[CCNC_data['Flow Ratio'] > (FlowRatio + 1)]= np.nan
        CCNC_data.loc[CCNC_data['Flow Ratio'] < (FlowRatio - 1)]= np.nan
        
        # Irrelevant SuperSaturation values
        CCNC_data.loc[CCNC_data['Current SS'] < 0] = np.nan
        
        # 80 < Laser current < 120
        CCNC_data.loc[CCNC_data['Laser Current'] > 120] = np.nan
        CCNC_data.loc[CCNC_data['Laser Current'] <  80] = np.nan
        
        # 1st Stage Mon > 4.7 V 
        CCNC_data.loc[CCNC_data['Laser Current'] > 120] = np.nan    
        
        # Temperatures deviating from their setpoints       
        CCNC_data.loc[abs(CCNC_data['T1 Set'] - CCNC_data['T1 Read']) > T1diffLim] = np.nan
        CCNC_data.loc[abs(CCNC_data['T2 Set'] - CCNC_data['T2 Read']) > T2diffLim]= np.nan
        CCNC_data.loc[abs(CCNC_data['T3 Set'] - CCNC_data['T3 Read']) > T3diffLim]= np.nan
        CCNC_data.loc[abs(CCNC_data['Nafion Set'] - CCNC_data['T Nafion']) > NafionTdiffLim]= np.nan
        CCNC_data.loc[abs(CCNC_data['OPC Set'] - CCNC_data['T OPC']) > 1]= np.nan 
        
    return CCNC_data#, ReviewData)


    
def uwy_filter(uwy_merge_data,
               uwy_path='c:/Dropbox/RuhiFiles/Research/Projects/2016_V02_CAPRICORN/Data_Raw/Underway/'):
    import numpy as np
    import pandas as pd
    if 'mask' not in uwy_merge_data.columns:
        import RVI_Underway
        RVI_Underway.create_uwy_masks(uwy_path,apply_mask_to_create_filt_dataset=False)
        
    uwy_merge_data.loc[pd.isnull(uwy_merge_data['mask'])] = np.nan

    return uwy_merge_data

def flow_cal(data, measured_flows_df, set_flow_rate, polydeg=2):
    ''' Calibrates CPC_data for measured flow rates.
    data - dataframe of raw CPC data
    measured_flows_df - a dataframe of the times and measured flow rates used for calibration. See CAPRICORN.py for an example
    set_flow_rate - the flow rate that the instrument SHOULD be at.
    polydeg - the degree of the polynomial to fit to the measured data and correct with.
    '''
    import numpy as np
    import pandas as pd
    # Convert dates to seconds since 1 Jan 2000
    x = (measured_flows_df.index - pd.to_datetime('2000-01-01 00:00:00')).total_seconds()
    y = measured_flows_df['flow rate']
    p = np.poly1d(np.polyfit(x,y,deg=polydeg))
    x_data = (data.index - pd.to_datetime('2000-01-01 00:00:00')).total_seconds()

    data['CCN Number Conc'] = data['CCN Number Conc']/set_flow_rate*p(x_data)
    #plt.plot(x,y,'.',xp,p(xp),'--')
    
    return data
    
def ccn_stat_filt(data, std_lim = 150, removeData = False):
    import numpy as np
    if removeData:
        data.loc[data['ccn_std'] > std_lim] = np.nan
    else:
        data.loc[data['ccn_std'] > std_lim,'ccn_std_mask'] = np.nan
    return data
    
    
    
def LoadAndProcess(CCN_path, 
                   filename_base = 'CCN', 
                   filtOrRaw='filt', 
                   timeResolution='5S',
                   mask_period_timestamp_list = [''],
                   CCN_flow_check_df = '',
                   CCN_flow_setpt = 500,
                   CCN_flow_polyDeg = 2
                   ):
    import pandas as pd
    import numpy as np
    import os
    os.chdir(CCN_path)
    
    import glob
    # Check if any h5 file has been produced yet (ie. the initial processing has occurred)
    if not glob.glob('*.h5'):
        Load_to_HDF(CCN_path,filename_base)
        
    filename = filename_base+'_'+timeResolution+'.h5'
    filename_1sec = filename_base+'_'+filtOrRaw+'.h5'
    filename_raw = 'CCNC.h5'
#
    
    if filtOrRaw.lower() == 'filt':
        if os.path.isfile(filename):
            ccn = pd.read_hdf(filename,key=filename_base)
            NeedsTimeResampling = False
            ccn = ccn_stat_filt(data = ccn, std_lim = 150, removeData=True)
            return ccn
        elif os.path.isfile(filename_1sec):
            ccn = pd.read_hdf(filename_1sec,key=filename_base)
            NeedsTimeResampling = True 
        return ccn
    elif filtOrRaw.lower() == 'raw':
        if os.path.isfile(filename_1sec):
            ccn = pd.read_hdf(filename_1sec,key=filename_base)
        elif os.path.isfile(filename_raw):
            ccn = pd.read_hdf(filename_raw,key=filename_base)
        if os.path.isfile(filename): # Return the raw file if resampling has already been done
            return ccn
        else:
            NeedsTimeResampling = True
    else:
        print("No hdf file exists with the raw data! Please run the following function before this one: CCNC.Load_to_HDF")
        return
        
    filt = False #Initialise
    if filtOrRaw.lower() == 'raw':
        # QC based on instrument parameters
        ccn = DataQC(ccn)
        
        # Flow calibrations
        if (not CCN_flow_check_df == ''):
            ccn = flow_cal(ccn,CCN_flow_check_df,CCN_flow_setpt,polydeg=CCN_flow_polyDeg)
            filt = True
        # work through mask periods and set values to nan
        for i in range(int(len(mask_period_timestamp_list)/2)):
            ccn.loc[(ccn.index >= mask_period_timestamp_list[2*i]) & (ccn.index < mask_period_timestamp_list[2*i+1])]= np.nan
            filt = True
        # Save to file as 1 second filtered data  
        if filt:
            ccn.to_hdf(filename_base+'_filt.h5',key = filename_base)
    else:
        print("Don't know what to load. Please specify either Raw or Filt")
        return
    
    
    if NeedsTimeResampling:     
        #Resample to 5 second time base, then join with UWY dataset
        ccn = resample_timebase(data = ccn, variable = filename_base, time_int=[timeResolution])   
        
        ccn = ccn_stat_filt(data = ccn, std_lim = 150, removeData=True)
        
    
    return ccn