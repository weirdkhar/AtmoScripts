"""
Function related to the loading and processing of CPC instruments from TSI

version: 0.0
date: 2016-09-09
"""
import pandas as pd
import os
import glob
import pickle


def Load_to_HDF(RawDataPath,
                output_h5_filename = 'CPC_sec', 
                InputTZ = 0, 
                OutputTZ = 0, 
                resample_time = False, 
                output_file_frequency = 'all'
                ):
    """
    Load performed after data has been exported to CSV file with just raw concentrations and times.
    """

    os.chdir(RawDataPath)
    
    filelist = glob.glob('*.csv')
    # Check if previous data has already been loaded, if so, don't load it again
    if os.path.isfile('files_loaded.txt'):
        with open('files_loaded.txt', 'rb') as f:
            files_already_loaded = pickle.load(f)
        # Get only the new files to be loaded:
        filelist = list(set(filelist).difference(set(files_already_loaded)))
    filelist.sort()       
    
    #Iterate through to load the raw files 
    for file in filelist:
        # Read cpc csv file
        read_cpc_csv(file, output_h5_filename, output_file_frequency, InputTZ, OutputTZ)  
    
    #Save the files that have already been loaded to file for next update
    with open('files_loaded.txt', 'wb') as f:
        try:
            files_already_loaded
        except NameError:
            filelist = filelist
        else:
            filelist = filelist + files_already_loaded
        pickle.dump(filelist, f)

    if resample_time:    
        resample_timebase(RawDataPath, output_h5_filename,variable = output_h5_filename,time_int=['5S'])
    
    return None

def save_to_hdf(data, output_h5_filename, output_file_frequency):    
    import datetime
    ''' Determine the destination file for each datapoint in the dataframe'''
    #year_str = [datetime.datetime.strftime(i, format = '%Y') for i in data.index]
    year_str = [str(i.isocalendar()[0]) for i in data.index]
    #mnth_str = [datetime.datetime.strftime(i, format = '%m') for i in data.index]
    mnth_str = ['0' + str(i.month) \
                if i.month <10 \
                else str(i.month) \
                for i in data.index]
    
    
    if output_file_frequency.lower() == 'monthly':
#        print('Saving to monthly HDF files')
        
        # Identify the destination file of each data point
        data['destination_file'] = [output_h5_filename+'_'+x+y for x,y in zip(year_str,mnth_str)]
        # Get the unique filenames
        output_filelist = set(data['destination_file'])
    
    elif output_file_frequency.lower() == 'weekly':
 #       print('Saving to weekly HDF files')
        
        wk_str = ['0'+str(i.isocalendar()[1]) \
                          if i.isocalendar()[1] < 10 \
                          else str(i.isocalendar()[1]) \
                          for i in data.index]
        
        # Identify the destination file of each data point
        data['destination_file'] = [output_h5_filename+'_'+x+'_wk'+y for x,y in zip(year_str,wk_str)]
        # Get the unique filenames
        output_filelist = set(data['destination_file'])
        
    elif output_file_frequency.lower() == 'all':
        # Continue as normal
#        print('Saving all data to a single HDF file')
        data['destination_file'] = output_h5_filename
    
    else:
#        if output_file_frequency.lower() == 'daily':
#            print('Saving to daily HDF files')
#        else:
        if output_file_frequency.lower() != 'daily':
            print("Cannot determine what frequency you want the output file, defaulting to saving to daily files.")
        
        day_str = [datetime.datetime.strftime(i, format = '%d') for i in data.index]
        
        # Identify the destination file of each data point
        data['destination_file'] = [output_h5_filename+'_'+x+y+z for x,y,z in zip(year_str,mnth_str,day_str)]
        # Get the unique filenames
        output_filelist = set(data['destination_file'])  
    
    
    ''' Save the appropriate data to each destination file '''
    for file in list(output_filelist):
        # Select data
        data_temp = data[data['destination_file']==file]
        # Delete extraneous columns
        del data_temp['destination_file']
        
        # Check if file already exists, if so, append, otherwise, create a new file
        if os.path.isfile(file+'.h5'):
            data_saved = pd.read_hdf(file+'.h5', key = output_h5_filename)
            data_temp = data_saved.append(data_temp)
            # Drop any duplicates which may be there, based only on the Timestamp
            data_temp = data_temp.reset_index().drop_duplicates(subset='Timestamp', keep='last')
            data_temp = data_temp.set_index('Timestamp')
            
            #os.remove(file+'.h5')
        # Save to file
        data_temp.to_hdf(file+'.h5', key = output_h5_filename, mode='a')
        
    
    # Remove additional columns that were added to the dataframe in the processing
    del data['destination_file']
    
    return file+'.h5'
    
def read_cpc_csv(read_filename, output_filename_base, output_file_frequency, InputTZ=0, OutputTZ=0):
    # Reads CPC data exports from AIM 10 and higher as row based, with ONLY concentration data output
    import numpy as np
    
    if output_file_frequency == 'all':
        print('Saving all data to a single HDF file')
    else:
        print('Saving to ' + output_file_frequency + ' HDF file')
    
        
    # Read each row of data, taking into account that each row can change length and parsing format (weird...)
    df = pd.read_csv(read_filename, skiprows = range(0,3), engine='python', skipinitialspace=True, iterator = True, chunksize = 1000)
    
    # Read the number of samples in the file
    with open(read_filename) as f: 
        lastline = f.readlines()[-1]
    numsamples = lastline.split(",")[0]
    
    for chunk in df:
        # Extract initial timestamp for each sample (i.e. each row)
        try:
            chunk['sample_timestamp'] = pd.to_datetime(chunk['Start Date']+' '+chunk['Start Time'], format = '%m/%d/%y %H:%M:%S')
        except KeyError:
            # The csv file that you've read isn't actually a TSI CPC file
            return
        
        data = pd.DataFrame(columns = {'Timestamp', 'Concentration'})
        for rowidx in range(0,len(chunk)):
            # Create timestamp and extract concentration for each sample in chunk
            timestamp = [chunk['sample_timestamp'][rowidx] + pd.Timedelta(seconds=x) for x in range(0, chunk['Sample Length'][rowidx])]
            conc = chunk.iloc[rowidx,12:(12+chunk['Sample Length'][rowidx])]
            
            print('Formatting sample ' + str(chunk['Sample #'].iloc[rowidx]) 
                  + ' of ' + numsamples + ' from file ' + read_filename)
            
            # Format data as dataframe
            data_temp = pd.DataFrame({'Timestamp': timestamp, 'Concentration': conc.values})
            # Append new data to current data
            data = pd.concat([data,data_temp])
      
        # Drop duplicates that may be present
        data = data.drop_duplicates(subset='Timestamp', keep='last')
        # Set index
        data = data.set_index('Timestamp')
        # Coerce data to the correct type, dealing with infinite values output from AIM
        data['Concentration'] = [np.nan if x == '1.#INF'  else float(x) for x in data['Concentration']]
        
        #Correct for Timezone offsets caused by AIM exporting process
        if InputTZ-OutputTZ != 0 :
            data = TimeZoneCorrection(data, CurrentTZ = InputTZ, OutputTZ = OutputTZ)
        
        # Save to hdf file
        print('Saving chunk to file')
        outputfilename = save_to_hdf(data, output_filename_base, output_file_frequency)

        # Alert the user where the process is up to
        print('Data loaded from ' + read_filename +' and samples ' + str(chunk['Sample #'].iloc[0]) 
                    + ' to ' + str(chunk['Sample #'].iloc[-1]) + ' saved to ' + outputfilename)

    return
    
def read_cpc_csv_row_AIM9(read_filename, output_filename_base, output_file_frequency, InputTZ=0, OutputTZ=0):
	# function to read row based cpc exported data from AIM versions 9 or below. 
	# This is VERY slow due to the exported format from AIM 
	#(error messages from sampling accumulate with each sample, causing the parsing of the data to slow down as it progresses).
	
    import datetime
    
    if output_file_frequency == 'all':
        print('Saving all data to a single HDF file')
    else:
        print('Saving to ' + output_file_frequency + ' HDF file')
    
    data_present = True
    i = 0
    while data_present:
        
        # Read each row of data, taking into account that each row can change length and parsing format (weird...)
        rows = pd.read_csv(read_filename, skiprows = range(0,3+i), engine='python', skipinitialspace=True, nrows=1)
        
        if len(rows)>0: # haven't yet reached the end of the file   
            # The reading of the rows is random, so need a few different techniques for reading
            if type(rows.index) != pd.indexes.multi.MultiIndex:
                # Extract initial timestamp
                sample_timestamp = pd.to_datetime(rows.iloc[0,0]+' '+rows.iloc[0,1], format = '%m/%d/%y %H:%M:%S')
                # Extract sample length
                sample_length = int(rows.iloc[0,2])
                # Extract sample number
                samplenum = rows.index[0]
            
            elif len(rows.index[0]) == 2:
                sample_timestamp = pd.to_datetime(rows.index[0][1]+' '+rows.iloc[0,0], format = '%m/%d/%y %H:%M:%S')
                sample_length = int(rows.iloc[0,1])
                samplenum = rows.index[0][0]
            elif len(rows.index[0]) == 3:
                sample_timestamp = pd.to_datetime(rows.index[0][1]+' '+rows.index[0][2], format = '%m/%d/%y %H:%M:%S')
                sample_length = int(rows.iloc[0,0])
                samplenum = rows.index[0][0]
            elif len(rows.index[0]) >= 4:
                sample_timestamp = pd.to_datetime(rows.index[0][1]+' '+rows.index[0][2], format = '%m/%d/%y %H:%M:%S')
                sample_length = int(rows.index[0][3])
                samplenum = rows.index[0][0]
      
            # Make sample timestamp array
            timestamps = [sample_timestamp + datetime.timedelta(seconds=x) for x in range(0, sample_length)]
            # Remove all non-numeric data (e.g. errors)
            data_num = rows.select_dtypes(include=['number'])
            # Remove all additional data that is output in the file
            conc = data_num.iloc[:,8:(data_num.shape[1]-1)]
            # Remove any nans
            conc = conc.dropna(axis=1)
                        
            
            # Create dataframe with the extracted data
            sample_data = {'Timestamp' : pd.Series(timestamps), 'Concentration' : pd.Series(conc.values[0,:])}
            sample_data = pd.DataFrame(sample_data)
            sample_data = sample_data.set_index('Timestamp')
    
            #Correct for Timezone offsets caused by AIM exporting process
            if InputTZ-OutputTZ != 0 :
                sample_data = TimeZoneCorrection(sample_data, CurrentTZ = InputTZ, OutputTZ = OutputTZ)
            
            save_to_hdf(sample_data, output_filename_base, output_file_frequency)
            
            print('Successfully saved sample ' + str(samplenum) + ' to file.')
            
            i += 1 #iterate
      
        else:
            data_present=False       

    return

def read_cpc_csv_column(read_filename, output_filename_base, output_file_frequency):
    # function to read column based cpc exported data from AIM versions 9 or below. 
	# This is slow - preferencially use the row based function 
	
    colnames = ['Timestamp','Concentration'] #Instantiate    
    
    # Load date
    date_row = pd.read_csv(read_filename,
                           nrows = 1,
                           skiprows = range(0,4)
                           )
    
    # Find the number of samples in the file
    temp = pd.read_csv(read_filename,nrows = 1,skiprows = range(0,2))
    j_lim = ''
    k = 1
    while isinstance(j_lim,str) & (k > 0):
        if 'Unnamed' not in temp.columns[-k]:
            j_lim = int(temp.columns[-k])
        k = k+1
    
    del temp

    #Iterate through each sample            
    for j in range(1, j_lim+1): 
    
        date_str = date_row.columns[2*j-1].split('.')[0]
        #date = pd.to_datetime(date_str)        
        
        
        data_temp = pd.read_csv(read_filename, 
                                names = colnames, 
                                skiprows = range(0,18),
                                nrows = 298, # ignore comments at the end of the file
                                engine='python',
                                skipinitialspace = True, 
                                iterator=True,
                                usecols=[2*j-2,2*j-1]#range(2*j-2,2*j)
                                )
        
        # Get the index of the last data point in the column, i.e. ignoring the comments at the end of the file
        #k_end = data_temp[data_temp.Concentration.isnull()].index[0]
        #data_temp = data_temp[0:k_end]
        
        data_temp['Timestamp'] = pd.to_datetime(date_str+ ' ' + data_temp['Timestamp'])
        
        # Increment the date when the time moves over midnight
        for i in range(0, len(data_temp)):
            if data_temp['Timestamp'][i] < data_temp['Timestamp'][0]:
                data_temp['Timestamp'][i] = data_temp['Timestamp']+pd.Timedelta(days=1)

        #Set index to Timestamp
        data_temp = data_temp.set_index('Timestamp') 
        
        save_to_hdf(data_temp, output_filename_base, output_file_frequency)
        
        print('Successfully saved sample ' + str(j) + ' to file.')
        
        del data_temp

    
    return #data

def TimeZoneCorrection(DataFrame, CurrentTZ, ConvertToUTC = True, OutputTZ = 0):
    import pandas as pd
    
    if ConvertToUTC:
        OutputTZ= 0
    elif OutputTZ == 0:
        print("You must define your output timezone if its not UTC")
        return        
    
    # Correct for any timezone offsest
    DataFrame = DataFrame.shift(freq=pd.Timedelta(hours=OutputTZ-CurrentTZ))
    
    return DataFrame


def resample_timebase(data=0,RawDataPath='',input_h5_filename='',variable='CN',time_int='default'):
    ### Time resampling

    import pandas as pd
    import numpy as np
    import os
    
    if not isinstance(data, pd.DataFrame): #if no data provided, try to load from file
        if (not RawDataPath == '') & (not input_h5_filename == ''):
            os.chdir(RawDataPath)
            if os.path.isfile(input_h5_filename): 
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
    for i in range(0, i_lim):
        t_int = time_int[i]    
        # Initialise    
        data_resamp = data['Concentration'].resample(t_int,fill_method=None, how=['count'])
        data_resamp.rename(columns={'count' : variable+'_count'},inplace=True)
        # Resample with stats    
        data_resamp[variable+'_median'] = data.resample(t_int,fill_method=None).median()
        data_resamp[variable+'_mad'] = data.resample(t_int,fill_method=None).apply(mad)
        data_resamp[variable+'_mean'] = data.resample(t_int,fill_method=None).mean()
        data_resamp[variable+'_std'] = data.resample(t_int,fill_method=None).std()
        
        #del data_resamp['Concentration']
        
        # Save to file
        if isinstance(data,pd.DataFrame):        
            outputfilename = variable+'_'+time_int[i]+'.h5'
        else:
            outputfilename = input_h5_filename+'_'+ time_int[i] +'.h5'
        data_resamp.to_hdf(outputfilename, key=variable)
    
    return data_resamp

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

    data['Concentration'] = data['Concentration']/set_flow_rate*p(x_data)
    #plt.plot(x,y,'.',xp,p(xp),'--')
    
    return data
    
def LoadAndProcess(CN_path, 
                   filename_base = 'CN3', 
                   filtOrRaw='filt', 
                   timeResolution='',
                   mask_period_timestamp_list = [''],
                   CurrentTZ = 0, 
                   OutputTZ = 0,
                   CN_flow_check_df = '',
                   CN_flow_setpt = 1500,
                   CN_flow_polyDeg = 2
                   ):
    import pandas as pd
    import numpy as np
    import os
    import re
    import glob
    os.chdir(CN_path)
    NeedsTZCorrection=True #Initialise
    
    # Check if any h5 file has been produced yet (ie. the initial processing has occurred)
    if not glob.glob('*.h5'):
        Load_to_HDF(CN_path,filename_base, InputTZ = CurrentTZ, OutputTZ = OutputTZ)
    
    filename = filename_base+'_'+timeResolution+'.h5'
    filename_1sec = filename_base+'_'+filtOrRaw+'.h5'

    if filtOrRaw.lower() == 'filt':
        if os.path.isfile(filename):
            CN = pd.read_hdf(filename,key=filename_base)
            NeedsTimeResampling = False
            return CN
        elif os.path.isfile(filename):
            CN = pd.read_hdf(filename_1sec,key=filename_base)
            NeedsTimeResampling = True   
            return CN
        else:
            filtOrRaw = 'raw' # filt file not available, produce it!
            filename_1sec = filename_base+'_'+filtOrRaw+'.h5'
            
    if filtOrRaw.lower() == 'raw':
        if os.path.isfile(filename_1sec):
            CN = pd.read_hdf(filename_1sec,key=filename_base)
            NeedsTimeResampling = False
            NeedsTZCorrection = False #Only correct for TZ when creating the H5 file for the first time. 
        if os.path.isfile(filename): # Return the raw file if resampling has already been done
            return CN
        else:
            NeedsTimeResampling = True
    else:
        print("No hdf file exists with the raw data! Please run the following function before this one: CPC_TSI.Load_to_HDF(RawDataPath,output_h5_filename = 'CPC', TZCorrect = False, InputTZ = 0, OutputTZ = 0,resample_time = False)")
        return
    
    # Correct timezone if necessary
    if NeedsTZCorrection:
        if CurrentTZ - OutputTZ != 0:
            if OutputTZ == 0:
                ToUTC = True
            else:
                ToUTC = False
            CN = TimeZoneCorrection(CN, CurrentTZ, ConvertToUTC = ToUTC, OutputTZ = 0)


    NeedsFiltering = False #Initialise
    if filtOrRaw.lower() == 'raw':
        # Flow calibrations
        if not isinstance(CN_flow_check_df,str):
            CN = flow_cal(CN,CN_flow_check_df,CN_flow_setpt,polydeg=CN_flow_polyDeg)
            NeedsFiltering = True
        
        # work through mask periods and set values to nan
        for i in range(int(len(mask_period_timestamp_list)/2)):
            CN.loc[(CN.index >= mask_period_timestamp_list[2*i]) & (CN.index < mask_period_timestamp_list[2*i+1])]= np.nan
            NeedsFiltering = True
        # Save to file as 1 second filtered data
        if NeedsFiltering:
            CN.to_hdf(filename_base+'_filt.h5',key = filename_base)
    else:
        print("Don't know what to load. Please specify either Raw or Filt")
        return

    
        
    if NeedsTimeResampling & (not timeResolution == ''):     
        # Check if current time resolution is what is being asked for, if not, resample. If so, floor to nearest interval
        current_time_res = (CN.index[1] - CN.index[0]).seconds
        if any(substring in timeResolution for substring in ['S', 'sec', 'Sec']):
            if int(re.findall('\d+', timeResolution)[0]) == current_time_res:
                if 'Concentration' in CN:
                    CN.rename(columns={'Concentration' : filename_base.lower()}, inplace=True)
                ns = 1*1*1000000000 # 1 second in nanoseconds
                CN.index = pd.DatetimeIndex(((CN.index.astype(np.int64) // ns + 1) * ns - ns ))
                return CN
        elif any(substring in timeResolution.lower() for substring in ['min']):
            if int(re.findall('\d+', timeResolution)[0]) == current_time_res/60:
                if 'Concentration' in CN:
                    CN.rename(columns={'Concentration' : filename_base.lower()}, inplace=True)
                ns = 1*60*1000000000 # 1 minute in nanoseconds
                CN.index = pd.DatetimeIndex(((CN.index.astype(np.int64) // ns + 1) * ns - ns))
                return CN
        elif any(substring in timeResolution for substring in ['H', 'Hr', 'hr', 'Hour', 'hour']):
            if int(re.findall('\d+', timeResolution)[0]) == current_time_res/60/60:
                if 'Concentration' in CN:
                    CN.rename(columns={'Concentration' : filename_base.lower()}, inplace=True)
                ns = 60*60*1000000000 #60 minutes in nanoseconds
                CN.index = pd.DatetimeIndex(((CN.index.astype(np.int64) // ns + 1) * ns - ns))
                return CN
        elif any(substring in timeResolution for substring in ['D', 'day', 'Day']):
            if int(re.findall('\d+', timeResolution)[0]) == current_time_res/60/60/24:
                if 'Concentration' in CN:
                    CN.rename(columns={'Concentration' : filename_base.lower()}, inplace=True)
                ns = 24*60*60*1000000000 # 1 day in nanoseconds
                CN.index = pd.DatetimeIndex(((CN.index.astype(np.int64) // ns + 1) * ns - ns))
                return CN
        CN = resample_timebase(data = CN, variable = filename_base, time_int=[timeResolution])    
        
    return CN
    